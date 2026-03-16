import argparse
import logging
import threading
from typing import Optional

import gradio as gr
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from internvlu import InternVLUPipeline


_PIPELINE: Optional[InternVLUPipeline] = None
_PIPELINE_DEVICE = "cuda"
_PIPELINE_LOCK = threading.Lock()
_PIPELINE_DTYPE = torch.float32
_LOGGER = logging.getLogger("internvlu.app")
_CANCEL_EVENT = threading.Event()
_REQUEST_RUNNING = threading.Event()
_AUTO_IMAGE_MAX_EDGE = 768
_AUTO_IMAGE_MAX_AREA = 768 * 768
_AUTO_IMAGE_MULTIPLE = 32


def _configure_logging() -> None:
    if _LOGGER.handlers:
        return

    _LOGGER.setLevel(logging.INFO)
    handler = logging.FileHandler("app.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    _LOGGER.addHandler(handler)
    _LOGGER.propagate = False


_configure_logging()


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def _resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def _decode_text_output(output, tokenizer) -> str:
    if output is None:
        return ""

    token_ids = output
    if hasattr(token_ids, "sequences"):
        token_ids = token_ids.sequences

    if isinstance(token_ids, torch.Tensor):
        if token_ids.ndim > 1:
            token_ids = token_ids[0]
        return tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    if isinstance(token_ids, (list, tuple)) and len(token_ids) > 0:
        first = token_ids[0]
        if isinstance(first, torch.Tensor):
            return tokenizer.decode(first, skip_special_tokens=True).strip()
        return str(first).strip()

    return str(token_ids).strip()


def _build_prompt(history: list[dict], user_message: str, max_turns: int) -> str:
    recent_history = history[-max_turns:] if max_turns > 0 else []
    parts: list[str] = []

    for item in recent_history:
        parts.append(f"User: {item['user']}")
        assistant_text = item["assistant_text"] or "[Generated image only]"
        parts.append(f"Assistant: {assistant_text}")

    parts.append(f"User: {user_message.strip()}")
    parts.append("Assistant:")
    return "\n".join(parts)


def load_pipeline(
    checkpoint_path: str,
    device_name: str,
    dtype_name: str,
    progress=gr.Progress(track_tqdm=True),
):
    global _PIPELINE
    global _PIPELINE_DEVICE
    global _PIPELINE_DTYPE

    checkpoint_path = checkpoint_path.strip()
    if not checkpoint_path:
        raise gr.Error("Checkpoint path or Hugging Face model id is required.")

    device = _resolve_device(device_name)
    dtype = _resolve_dtype(dtype_name)

    try:
        _LOGGER.info(
            "Loading pipeline checkpoint=%s device=%s dtype=%s",
            checkpoint_path,
            device,
            dtype,
        )
        progress(0.1, desc="Loading pipeline")
        pipeline = InternVLUPipeline.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype,
        )
        progress(0.7, desc="Moving pipeline to target device")
        pipeline.to(device=device, dtype=dtype)

        _PIPELINE = pipeline
        _PIPELINE_DEVICE = device
        _PIPELINE_DTYPE = dtype
        _LOGGER.info("Pipeline loaded successfully")
        return (
            f"Loaded `{checkpoint_path}` on `{device}` with dtype `{dtype}`. "
            "Inference requests are serialized to keep the model state stable."
        )
    except Exception:
        _LOGGER.exception("Failed to load pipeline")
        raise


def _make_generator(seed: int) -> torch.Generator:
    if _PIPELINE_DEVICE == "cuda":
        return torch.Generator(device="cuda").manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def _autocast_context():
    if _PIPELINE_DEVICE != "cuda":
        return torch.no_grad()

    autocast_dtype = _PIPELINE_DTYPE
    if autocast_dtype not in {torch.float16, torch.bfloat16}:
        autocast_dtype = torch.float16
    return torch.autocast(device_type="cuda", dtype=autocast_dtype)


def _round_image_size(value: float) -> int:
    rounded = int(round(value / _AUTO_IMAGE_MULTIPLE) * _AUTO_IMAGE_MULTIPLE)
    return max(_AUTO_IMAGE_MULTIPLE, rounded)


def _resolve_generation_size(height, width, input_image) -> tuple[Optional[int], Optional[int]]:
    if height is not None and width is not None:
        return int(height), int(width)
    if height is not None:
        resolved = int(height)
        return resolved, resolved
    if width is not None:
        resolved = int(width)
        return resolved, resolved
    if input_image is None:
        return None, None

    source_width, source_height = input_image.size
    scale = min(
        1.0,
        _AUTO_IMAGE_MAX_EDGE / max(source_width, source_height),
        (_AUTO_IMAGE_MAX_AREA / (source_width * source_height)) ** 0.5,
    )
    resolved_width = _round_image_size(source_width * scale)
    resolved_height = _round_image_size(source_height * scale)
    return resolved_height, resolved_width


def _update_progress(progress, value: float, desc: str) -> None:
    if progress is not None:
        progress(value, desc=desc)


class CancelStoppingCriteria(StoppingCriteria):
    def __init__(self, cancel_event: threading.Event):
        self.cancel_event = cancel_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.cancel_event.is_set()


def _make_diffusion_callback(progress, total_steps: int, start: float, end: float):
    def _callback(pipe, step_index, timestep, callback_kwargs):
        if _CANCEL_EVENT.is_set():
            pipe._interrupt = True

        if total_steps > 0:
            ratio = min((step_index + 1) / total_steps, 1.0)
            progress_value = start + (end - start) * ratio
            _update_progress(
                progress,
                progress_value,
                f"Generating image {step_index + 1}/{total_steps}",
            )
        return callback_kwargs

    return _callback


def _history_to_chatbot(history: list[dict]) -> list[dict]:
    chatbot = []
    for item in history:
        chatbot.append({"role": "user", "content": item["user"]})
        chatbot.append({"role": "assistant", "content": item["assistant_text"]})
    return chatbot


def request_cancel():
    if not _REQUEST_RUNNING.is_set():
        return "No active request."

    _CANCEL_EVENT.set()
    if _PIPELINE is not None and hasattr(_PIPELINE, "image_pipeline"):
        setattr(_PIPELINE.image_pipeline, "_interrupt", True)
    _LOGGER.info("Cancellation requested")
    return "Cancellation requested. The current request will stop as soon as the model yields control."


def run_inference(
    history: list[dict],
    user_message: str,
    input_image,
    generation_mode: str,
    system_prompt: str,
    history_turns: int,
    max_new_tokens: int,
    num_beams: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_inference_steps: int,
    all_cfg_scale: float,
    part_cfg_scale: float,
    height: Optional[float],
    width: Optional[float],
    seed: int,
    progress=gr.Progress(track_tqdm=True),
):
    if _PIPELINE is None:
        raise gr.Error("Load a checkpoint first.")

    user_message = user_message.strip()
    if not user_message:
        raise gr.Error("Enter a prompt before sending.")

    system_prompt = (system_prompt or "").strip()
    _CANCEL_EVENT.clear()
    _REQUEST_RUNNING.set()

    history = history or []
    prompt = _build_prompt(history, user_message, history_turns)

    generation_kwargs = {
        "prompt": prompt,
        "image": input_image,
        "generation_mode": generation_mode,
        "system_prompt": system_prompt or None,
    }

    _update_progress(progress, 0.02, "Preparing request")

    if generation_mode in {"text", "text_image"}:
        if generation_mode == "text":
            _update_progress(progress, 0.1, "Running multimodal text generation")
        else:
            _update_progress(progress, 0.15, "Generating reasoning text")
        generation_kwargs["max_new_tokens"] = int(max_new_tokens)
        generation_kwargs["num_beams"] = int(num_beams)
        generation_kwargs["do_sample"] = do_sample
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [CancelStoppingCriteria(_CANCEL_EVENT)]
        )
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

    if generation_mode in {"image", "text_image"}:
        resolved_height, resolved_width = _resolve_generation_size(
            height, width, input_image
        )
        if generation_mode == "image":
            _update_progress(progress, 0.12, "Encoding image and prompt")
            callback_start, callback_end = 0.2, 0.95
        else:
            callback_start, callback_end = 0.45, 0.95
        generation_kwargs.update(
            {
                "num_inference_steps": int(num_inference_steps),
                "all_cfg_scale": all_cfg_scale,
                "part_cfg_scale": part_cfg_scale,
                "generator": _make_generator(int(seed)),
                "callback_on_step_end": _make_diffusion_callback(
                    progress,
                    int(num_inference_steps),
                    callback_start,
                    callback_end,
                ),
            }
        )
        if resolved_height is not None and resolved_width is not None:
            generation_kwargs["height"] = resolved_height
            generation_kwargs["width"] = resolved_width

    try:
        _LOGGER.info(
            "Inference start mode=%s has_image=%s max_new_tokens=%s steps=%s",
            generation_mode,
            input_image is not None,
            generation_kwargs.get("max_new_tokens"),
            generation_kwargs.get("num_inference_steps"),
        )

        with _PIPELINE_LOCK, torch.no_grad():
            if _PIPELINE_DEVICE == "cuda":
                with _autocast_context():
                    output = _PIPELINE(**generation_kwargs)
            else:
                output = _PIPELINE(**generation_kwargs)

        if _CANCEL_EVENT.is_set():
            _LOGGER.info("Inference canceled mode=%s", generation_mode)
            _update_progress(progress, 1.0, "Request canceled")
            return history, _history_to_chatbot(history), None, "Request canceled."

        response_text = _decode_text_output(
            output.generate_output, _PIPELINE.processor.tokenizer
        )
        response_image = None
        if output.images is not None and len(output.images) > 0:
            response_image = output.images[0]

        if generation_mode == "image" and not response_text:
            response_text = "Generated image."
        elif generation_mode == "text_image" and not response_text:
            response_text = "Generated text and image."

        history.append(
            {
                "user": user_message,
                "assistant_text": response_text,
                "mode": generation_mode,
            }
        )

        chatbot = _history_to_chatbot(history)

        _LOGGER.info(
            "Inference success mode=%s response_text_len=%s has_output_image=%s",
            generation_mode,
            len(response_text),
            response_image is not None,
        )
        _update_progress(progress, 1.0, "Completed")
        return history, chatbot, response_image, ""
    except Exception as exc:
        _LOGGER.exception("Inference failed mode=%s", generation_mode)
        _update_progress(progress, 1.0, "Failed")
        raise gr.Error(f"Inference failed: {type(exc).__name__}: {exc}") from exc
    finally:
        _REQUEST_RUNNING.clear()
        _CANCEL_EVENT.clear()
        if _PIPELINE is not None and hasattr(_PIPELINE, "image_pipeline"):
            setattr(_PIPELINE.image_pipeline, "_interrupt", False)


def clear_history():
    return [], [], None, ""


def build_demo(default_checkpoint: str):
    with gr.Blocks(title="InternVL-U Chat") as demo:
        gr.Markdown(
            """
            # InternVL-U Chat
            A Gradio chat interface built on top of the README examples.
            Use the current turn's optional image together with `text`, `image`,
            or `text_image` mode.
            """
        )

        history_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=2):
                checkpoint = gr.Textbox(
                    value=default_checkpoint,
                    label="Checkpoint Path or Hugging Face Model ID",
                    placeholder="InternVL-U/InternVL-U or /path/to/checkpoint",
                )
            with gr.Column(scale=1):
                device_name = gr.Dropdown(
                    choices=["auto", "cuda", "cpu"],
                    value="auto",
                    label="Device",
                )
                dtype_name = gr.Dropdown(
                    choices=["auto", "bfloat16", "float16", "float32"],
                    value="auto",
                    label="Torch Dtype",
                )
            with gr.Column(scale=1):
                load_button = gr.Button("Load Model", variant="primary")
                load_status = gr.Markdown("Model not loaded.")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Conversation", height=520)
                latest_image = gr.Image(label="Latest Generated Image", type="pil")
            with gr.Column(scale=2):
                input_image = gr.Image(label="Input Image", type="pil")
                user_message = gr.Textbox(
                    label="Message",
                    lines=6,
                    placeholder=(
                        "Ask a question, describe an image to generate, "
                        "or upload an image to edit."
                    ),
                )
                generation_mode = gr.Radio(
                    choices=["text", "image", "text_image"],
                    value="text",
                    label="Generation Mode",
                )
                send_button = gr.Button("Send", variant="primary")
                cancel_button = gr.Button("Cancel Current Request")
                clear_button = gr.Button("Clear Chat")
                request_status = gr.Markdown("")

                with gr.Accordion("Advanced Options", open=False):
                    system_prompt = gr.Textbox(
                        label="System Prompt",
                        lines=3,
                        placeholder="Optional system prompt",
                    )
                    history_turns = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=4,
                        step=1,
                        label="History Turns Included in Prompt",
                    )
                    max_new_tokens = gr.Slider(
                        minimum=32,
                        maximum=2048,
                        value=512,
                        step=32,
                        label="Max New Tokens",
                    )
                    num_beams = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=1,
                        step=1,
                        label="Num Beams",
                    )
                    do_sample = gr.Checkbox(label="Enable Sampling", value=False)
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-p",
                    )
                    num_inference_steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=20,
                        step=1,
                        label="Diffusion Steps",
                    )
                    all_cfg_scale = gr.Slider(
                        minimum=0.0,
                        maximum=10.0,
                        value=4.5,
                        step=0.1,
                        label="All CFG Scale",
                    )
                    part_cfg_scale = gr.Slider(
                        minimum=0.0,
                        maximum=10.0,
                        value=2.0,
                        step=0.1,
                        label="Part CFG Scale",
                    )
                    height = gr.Number(
                        label="Height",
                        precision=0,
                        value=None,
                        placeholder="Auto, capped for responsiveness",
                    )
                    width = gr.Number(
                        label="Width",
                        precision=0,
                        value=None,
                        placeholder="Auto, capped for responsiveness",
                    )
                    seed = gr.Number(label="Seed", precision=0, value=42)

        load_button.click(
            fn=load_pipeline,
            inputs=[checkpoint, device_name, dtype_name],
            outputs=load_status,
        )

        send_inputs = [
            history_state,
            user_message,
            input_image,
            generation_mode,
            system_prompt,
            history_turns,
            max_new_tokens,
            num_beams,
            do_sample,
            temperature,
            top_p,
            num_inference_steps,
            all_cfg_scale,
            part_cfg_scale,
            height,
            width,
            seed,
        ]
        send_outputs = [history_state, chatbot, latest_image, request_status]

        send_button.click(
            fn=run_inference,
            inputs=send_inputs,
            outputs=send_outputs,
        )
        cancel_button.click(
            fn=request_cancel,
            outputs=request_status,
            queue=False,
        )
        user_message.submit(
            fn=run_inference,
            inputs=send_inputs,
            outputs=send_outputs,
        )
        clear_button.click(
            fn=clear_history,
            outputs=[history_state, chatbot, latest_image, request_status],
        )

    return demo.queue(default_concurrency_limit=1)


def parse_args():
    parser = argparse.ArgumentParser(description="InternVL-U Gradio demo")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Local checkpoint path or Hugging Face model id",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = build_demo(args.checkpoint)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
