from __future__ import annotations

from pathlib import Path
from typing import Any


def load_qwen_model(base_model: str, adapter: str | None = None, flash_attention_2: bool = False):
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the project dependencies before loading Qwen3-VL") from exc
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if flash_attention_2:
        kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForImageTextToText.from_pretrained(base_model, **kwargs)
    processor = AutoProcessor.from_pretrained(base_model)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, processor


def device_for(model):
    import torch
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def make_messages(image: str | Path, conversation: list[dict[str, str]], upto_user_index: int | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    user_count = 0
    for item in conversation:
        role = "user" if item["from"] == "human" else "assistant"
        text = item["value"].replace("<image>", "").strip()
        if role == "user":
            content: list[dict[str, str]] = []
            if user_count == 0:
                content.append({"type": "image", "image": str(image)})
            content.append({"type": "text", "text": text})
            messages.append({"role": "user", "content": content})
            user_count += 1
            if upto_user_index is not None and user_count - 1 >= upto_user_index:
                break
        else:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
    return messages


def processor_inputs(processor, messages: list[dict[str, Any]], max_pixels: int | None = None):
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    kwargs: dict[str, Any] = {
        "text": [text],
        "images": image_inputs,
        "videos": video_inputs,
        "padding": True,
        "return_tensors": "pt",
    }
    if max_pixels is not None:
        kwargs["max_pixels"] = max_pixels
    inputs = processor(**kwargs)
    return inputs
