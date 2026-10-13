#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modeling import device_for, load_qwen_model, make_messages, processor_inputs
from rsllava_repro.data import load_json_or_jsonl


def attach_language_lora(model):
    """Attach a tiny language-only LoRA adapter for a realistic backward test.

    The official trainer applies the production LoRA configuration.  This
    smoke test deliberately uses a smaller adapter so that a backward pass
    does not allocate gradients for the frozen 4B base model or the vision
    tower.
    """
    try:
        import torch.nn as nn
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover - optional model runtime
        raise RuntimeError("Install peft before running the model smoke test") from exc

    model.requires_grad_(False)
    target_modules = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        lowered = name.lower()
        if any(token in lowered for token in ("visual", "vision", "merger", "projector", "lm_head")):
            continue
        target_modules.append(name)
    if not target_modules:
        raise RuntimeError("Could not find language-model linear layers for LoRA")
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Qwen3-VL processor/forward/backward smoke test")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--max-pixels", type=int, default=256 * 256)
    args = parser.parse_args()
    rows = load_json_or_jsonl(args.data)
    if not rows:
        raise SystemExit("No records in data file")
    model, processor = load_qwen_model(args.model)
    model = attach_language_lora(model)
    model.train()
    row = rows[0]
    messages = make_messages(row["image"], row["conversations"], upto_user_index=0)
    inputs = processor_inputs(processor, messages, args.max_pixels)
    device = device_for(model)
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    labels = inputs["input_ids"].clone()
    if "attention_mask" in inputs:
        labels[inputs["attention_mask"] == 0] = -100
    outputs = model(**inputs, labels=labels)
    if outputs.loss is None or not math.isfinite(float(outputs.loss.detach().cpu())):
        raise SystemExit("Non-finite loss")
    outputs.loss.backward()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    gradients = sum(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    if not trainable or not gradients:
        raise SystemExit("Backward completed but no LoRA parameter received a gradient")
    print(
        f"forward/backward ok; loss={float(outputs.loss.detach().cpu()):.6f}; "
        f"trainable_parameters={trainable}; tensors_with_grad={gradients}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
