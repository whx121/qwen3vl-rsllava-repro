#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modeling import device_for, load_qwen_model, processor_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL on one remote-sensing image")
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-pixels", type=int, default=336 * 336)
    parser.add_argument("--flash-attention-2", action="store_true")
    args = parser.parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Image does not exist: {args.image}")
    model, processor = load_qwen_model(args.base_model, args.adapter, args.flash_attention_2)
    messages = [{"role": "user", "content": [{"type": "image", "image": str(args.image)}, {"type": "text", "text": args.prompt}]}]
    inputs = processor_inputs(processor, messages, args.max_pixels)
    device = device_for(model)
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    prompt_length = inputs["input_ids"].shape[1]
    output = processor.batch_decode(generated_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(output.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
