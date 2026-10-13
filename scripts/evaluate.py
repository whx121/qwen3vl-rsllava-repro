#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modeling import device_for, load_qwen_model, make_messages, processor_inputs
from rsllava_repro.data import classify_question, load_json_or_jsonl


def normalize(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize(prediction).split()
    ref = normalize(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    pred_counts = defaultdict(int)
    ref_counts = defaultdict(int)
    for token in pred:
        pred_counts[token] += 1
    for token in ref:
        ref_counts[token] += 1
    common = sum(min(count, ref_counts[token]) for token, count in pred_counts.items())
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    pred = normalize(prediction).split()
    ref = normalize(reference).split()
    table = [[0] * (len(ref) + 1) for _ in range(len(pred) + 1)]
    for i, ptoken in enumerate(pred, 1):
        for j, rtoken in enumerate(ref, 1):
            table[i][j] = table[i - 1][j - 1] + 1 if ptoken == rtoken else max(table[i - 1][j], table[i][j - 1])
    lcs = table[-1][-1]
    if not pred or not ref:
        return float(not pred and not ref)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def generate(model, processor, messages, max_pixels: int, max_new_tokens: int) -> str:
    inputs = processor_inputs(processor, messages, max_pixels)
    device = device_for(model)
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    prompt_length = inputs["input_ids"].shape[1]
    return processor.batch_decode(generated_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def make_single_turn_messages(image: str, question: str) -> list[dict[str, object]]:
    """Build an independent image-question prompt without conversation history."""
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]


def score_item(item: dict[str, str]) -> dict[str, float]:
    prediction = item["prediction"]
    reference = item["reference"]
    return {
        "exact_match": float(normalize(prediction) == normalize(reference)),
        "token_f1": token_f1(prediction, reference),
        "rouge_l": rouge_l(prediction, reference),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL on the prepared RS-LLaVA validation split")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--random-select",
        action="store_true",
        help="Randomly select --max-samples image records using --sample-seed",
    )
    parser.add_argument(
        "--history-mode",
        choices=("oracle", "none"),
        default="oracle",
        help="oracle keeps previous reference answers; none evaluates every turn independently",
    )
    parser.add_argument("--max-pixels", type=int, default=336 * 336)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sample-count", type=int, default=20, help="Number of random examples to save")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10, help="Print progress every N image records")
    parser.add_argument("--flash-attention-2", action="store_true")
    args = parser.parse_args()
    rows = load_json_or_jsonl(args.data)
    if args.max_samples is not None:
        if args.random_select:
            if args.max_samples < 0:
                raise ValueError("--max-samples must be non-negative")
            rng = random.Random(args.sample_seed)
            rows = rng.sample(rows, min(args.max_samples, len(rows)))
        else:
            rows = rows[: args.max_samples]
    selected_records = [
        {
            "conversation_id": row.get("conversation_id", row.get("id")),
            "image": row["image"],
            "turn_count": len(row["conversations"]) // 2,
        }
        for row in rows
    ]
    model, processor = load_qwen_model(args.model, args.adapter, args.flash_attention_2)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_config.json").write_text(
        json.dumps(
            {
                "data": str(args.data),
                "model": args.model,
                "adapter": args.adapter,
                "history_mode": args.history_mode,
                "random_select": args.random_select,
                "sample_seed": args.sample_seed,
                "image_records": len(rows),
                "turns": sum(item["turn_count"] for item in selected_records),
                "selected_records": selected_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    predictions = []
    for row_index, row in enumerate(rows, 1):
        conversation = row["conversations"]
        for index in range(0, len(conversation), 2):
            question = conversation[index]["value"].replace("<image>", "").strip()
            reference = conversation[index + 1]["value"].strip()
            if args.history_mode == "none":
                messages = make_single_turn_messages(row["image"], question)
            else:
                messages = make_messages(row["image"], conversation, upto_user_index=index // 2)
            prediction = generate(model, processor, messages, args.max_pixels, args.max_new_tokens)
            task_type = classify_question(question)
            item = {"image": row["image"], "question": question, "reference": reference, "prediction": prediction, "conversation_id": row.get("conversation_id", row.get("id")), "task_type": task_type, "turn_index": index // 2, "history_mode": args.history_mode}
            item.update(score_item(item))
            predictions.append(item)
        if args.log_every > 0 and (row_index % args.log_every == 0 or row_index == len(rows)):
            print(f"evaluated {row_index}/{len(rows)} image records ({len(predictions)} turns)", flush=True)
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    metrics: dict[str, dict[str, float | int]] = {}
    for task_type in ("caption", "vqa"):
        subset = [item for item in predictions if item["task_type"] == task_type]
        if not subset:
            continue
        metrics[task_type] = {
            "count": len(subset),
            "exact_match": sum(x["exact_match"] for x in subset) / len(subset),
            "token_f1": sum(token_f1(x["prediction"], x["reference"]) for x in subset) / len(subset),
            "rouge_l": sum(rouge_l(x["prediction"], x["reference"]) for x in subset) / len(subset),
        }
    metrics["overall"] = {
        "count": len(predictions),
        "exact_match": sum(x["exact_match"] for x in predictions) / len(predictions) if predictions else 0.0,
        "token_f1": sum(x["token_f1"] for x in predictions) / len(predictions) if predictions else 0.0,
        "rouge_l": sum(x["rouge_l"] for x in predictions) / len(predictions) if predictions else 0.0,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    rng = random.Random(args.sample_seed)
    sample_count = max(0, min(args.sample_count, len(predictions)))
    samples = rng.sample(predictions, sample_count) if sample_count else []
    errors = [item for item in predictions if item["exact_match"] == 0.0 and item["token_f1"] < 1.0]
    errors.sort(key=lambda item: (item["token_f1"], item["rouge_l"]))
    (args.output / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for item in errors:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
