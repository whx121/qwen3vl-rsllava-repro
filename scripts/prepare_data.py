#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsllava_repro.data import filter_rows_by_image_prefix, load_huggingface_rows, load_json_or_jsonl, parse_path_maps, prepare_records, split_by_image, validate_split_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert RS-LLaVA annotations to Qwen3-VL format")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--annotation", type=Path)
    source.add_argument("--hf-dataset")
    parser.add_argument("--hf-config")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--path-map", action="append", default=[], help="PREFIX=PATH, repeatable")
    parser.add_argument("--include-prefix", action="append", default=[], help="Only keep image paths under PREFIX/, repeatable")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--output-format", choices=("json", "jsonl", "both"), default="json")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    rows = load_json_or_jsonl(args.annotation) if args.annotation else load_huggingface_rows(args.hf_dataset, args.hf_split, args.hf_config)
    rows_before_filter = len(rows)
    rows = filter_rows_by_image_prefix(rows, args.include_prefix)
    path_maps = parse_path_maps(args.path_map)
    records, rejected, stats = prepare_records(rows, args.data_root.expanduser().resolve(), path_maps, args.allow_missing)
    if rejected and not args.allow_missing:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "rejected.json").write_text(json.dumps([r.__dict__ for r in rejected], ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"Rejected {len(rejected)} rows. See {args.out_dir / 'rejected.json'}; use --allow-missing only for annotation inspection.")
    train, validation = split_by_image(records, args.validation_ratio, args.seed)
    if args.max_train_samples is not None:
        train = train[: args.max_train_samples]
    if args.max_validation_samples is not None:
        validation = validation[: args.max_validation_samples]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if rejected:
        (args.out_dir / "rejected.json").write_text(json.dumps([r.__dict__ for r in rejected], ensure_ascii=False, indent=2), encoding="utf-8")
    for stem, value in (("train", train), ("validation", validation)):
        if args.output_format in ("json", "both"):
            (args.out_dir / f"{stem}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.output_format in ("jsonl", "both"):
            with (args.out_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as handle:
                for row in value:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    split_report = validate_split_files(train, validation)
    manifest = {
        "source": args.annotation.as_posix() if args.annotation else args.hf_dataset,
        "source_split": args.hf_split if args.hf_dataset else None,
        "rows_before_prefix_filter": rows_before_filter,
        "include_prefix": args.include_prefix,
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "stats": stats.to_dict(),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_images": len({row["image"] for row in train}),
        "validation_images": len({row["image"] for row in validation}),
        "split_report": split_report,
        "rejected_rows": [r.__dict__ for r in rejected],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
