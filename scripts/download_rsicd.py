#!/usr/bin/env python3
"""Download the RSICD image files used by RS-instructions."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download RSICD images from Hugging Face")
    parser.add_argument("--dataset", default="arampacha/rsicd")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/RSICD"))
    parser.add_argument("--splits", nargs="+", default=["train", "test", "valid"])
    parser.add_argument("--max-images", type=int, help="Optional limit for a smoke-test download")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional data runtime
        raise SystemExit("Install datasets in the transformer environment first") from exc

    image_dir = args.out_dir.expanduser().resolve() / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for split in args.splits:
        dataset = load_dataset(args.dataset, split=split)
        for row in dataset:
            filename = row.get("filename")
            image = row.get("image")
            if not isinstance(filename, str) or image is None:
                raise RuntimeError(f"Unexpected RSICD row in split {split}: expected filename and image")
            target = image_dir / Path(filename).name
            if not target.exists():
                image.convert("RGB").save(target, format="JPEG", quality=95)
            saved += 1
            if args.max_images is not None and saved >= args.max_images:
                print(f"Saved {saved} images under {image_dir}")
                return 0
        print(f"Finished split {split}; cumulative rows={saved}")
    print(f"Saved/verified {saved} RSICD rows under {image_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
