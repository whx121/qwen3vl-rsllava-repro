#!/usr/bin/env python3
"""Extract image bytes from Hugging Face RSICD Parquet shards."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract RSICD images from Parquet files")
    parser.add_argument("--parquet-dir", type=Path, default=Path("data/raw/RSICD/images"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/RSICD"))
    parser.add_argument("--max-images", type=int, help="Optional limit for testing")
    args = parser.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional data runtime
        raise SystemExit("Install pyarrow in the transformer environment first") from exc

    parquet_files = sorted(args.parquet_dir.expanduser().resolve().glob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No .parquet files found under {args.parquet_dir}")
    image_dir = args.out_dir.expanduser().resolve() / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped = 0
    for parquet_path in parquet_files:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(columns=["filename", "image"], batch_size=128):
            for row in batch.to_pylist():
                filename = row.get("filename")
                image = row.get("image") or {}
                image_bytes = image.get("bytes") if isinstance(image, dict) else None
                if not isinstance(filename, str) or not isinstance(image_bytes, bytes):
                    raise RuntimeError(f"Unexpected row in {parquet_path}: expected filename and image.bytes")
                target = image_dir / Path(filename).name
                if target.exists():
                    skipped += 1
                else:
                    target.write_bytes(image_bytes)
                    extracted += 1
                if args.max_images is not None and extracted >= args.max_images:
                    print(f"Extracted {extracted} images; skipped existing={skipped}; output={image_dir}")
                    return 0
        print(f"Processed {parquet_path.name}")
    print(f"Extracted {extracted} images; skipped existing={skipped}; output={image_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
