#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python scripts/train_qwen3vl.py \
  --train-json "${TRAIN_JSON:-data/rsllava/train.json}" \
  --output-dir "${OUTPUT_DIR:-runs/qwen3vl4b_rsllava_sft}" \
  "$@"
