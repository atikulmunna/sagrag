#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_PATH="${1:-/data/learning/train.jsonl}"
MIN_RATING="${MIN_RATING:-4}"
LIMIT="${LIMIT:-1000}"

python scripts/export_training_data.py --path "$OUT_PATH" --limit "$LIMIT" --min-rating "$MIN_RATING"
