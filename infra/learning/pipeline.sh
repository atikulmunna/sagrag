#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data/learning}"
RAW="${DATA_DIR}/train.jsonl"
CURATED="${DATA_DIR}/curated.jsonl"
TRAIN="${DATA_DIR}/train_split.jsonl"
EVAL="${DATA_DIR}/eval_split.jsonl"
ARTIFACTS="${DATA_DIR}/artifacts"

MIN_RATING="${MIN_RATING:-4}"
EVAL_RATIO="${EVAL_RATIO:-0.1}"

python scripts/curate_dataset.py --input "$RAW" --output "$CURATED" --min-rating "$MIN_RATING"
python scripts/split_dataset.py --input "$CURATED" --train "$TRAIN" --eval "$EVAL" --eval-ratio "$EVAL_RATIO"
python scripts/train_stub.py --data "$TRAIN" --out "$ARTIFACTS"
python scripts/eval_stub.py --data "$EVAL" --out "${ARTIFACTS}/eval.json"
