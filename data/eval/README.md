# Evaluation set

`queries.jsonl` is a small labeled ground-truth set for measuring retrieval and
answer quality. Each row:

```json
{"query": "...", "expected_sources": ["file.txt", ...], "reference_answer": "..."}
```

- `expected_sources` — source filenames that *should* be retrieved for the query
  (matched case-insensitively against the basename of each `results[].source`).
- `reference_answer` — a short gold answer used for lexical-overlap scoring and
  as context for the optional LLM judge.

## Running an evaluation

Against a running stack (`/v1/query`):

```bash
# Run queries and score in one shot (retrieval + lexical only):
python tools/ablation_eval.py \
    --base-urls http://localhost:8000 \
    --queries data/eval/queries.jsonl \
    --output ablation_results.jsonl \
    --score --ground-truth data/eval/queries.jsonl --k 5

# Add the deterministic LLM judge (faithfulness + relevance) via Ollama:
python tools/ablation_eval.py \
    --base-urls http://localhost:8000 \
    --queries data/eval/queries.jsonl \
    --score --judge --ollama-url http://localhost:11434 --model qwen2.5:7b
```

To score an existing results file without re-running queries:

```bash
python tools/eval_metrics.py \
    --results ablation_results.jsonl \
    --ground-truth data/eval/queries.jsonl \
    --k 5 --report docs/eval_report.md
```

## Metrics

- **recall@k** — fraction of `expected_sources` present in the top-k retrieved.
- **hit_rate@k** — 1.0 if any expected source appears in the top-k.
- **lexical_overlap** — token-F1 of the answer vs `reference_answer` (a cheap
  baseline that works without the judge).
- **faithfulness / relevance** — LLM-judge scores (0..1); `n/a` unless `--judge`.
  The judge uses `format=json` + `temperature=0` so scores are reproducible.

Pass `--base-urls` a comma-separated list to compare configurations
(e.g. baseline vs SAG-RAG) side by side in one report.
