#!/usr/bin/env python3
"""Scoring for SAG-RAG eval runs.

Turns the JSONL emitted by ``ablation_eval.py`` into retrieval and
answer-quality scores against a labeled ground-truth set
(``data/eval/queries.jsonl``), then writes a Markdown comparison report.

Stdlib-only (mirrors ``ablation_eval.py``): the optional LLM-judge talks to
Ollama over plain HTTP with ``format=json`` + ``temperature=0`` so scores are
deterministic across runs. Without the judge, a lexical-overlap baseline still
produces answer-quality numbers.

Usage:
    python tools/eval_metrics.py \
        --results ablation_results.jsonl \
        --ground-truth data/eval/queries.jsonl \
        --k 5 --report docs/eval_report.md [--judge]
"""
import argparse
import json
import os
import re
import statistics
from urllib import request

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "that", "this", "it", "as", "we", "our", "us", "be", "by", "with", "not",
    "what", "how", "does", "do", "say", "about",
}


# --- text helpers -----------------------------------------------------------


def _tokens(text: str) -> set:
    if not text:
        return set()
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def lexical_overlap(answer: str, reference: str) -> float:
    """Token-level F1 between an answer and a reference answer (0..1)."""
    a, b = _tokens(answer), _tokens(reference)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    precision = inter / len(a)
    recall = inter / len(b)
    return 2 * precision * recall / (precision + recall)


# --- retrieval metrics ------------------------------------------------------


def _norm_source(s) -> str:
    """Normalize a source to a bare filename for comparison."""
    if not s:
        return ""
    s = str(s).replace("\\", "/")
    return s.rsplit("/", 1)[-1].strip().lower()


def recall_at_k(retrieved_sources, expected_sources, k: int) -> float:
    """Fraction of expected sources present in the top-k retrieved sources."""
    expected = {_norm_source(s) for s in expected_sources if s}
    if not expected:
        return 0.0
    topk = {_norm_source(s) for s in retrieved_sources[:k] if s}
    return len(expected & topk) / len(expected)


def hit_rate_at_k(retrieved_sources, expected_sources, k: int) -> float:
    """1.0 if any expected source appears in the top-k, else 0.0."""
    expected = {_norm_source(s) for s in expected_sources if s}
    if not expected:
        return 0.0
    topk = {_norm_source(s) for s in retrieved_sources[:k] if s}
    return 1.0 if expected & topk else 0.0


# --- ground truth -----------------------------------------------------------


def load_ground_truth(path: str) -> dict:
    """Load queries.jsonl into {query: {expected_sources, reference_answer}}."""
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = row.get("query")
            if not q:
                continue
            gt[q] = {
                "expected_sources": row.get("expected_sources", []),
                "reference_answer": row.get("reference_answer", ""),
            }
    return gt


def load_results(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# --- LLM judge (optional) ---------------------------------------------------


def _ollama_json(prompt: str, url: str, model: str, timeout_s: int = 60):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 200},
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{url.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    text = body.get("response", "")
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"{.*}", text, re.DOTALL)
        return json.loads(m.group()) if m else None


def llm_judge(query, answer, evidence, url, model):
    """Score faithfulness + answer-relevance in [0,1] via an Ollama judge.

    Returns (faithfulness, relevance) or (None, None) if the judge is
    unavailable / returns unparsable output.
    """
    snippets = "\n".join(
        f"- ({e.get('source')}) {e.get('text', '')[:300]}"
        for e in (evidence or [])[:5]
    )
    prompt = f"""You are a strict RAG evaluator. Score the ANSWER on two axes from 0.0 to 1.0.
Return JSON only: {{"faithfulness": 0.0-1.0, "relevance": 0.0-1.0}}
- faithfulness: is every claim in the answer supported by the EVIDENCE below? Unsupported claims lower this.
- relevance: does the answer actually address the QUERY?

QUERY:
{query}

EVIDENCE:
{snippets or "(none)"}

ANSWER:
{answer}
"""
    try:
        parsed = _ollama_json(prompt, url, model)
    except Exception:
        return None, None
    if not isinstance(parsed, dict):
        return None, None

    def _clamp(v):
        try:
            return max(0.0, min(1.0, float(v)))
        except Exception:
            return None

    return _clamp(parsed.get("faithfulness")), _clamp(parsed.get("relevance"))


# --- scoring ----------------------------------------------------------------


def _mean(values):
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def score_run(results_rows, ground_truth, k=5, judge=False, ollama_url="", model=""):
    """Score result rows, grouped per base_url. Returns {base_url: aggregate}."""
    per_config: dict = {}
    for row in results_rows:
        base = row.get("base_url", "unknown")
        query = row.get("query", "")
        gt = ground_truth.get(query)
        bucket = per_config.setdefault(
            base,
            {"recall": [], "hit": [], "lexical": [], "faithfulness": [],
             "relevance": [], "confidence": [], "errors": 0, "n": 0},
        )
        bucket["n"] += 1
        if row.get("error"):
            bucket["errors"] += 1
            continue
        sources = row.get("sources") or []
        answer = row.get("answer") or ""
        conf = row.get("confidence")
        if conf is not None:
            bucket["confidence"].append(conf)
        if gt:
            bucket["recall"].append(recall_at_k(sources, gt["expected_sources"], k))
            bucket["hit"].append(hit_rate_at_k(sources, gt["expected_sources"], k))
            if gt["reference_answer"]:
                bucket["lexical"].append(lexical_overlap(answer, gt["reference_answer"]))
        if judge and answer:
            f, r = llm_judge(query, answer, row.get("evidence"), ollama_url, model)
            if f is not None:
                bucket["faithfulness"].append(f)
            if r is not None:
                bucket["relevance"].append(r)

    aggregates = {}
    for base, b in per_config.items():
        aggregates[base] = {
            "n": b["n"],
            "errors": b["errors"],
            f"recall@{k}": _mean(b["recall"]),
            f"hit_rate@{k}": _mean(b["hit"]),
            "lexical_overlap": _mean(b["lexical"]),
            "faithfulness": _mean(b["faithfulness"]),
            "relevance": _mean(b["relevance"]),
            "confidence": _mean(b["confidence"]),
        }
    return aggregates


def render_markdown(aggregates: dict, k: int) -> str:
    cols = [
        "config", "n", "errors", f"recall@{k}", f"hit_rate@{k}",
        "lexical_overlap", "faithfulness", "relevance", "confidence",
    ]

    def fmt(v):
        if v is None:
            return "n/a"
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = ["# SAG-RAG Evaluation Report", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for base, agg in aggregates.items():
        row = [base] + [fmt(agg.get(c)) for c in cols[1:]]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("_recall/hit_rate are against labeled `expected_sources`; "
                 "lexical_overlap is token-F1 vs `reference_answer`; "
                 "faithfulness/relevance are from the LLM judge (n/a if not run)._")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="ablation results JSONL")
    parser.add_argument("--ground-truth", required=True, help="labeled queries JSONL")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--report", default="docs/eval_report.md")
    parser.add_argument("--judge", action="store_true", help="enable LLM-judge scoring")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen2.5:7b")
    args = parser.parse_args()

    gt = load_ground_truth(args.ground_truth)
    rows = load_results(args.results)
    aggregates = score_run(
        rows, gt, k=args.k, judge=args.judge,
        ollama_url=args.ollama_url, model=args.model,
    )
    report = render_markdown(aggregates, args.k)
    write_report(report, args.report)
    print(report)
    print(f"Wrote report to {args.report}")


def write_report(report: str, path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
