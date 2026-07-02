#!/usr/bin/env python3
import argparse
import json
import time
from urllib import request


def _post_json(url: str, payload: dict, timeout_s: int = 60):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _load_queries(path: str):
    """Yield query strings from a file.

    Accepts either one query per line (plain text) or JSONL rows carrying a
    ``query`` key, so the same labeled ``queries.jsonl`` can drive both the run
    and (with --score) the scoring.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    row = json.loads(line)
                    if isinstance(row, dict) and row.get("query"):
                        yield row["query"]
                        continue
                except Exception:
                    pass
            yield line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-urls", required=True, help="Comma-separated list of base URLs")
    parser.add_argument("--queries", required=True, help="File with one query per line")
    parser.add_argument("--output", default="ablation_results.jsonl")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--score", action="store_true",
                        help="score the run against --ground-truth after writing")
    parser.add_argument("--ground-truth", default=None,
                        help="labeled queries JSONL (defaults to --queries)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--report", default="docs/eval_report.md")
    parser.add_argument("--judge", action="store_true",
                        help="enable LLM-judge scoring (requires Ollama)")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen2.5:7b")
    args = parser.parse_args()

    base_urls = [u.strip().rstrip("/") for u in args.base_urls.split(",") if u.strip()]
    if not base_urls:
        raise SystemExit("No base URLs provided")

    out = open(args.output, "w", encoding="utf-8")
    total = 0
    for query in _load_queries(args.queries):
        for base in base_urls:
            t0 = time.time()
            payload = {"user_id": "ablation", "query": query}
            try:
                resp = _post_json(f"{base}/v1/query", payload, timeout_s=args.timeout)
                elapsed = int((time.time() - t0) * 1000)
                results = resp.get("results") or []
                # Ordered list of retrieved source files (for recall@k scoring)
                # and a few evidence snippets (for the optional LLM judge).
                sources = [r.get("source") for r in results if isinstance(r, dict)]
                evidence = [
                    {"source": r.get("source"), "text": (r.get("text") or "")[:400]}
                    for r in results[:5]
                    if isinstance(r, dict)
                ]
                record = {
                    "base_url": base,
                    "query": query,
                    "elapsed_ms": elapsed,
                    "answer": resp.get("answer"),
                    "confidence": resp.get("confidence"),
                    "hallucination_risk": resp.get("hallucination_risk"),
                    "retrieval_failures": resp.get("retrieval_failures"),
                    "author_gap": resp.get("author_gap"),
                    "sources": sources,
                    "evidence": evidence,
                }
            except Exception as e:
                record = {
                    "base_url": base,
                    "query": query,
                    "error": str(e),
                }
            out.write(json.dumps(record, ensure_ascii=True) + "\n")
            out.flush()
            total += 1
    out.close()
    print(f"Wrote {total} rows to {args.output}")

    if args.score:
        import eval_metrics

        gt_path = args.ground_truth or args.queries
        gt = eval_metrics.load_ground_truth(gt_path)
        rows = eval_metrics.load_results(args.output)
        aggregates = eval_metrics.score_run(
            rows, gt, k=args.k, judge=args.judge,
            ollama_url=args.ollama_url, model=args.model,
        )
        report = eval_metrics.render_markdown(aggregates, args.k)
        eval_metrics.write_report(report, args.report)
        print(report)
        print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
