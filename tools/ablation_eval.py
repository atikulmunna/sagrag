#!/usr/bin/env python3
import argparse
import json
import sys
import time
from urllib import request


def _post_json(url: str, payload: dict, timeout_s: int = 60):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _load_queries(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-urls", required=True, help="Comma-separated list of base URLs")
    parser.add_argument("--queries", required=True, help="File with one query per line")
    parser.add_argument("--output", default="ablation_results.jsonl")
    parser.add_argument("--timeout", type=int, default=60)
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
                record = {
                    "base_url": base,
                    "query": query,
                    "elapsed_ms": elapsed,
                    "answer": resp.get("answer"),
                    "confidence": resp.get("confidence"),
                    "hallucination_risk": resp.get("hallucination_risk"),
                    "retrieval_failures": resp.get("retrieval_failures"),
                    "author_gap": resp.get("author_gap"),
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


if __name__ == "__main__":
    main()
