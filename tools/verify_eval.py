#!/usr/bin/env python3
"""Sanity-check the claim verifier against hand-labeled cases.

Runs the real NLI model (downloads it on first use), so it is not part of the
unit suite. Every claim is checked against all passages in the file.

Usage (from the repo root):
    python tools/verify_eval.py [--cases data/eval/nli_cases.jsonl]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))

import verification  # noqa: E402
from config import settings  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir(settings.domain_packs_path):
    # Outside the container: use the repo's packs (they name the authors).
    settings.domain_packs_path = os.path.join(_REPO, "data", "domain_packs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="data/eval/nli_cases.jsonl")
    args = parser.parse_args()
    with open(args.cases, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    passages = rows[0]["passages"]
    cases = rows[1:]
    sources = rows[0].get("sources", {})
    evidence = [{"id": pid, "text": text, "source": sources.get(pid)} for pid, text in passages.items()]
    provenance = [{"id": pid} for pid in passages]
    answer_claims = [c["claim"] for c in cases]
    correct = 0
    for case, claim in zip(cases, answer_claims):
        out = verification.verify_sync(claim, evidence, provenance)
        status = out["claims"][0]["status"] if out else "error"
        ok = status == case["label"]
        correct += ok
        print(f"{'  ' if ok else 'XX'} {status:<12} (label {case['label']:<12}) {claim}")
    print(f"\n{correct}/{len(cases)} correct with {settings.verify_model_name} at threshold {settings.verify_threshold}")


if __name__ == "__main__":
    main()
