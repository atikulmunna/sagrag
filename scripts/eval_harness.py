import argparse
import json
import requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--cases", default="data/eval_cases.jsonl")
    args = parser.parse_args()

    total = 0
    passed = 0
    with open(args.cases, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            case = json.loads(line)
            resp = requests.post(
                f"{args.base_url}/v1/query",
                json={"user_id": "eval", "query": case["query"]},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            sources = {p.get("source") for p in data.get("provenance", [])}
            expected = set(case.get("expected_sources", []))
            if expected.intersection(sources):
                passed += 1
            else:
                print(f"FAIL: {case['query']} expected {expected} got {sources}")
    print({"passed": passed, "total": total})

if __name__ == "__main__":
    main()
