import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    total = 0
    with open(args.data, "r", encoding="utf-8") as f:
        for _ in f:
            total += 1
    metrics = {"examples": total, "placeholder_metric": 0.0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f)
    print(metrics)

if __name__ == "__main__":
    main()
