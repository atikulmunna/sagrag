import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to JSONL training data")
    parser.add_argument("--out", required=True, help="Output directory for artifacts")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    total = 0
    with open(args.data, "r", encoding="utf-8") as f:
        for _ in f:
            total += 1
    stats = {"examples": total}
    with open(Path(args.out) / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f)
    print(stats)

if __name__ == "__main__":
    main()
