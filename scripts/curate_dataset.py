import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-rating", type=int, default=4)
    args = parser.parse_args()

    kept = 0
    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                item = json.loads(line)
            except Exception:
                continue
            rating = item.get("rating")
            if rating is None or rating < args.min_rating:
                continue
            fout.write(json.dumps(item) + "\n")
            kept += 1
    print({"kept": kept, "output": args.output})

if __name__ == "__main__":
    main()
