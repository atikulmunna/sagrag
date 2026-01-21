import argparse
import json
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    train_count = 0
    eval_count = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.train, "w", encoding="utf-8") as ftrain, \
         open(args.eval, "w", encoding="utf-8") as feval:
        for line in fin:
            try:
                item = json.loads(line)
            except Exception:
                continue
            if random.random() < args.eval_ratio:
                feval.write(json.dumps(item) + "\n")
                eval_count += 1
            else:
                ftrain.write(json.dumps(item) + "\n")
                train_count += 1
    print({"train": train_count, "eval": eval_count})

if __name__ == "__main__":
    main()
