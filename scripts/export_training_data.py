import argparse
from app.continuous_learning import export_training_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--min-rating", type=int, default=None)
    args = parser.parse_args()
    res = export_training_data(args.path, limit=args.limit, min_rating=args.min_rating)
    print(res)

if __name__ == "__main__":
    main()
