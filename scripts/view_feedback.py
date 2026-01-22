import argparse
import json
import sqlite3
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/feedback/feedback.db")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--user-id", default=None)
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")
    conn = sqlite3.connect(db_path)
    if args.user_id:
        rows = conn.execute(
            """
            SELECT user_id, query, rating, comment, created_at
            FROM feedback
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (args.user_id, args.limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT user_id, query, rating, comment, created_at
            FROM feedback
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (args.limit,),
        ).fetchall()
    for r in rows:
        print(json.dumps({
            "user_id": r[0],
            "query": r[1],
            "rating": r[2],
            "comment": r[3],
            "created_at": r[4],
        }, ensure_ascii=False))

if __name__ == "__main__":
    main()
