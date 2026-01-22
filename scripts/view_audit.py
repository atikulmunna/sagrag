import argparse
import json
import sqlite3
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/audit/sag_rag_audit.db")
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
            SELECT user_id, query, intent, confidence, answer, provenance_json, created_at
            FROM query_audit
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (args.user_id, args.limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT user_id, query, intent, confidence, answer, provenance_json, created_at
            FROM query_audit
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (args.limit,),
        ).fetchall()
    for r in rows:
        print(json.dumps({
            "user_id": r[0],
            "query": r[1],
            "intent": r[2],
            "confidence": r[3],
            "answer": r[4],
            "provenance": json.loads(r[5] or "[]"),
            "created_at": r[6],
        }, ensure_ascii=False))

if __name__ == "__main__":
    main()
