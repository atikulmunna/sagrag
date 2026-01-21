import json
from pathlib import Path

from store import fetch_audit_logs, fetch_feedback
from config import settings

def export_training_data(path: str, limit: int = 1000, min_rating: int | None = None):
    logs = fetch_audit_logs(limit=limit)
    feedback = fetch_feedback(limit=limit, min_rating=min_rating)
    feedback_by_key = {}
    for f in feedback:
        key = (f.get("user_id"), f.get("query"))
        feedback_by_key[key] = f
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in logs:
            key = (row.get("user_id"), row.get("query"))
            fb = feedback_by_key.get(key)
            item = {
                "query": row.get("query"),
                "answer": row.get("answer"),
                "provenance": row.get("provenance"),
                "confidence": row.get("confidence"),
                "rating": fb.get("rating") if fb else None,
                "comment": fb.get("comment") if fb else None,
            }
            f.write(json.dumps(item) + "\n")
    return {"exported": len(logs), "path": path}

def export_default_training_data():
    return export_training_data(
        settings.learning_export_path,
        limit=1000,
        min_rating=settings.learning_min_rating,
    )
