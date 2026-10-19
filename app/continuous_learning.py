from store import ANY_TENANT, fetch_audit_logs, fetch_feedback


def build_training_rows(limit: int = 1000, min_rating: int = 4, tenant=ANY_TENANT) -> list[dict]:
    """Interactions whose answer was rated ``>= min_rating``.

    Feedback is matched to the exact answer by ``query_id``; rows from before
    query ids existed fall back to (user_id, query). The newest rating wins.
    Unrated or low-rated answers are never emitted.
    """
    feedback = fetch_feedback(limit=limit * 5, tenant=tenant)
    by_query_id: dict = {}
    by_text: dict = {}
    for f in feedback:  # newest first
        if f.get("query_id"):
            by_query_id.setdefault(f["query_id"], f)
        else:
            by_text.setdefault((f.get("user_id"), f.get("query")), f)
    rows = []
    for row in fetch_audit_logs(limit=limit * 5, tenant=tenant):
        fb = by_query_id.get(row.get("query_id")) if row.get("query_id") else None
        if fb is None:
            fb = by_text.get((row.get("user_id"), row.get("query")))
        if fb is None or fb.get("rating") is None or fb["rating"] < min_rating:
            continue
        rows.append({
            "query_id": row.get("query_id"),
            "query": row.get("query"),
            "answer": row.get("answer"),
            "provenance": row.get("provenance"),
            "confidence": row.get("confidence"),
            "rating": fb.get("rating"),
            "comment": fb.get("comment"),
        })
        if len(rows) >= limit:
            break
    return rows
