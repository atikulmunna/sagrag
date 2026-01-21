import json
import sqlite3
import time
from pathlib import Path

from config import settings

def _connect():
    Path(settings.audit_db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(settings.audit_db_path)

def init_db():
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS query_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                query TEXT,
                intent TEXT,
                domain TEXT,
                domain_source TEXT,
                confidence REAL,
                answer TEXT,
                provenance_json TEXT,
                created_at REAL
            )
            """
        )
        # Best-effort migrations for older schemas.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(query_audit)").fetchall()}
        if "domain" not in cols:
            conn.execute("ALTER TABLE query_audit ADD COLUMN domain TEXT")
        if "domain_source" not in cols:
            conn.execute("ALTER TABLE query_audit ADD COLUMN domain_source TEXT")
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_audit_created_at ON query_audit(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_audit_user_id ON query_audit(user_id)")
        except Exception:
            pass

def _connect_feedback():
    Path(settings.feedback_db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(settings.feedback_db_path)

def init_feedback_db():
    with _connect_feedback() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                query TEXT,
                rating INTEGER,
                comment TEXT,
                created_at REAL
            )
            """
        )

def log_feedback(user_id, query, rating, comment):
    try:
        init_feedback_db()
        with _connect_feedback() as conn:
            conn.execute(
                """
                INSERT INTO feedback (user_id, query, rating, comment, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, query, rating, comment, time.time()),
            )
    except Exception:
        pass

def fetch_feedback(limit=1000, min_rating=None):
    try:
        init_feedback_db()
        with _connect_feedback() as conn:
            if min_rating is None:
                rows = conn.execute(
                    """
                    SELECT user_id, query, rating, comment, created_at
                    FROM feedback
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT user_id, query, rating, comment, created_at
                    FROM feedback
                    WHERE rating >= ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (min_rating, limit),
                ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "user_id": r[0],
                    "query": r[1],
                    "rating": r[2],
                    "comment": r[3],
                    "created_at": r[4],
                }
            )
        return out
    except Exception:
        return []

def log_query_result(user_id, query, intent, answer, provenance, confidence, domain=None, domain_source=None):
    try:
        init_db()
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO query_audit
                (user_id, query, intent, domain, domain_source, confidence, answer, provenance_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    query,
                    intent,
                    domain,
                    domain_source,
                    float(confidence) if confidence is not None else None,
                    answer,
                    json.dumps(provenance or []),
                    time.time(),
                ),
            )
    except Exception:
        # best-effort audit logging
        pass

def _encode_cursor(created_at, row_id):
    return f"{created_at}|{row_id}"

def _decode_cursor(token):
    try:
        created_at_str, row_id_str = token.split("|", 1)
        return float(created_at_str), int(row_id_str)
    except Exception:
        return None, None

def fetch_audit_logs(limit=50, user_id=None, cursor=None):
    try:
        init_db()
        cursor_clause = ""
        params = []
        if cursor:
            created_at, row_id = _decode_cursor(cursor)
            if created_at is not None and row_id is not None:
                cursor_clause = "AND (created_at < ? OR (created_at = ? AND id < ?))"
                params.extend([created_at, created_at, row_id])
        with _connect() as conn:
            if user_id:
                rows = conn.execute(
                    """
                    SELECT id, user_id, query, intent, domain, domain_source, confidence, answer, provenance_json, created_at
                    FROM query_audit
                    WHERE user_id = ?
                    """ + cursor_clause + """
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    [user_id, *params, limit],
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, user_id, query, intent, domain, domain_source, confidence, answer, provenance_json, created_at
                    FROM query_audit
                    WHERE 1=1
                    """ + cursor_clause + """
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    [*params, limit],
                ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "user_id": r[1],
                    "query": r[2],
                    "intent": r[3],
                    "domain": r[4],
                    "domain_source": r[5],
                    "confidence": r[6],
                    "answer": r[7],
                    "provenance": json.loads(r[8] or "[]"),
                    "created_at": r[9],
                    "cursor": _encode_cursor(r[9], r[0]),
                }
            )
        return out
    except Exception:
        return []

def export_audit_jsonl(path, limit=1000, user_id=None):
    try:
        logs = fetch_audit_logs(limit=limit, user_id=user_id)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in logs:
                f.write(json.dumps(row) + "\n")
        return {"exported": len(logs), "path": path}
    except Exception:
        return {"exported": 0, "path": path}
