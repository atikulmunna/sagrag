"""SQLite audit log and feedback store.

Every row records the tenant (when tenant isolation is on) and the
``query_id`` returned by /query, so reads can be scoped to the caller and
feedback can be tied to the exact answer it rates. Writes are best-effort: a
storage failure is logged and never fails the request. These functions are
synchronous; async callers run them in a worker thread.
"""
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

from config import settings

_LOG = logging.getLogger(__name__)
_INIT_LOCK = threading.Lock()
_INITIALIZED: set[tuple[str, str]] = set()

# Sentinel: "do not filter by tenant" (tenant isolation off). Passing None
# instead means "rows without a tenant".
ANY_TENANT = object()


def _connect(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path, timeout=5)


def _add_missing_columns(conn, table: str, columns: dict[str, str]):
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def _ensure_schema(path: str, kind: str):
    key = (path, kind)
    if key in _INITIALIZED:
        return
    with _INIT_LOCK:
        if key in _INITIALIZED:
            return
        with _connect(path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            if kind == "audit":
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
                _add_missing_columns(conn, "query_audit", {
                    "domain": "TEXT", "domain_source": "TEXT", "tenant": "TEXT", "query_id": "TEXT",
                })
                conn.execute("CREATE INDEX IF NOT EXISTS idx_query_audit_created_at ON query_audit(created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_query_audit_user_id ON query_audit(user_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_query_audit_tenant ON query_audit(tenant, created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_query_audit_query_id ON query_audit(query_id)")
            else:
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
                _add_missing_columns(conn, "feedback", {"tenant": "TEXT", "query_id": "TEXT"})
                conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_tenant ON feedback(tenant, created_at)")
        _INITIALIZED.add(key)


def init_db():
    _ensure_schema(settings.audit_db_path, "audit")


def init_feedback_db():
    _ensure_schema(settings.feedback_db_path, "feedback")


def _tenant_clause(tenant, clauses: list, params: list):
    if tenant is ANY_TENANT:
        return
    if tenant is None:
        clauses.append("tenant IS NULL")
    else:
        clauses.append("tenant = ?")
        params.append(tenant)


def log_feedback(user_id, query, rating, comment, tenant=None, query_id=None):
    try:
        init_feedback_db()
        with _connect(settings.feedback_db_path) as conn:
            conn.execute(
                """
                INSERT INTO feedback (user_id, query, rating, comment, created_at, tenant, query_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, query, rating, comment, time.time(), tenant, query_id),
            )
        return True
    except Exception:
        _LOG.exception("feedback_write_failed")
        return False


def fetch_feedback(limit=1000, min_rating=None, tenant=ANY_TENANT):
    try:
        init_feedback_db()
        clauses, params = [], []
        if min_rating is not None:
            clauses.append("rating >= ?")
            params.append(min_rating)
        _tenant_clause(tenant, clauses, params)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with _connect(settings.feedback_db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT user_id, query, rating, comment, created_at, query_id
                FROM feedback {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
        return [
            {"user_id": r[0], "query": r[1], "rating": r[2], "comment": r[3], "created_at": r[4], "query_id": r[5]}
            for r in rows
        ]
    except Exception:
        _LOG.exception("feedback_read_failed")
        return []


def log_query_result(
    user_id, query, intent, answer, provenance, confidence,
    domain=None, domain_source=None, tenant=None, query_id=None,
):
    try:
        init_db()
        with _connect(settings.audit_db_path) as conn:
            conn.execute(
                """
                INSERT INTO query_audit
                (user_id, query, intent, domain, domain_source, confidence, answer, provenance_json,
                 created_at, tenant, query_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    tenant,
                    query_id,
                ),
            )
        return True
    except Exception:
        # best-effort audit logging
        _LOG.exception("audit_write_failed")
        return False


def _encode_cursor(created_at, row_id):
    return f"{created_at}|{row_id}"


def _decode_cursor(token):
    try:
        created_at_str, row_id_str = token.split("|", 1)
        return float(created_at_str), int(row_id_str)
    except Exception:
        return None, None


def fetch_audit_logs(limit=50, user_id=None, cursor=None, tenant=ANY_TENANT):
    try:
        init_db()
        clauses, params = [], []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        _tenant_clause(tenant, clauses, params)
        if cursor:
            created_at, row_id = _decode_cursor(cursor)
            if created_at is not None and row_id is not None:
                clauses.append("(created_at < ? OR (created_at = ? AND id < ?))")
                params.extend([created_at, created_at, row_id])
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with _connect(settings.audit_db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT id, user_id, query, intent, domain, domain_source, confidence, answer,
                       provenance_json, created_at, query_id
                FROM query_audit {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
        return [
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
                "query_id": r[10],
                "cursor": _encode_cursor(r[9], r[0]),
            }
            for r in rows
        ]
    except Exception:
        _LOG.exception("audit_read_failed")
        return []


def write_jsonl(rows, path: str) -> dict:
    """Write rows to a server-configured path (never a client-supplied one)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return {"exported": len(rows), "path": path}
