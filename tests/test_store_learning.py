"""Audit/feedback storage: tenant scoping, query ids, schema migration, and
the training-data export filter."""

import sqlite3

import store
from continuous_learning import build_training_rows


def test_learning_export_only_emits_rated_rows(tmp_store):
    # Regression: every audit row was exported, low-rated answers included
    # (as rating=null, i.e. indistinguishable from unlabeled data).
    store.log_query_result("u1", "q-good", "i", "good answer", [], 0.5, query_id="id-good")
    store.log_query_result("u2", "q-unrated", "i", "answer", [], 0.5, query_id="id-unrated")
    store.log_query_result("u3", "q-bad", "i", "bad answer", [], 0.5, query_id="id-bad")
    store.log_feedback("u1", "q-good", 5, "great", query_id="id-good")
    store.log_feedback("u3", "q-bad", 1, "wrong", query_id="id-bad")
    rows = build_training_rows(min_rating=4)
    assert [(r["query"], r["rating"]) for r in rows] == [("q-good", 5)]


def test_learning_export_matches_feedback_by_query_id(tmp_store):
    # Same user asks the same question twice; only the second answer is rated.
    store.log_query_result("u1", "same q", "i", "first answer", [], 0.5, query_id="first")
    store.log_query_result("u1", "same q", "i", "second answer", [], 0.5, query_id="second")
    store.log_feedback("u1", "same q", 5, "", query_id="second")
    rows = build_training_rows(min_rating=4)
    assert [r["answer"] for r in rows] == ["second answer"]


def test_learning_export_legacy_feedback_without_query_id(tmp_store):
    store.log_query_result("u1", "old q", "i", "answer", [], 0.5)
    store.log_feedback("u1", "old q", 4, "")
    assert [r["query"] for r in build_training_rows(min_rating=4)] == ["old q"]


def test_audit_and_feedback_are_tenant_scoped(tmp_store):
    store.log_query_result("u1", "acme q", "i", "a", [], 0.5, tenant="acme")
    store.log_query_result("u2", "globex q", "i", "a", [], 0.5, tenant="globex")
    store.log_feedback("u1", "acme q", 5, "", tenant="acme")
    store.log_feedback("u2", "globex q", 5, "", tenant="globex")
    assert [r["query"] for r in store.fetch_audit_logs(tenant="acme")] == ["acme q"]
    assert [r["query"] for r in store.fetch_feedback(tenant="globex")] == ["globex q"]
    assert len(store.fetch_audit_logs()) == 2  # ANY_TENANT (isolation off)
    assert build_training_rows(min_rating=4, tenant="acme")[0]["query"] == "acme q"


def test_audit_cursor_paging(tmp_store):
    for i in range(5):
        store.log_query_result("u", f"q{i}", "i", "a", [], 0.5)
    first = store.fetch_audit_logs(limit=2)
    second = store.fetch_audit_logs(limit=2, cursor=first[-1]["cursor"])
    assert [r["query"] for r in first] == ["q4", "q3"]
    assert [r["query"] for r in second] == ["q2", "q1"]


def test_old_schema_is_migrated(tmp_store):
    path = store.settings.audit_db_path
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE query_audit (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, query TEXT, "
            "intent TEXT, confidence REAL, answer TEXT, provenance_json TEXT, created_at REAL)"
        )
    assert (
        store.log_query_result("u", "q", "i", "a", [], 0.5, tenant="acme", query_id="qid") is True
    )
    row = store.fetch_audit_logs(tenant="acme")[0]
    assert row["query_id"] == "qid"


def test_write_jsonl(tmp_store):
    out = store.write_jsonl([{"a": 1}, {"b": 2}], str(tmp_store / "x" / "out.jsonl"))
    assert out["exported"] == 2
    assert (tmp_store / "x" / "out.jsonl").read_text(encoding="utf-8").splitlines() == [
        '{"a": 1}',
        '{"b": 2}',
    ]
