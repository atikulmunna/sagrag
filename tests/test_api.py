"""Endpoint tests through FastAPI's TestClient.

External services are faked at the pipeline boundary (planner, retrievers,
reranker, judge, synthesis), so these exercise routing, validation, tenancy,
middleware order, streaming, caching, and export behavior end to end.
"""

import json
import re

import pytest

import api
import ingest_jobs
import pipeline
import store
from config import settings

_HIT = {
    "id": "c1",
    "score": 0.9,
    "agent": "vector",
    "payload": {
        "text": "Virtue is the only good, said the Stoics.",
        "source": "ethics.txt",
        "offset_start": 0,
        "offset_end": 41,
        "domain": "stoicism",
        "source_type": "txt",
    },
}
_OK = {"attempted": 1, "ok": 1, "zero_hits": 0, "timeout": 0, "error": 0}


@pytest.fixture
def fake_pipeline(monkeypatch):
    calls = {"run_agents": []}

    async def plan(q):
        return {"intent": "lookup", "hypotheses": [], "queries": [q], "constraints": {}}

    async def run_agents(queries, domain=None, fallback_domains=None, tenant=None):
        calls["run_agents"].append({"queries": queries, "domain": domain, "tenant": tenant})
        return [json.loads(json.dumps(_HIT))], {
            "vector": dict(_OK),
            "lexical": dict(_OK),
            "structured": dict(_OK),
        }

    async def rerank(query, docs, top_k=None):
        return [dict(d, rerank_score=3.0) for d in docs]

    async def judge(query, evidence, *args, **kwargs):
        return {"confidence": 0.7, "trusted_ids": ["c1"], "notes": "ok"}

    async def synth(query, evidence, judge_output, author_terms=None, author_gap=False):
        return {
            "answer": "Virtue is the only good.",
            "provenance": [
                {"id": "c1", "source": "ethics.txt", "offset_start": 0, "offset_end": 41}
            ],
            "confidence": 0.7,
            "explain_trace": "synthesis_json",
        }

    async def stream(query, evidence, judge_output, author_terms=None, author_gap=False):
        for part in ("Virtue ", "is the only good."):
            yield part

    async def author_search(*args, **kwargs):
        return [], "zero_hits"

    monkeypatch.setattr(pipeline, "plan_query", plan)
    monkeypatch.setattr(pipeline.agents, "run_agents", run_agents)
    monkeypatch.setattr(pipeline.agents, "author_lexical_search", author_search)
    monkeypatch.setattr(pipeline, "rerank", rerank)
    monkeypatch.setattr(pipeline, "judge_evidence", judge)
    monkeypatch.setattr(pipeline, "synthesize_answer", synth)
    monkeypatch.setattr(api, "synthesize_answer_stream", stream)
    return calls


def _enable_auth(monkeypatch, isolation=True):
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "tenant_isolation", isolation)
    monkeypatch.setattr(settings, "api_key_map", {"key-a": "acme", "key-b": "globex"})


def _q(text="What is virtue?", **extra):
    return {"user_id": "u1", "query": text, **extra}


def _sse_events(body: str):
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.split("\n")
        events.append((lines[0][len("event: ") :], json.loads(lines[1][len("data: ") :])))
    return events


# --- query -------------------------------------------------------------------


def test_query_returns_query_id_and_is_audited(api_client, fake_pipeline):
    r = api_client.post("/v1/query", json=_q())
    assert r.status_code == 200, r.text
    body = r.json()
    assert re.fullmatch(r"[0-9a-f]{32}", body["query_id"])
    assert body["provenance"] == [
        {"id": "c1", "source": "ethics.txt", "offset_start": 0, "offset_end": 41}
    ]
    assert set(body["timings_ms"]) == {"plan", "retrieve", "rerank", "judge", "synthesize"}
    audit = api_client.get("/v1/audit").json()["results"]
    assert audit[0]["query_id"] == body["query_id"]


@pytest.mark.parametrize(
    "payload",
    [
        {"user_id": "u1"},
        {"user_id": "   ", "query": "q"},
        {"user_id": "u1", "query": "x" * 2001},
        {"user_id": "u1", "query": "q", "preferences": {"freshness_days": 0}},
    ],
)
def test_query_validation(api_client, fake_pipeline, payload):
    assert api_client.post("/v1/query", json=payload).status_code == 422


def test_unknown_domain_preference_is_rejected(api_client, fake_pipeline):
    r = api_client.post("/v1/query", json=_q(preferences={"domain": "finance"}))
    assert r.status_code == 422
    assert "unknown domain" in r.json()["detail"]


def test_tenant_cannot_select_another_tenants_namespace(api_client, fake_pipeline, monkeypatch):
    # Regression: preferences.domain was used verbatim as a collection suffix,
    # so tenant globex could read tenant acme's docs_acme namespace.
    _enable_auth(monkeypatch)
    monkeypatch.setattr(
        pipeline.agents,
        "_list_domain_indices",
        lambda tenant=None: ["history"] if tenant == "acme" else [],
    )
    hdr_b = {"x-api-key": "key-b"}
    assert (
        api_client.post(
            "/v1/query", json=_q(preferences={"domain": "acme"}), headers=hdr_b
        ).status_code
        == 422
    )
    assert (
        api_client.post(
            "/v1/query", json=_q(preferences={"domain": "history"}), headers=hdr_b
        ).status_code
        == 422
    )
    # The owner can use its own domain, and a body tenant never overrides the key.
    r = api_client.post(
        "/v1/query",
        json=_q(tenant="globex", preferences={"domain": "history"}),
        headers={"x-api-key": "key-a"},
    )
    assert r.status_code == 200, r.text
    assert fake_pipeline["run_agents"][-1] == {
        "queries": ["What is virtue?"],
        "domain": "history",
        "tenant": "acme",
    }


def test_audit_and_feedback_are_scoped_to_the_callers_tenant(
    api_client, fake_pipeline, monkeypatch
):
    _enable_auth(monkeypatch)
    api_client.post("/v1/query", json=_q("acme question"), headers={"x-api-key": "key-a"})
    api_client.post("/v1/query", json=_q("globex question"), headers={"x-api-key": "key-b"})
    api_client.post(
        "/v1/feedback",
        json={"user_id": "u1", "query": "acme question", "rating": 5},
        headers={"x-api-key": "key-a"},
    )
    audit_a = api_client.get("/v1/audit", headers={"x-api-key": "key-a"}).json()["results"]
    assert [r["query"] for r in audit_a] == ["acme question"]
    # A query-string tenant cannot widen an authenticated caller's scope.
    audit_b = api_client.get("/v1/audit?tenant=acme", headers={"x-api-key": "key-b"}).json()[
        "results"
    ]
    assert [r["query"] for r in audit_b] == ["globex question"]
    assert (
        api_client.get("/v1/feedback/list", headers={"x-api-key": "key-b"}).json()["results"] == []
    )


def test_cache_hit_gets_a_fresh_query_id(api_client, fake_pipeline, monkeypatch):
    cache = {}

    async def get(key):
        return json.loads(cache[key]) if key in cache else None

    async def set_(key, value, ttl):
        cache[key] = json.dumps(value)
        return True

    monkeypatch.setattr(settings, "redis_cache_enabled", True)
    monkeypatch.setattr(api.redis_client, "cache_get_json", get)
    monkeypatch.setattr(api.redis_client, "cache_set_json", set_)
    first = api_client.post("/v1/query", json=_q()).json()
    second = api_client.post("/v1/query", json=_q()).json()
    assert (first["cache"], second["cache"]) == ("miss", "hit")
    assert first["query_id"] != second["query_id"]
    assert len(fake_pipeline["run_agents"]) == 1
    assert len(api_client.get("/v1/audit").json()["results"]) == 2


# --- streaming -----------------------------------------------------------------


def test_stream_reports_stages_tokens_and_final(api_client, fake_pipeline):
    r = api_client.post("/v1/query/stream", json=_q())
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(r.text)
    names = [e for e, _ in events]
    stages = [(d["stage"], d["state"]) for e, d in events if e == "stage"]
    assert stages == [
        ("plan", "running"),
        ("plan", "done"),
        ("retrieve", "running"),
        ("retrieve", "done"),
        ("rerank", "running"),
        ("rerank", "done"),
        ("judge", "running"),
        ("judge", "done"),
        ("synthesize", "running"),
        ("synthesize", "done"),
    ]
    assert names[-2:] == ["final", "done"]
    tokens = "".join(d["text"] for e, d in events if e == "token")
    final = dict(events)["final"]
    meta = dict(events)["meta"]
    assert tokens == final["answer"] == "Virtue is the only good."
    assert final["query_id"] == meta["query_id"]
    # Same finalize step as /query: failure tags and calibrated scores included.
    assert "retrieval_failures" in final and "hallucination_risk" in final
    assert final["provenance"][0]["source"] == "ethics.txt"
    assert api_client.get("/v1/audit").json()["results"][0]["query_id"] == final["query_id"]


def test_stream_reports_pipeline_errors_in_band(api_client, fake_pipeline, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("qdrant exploded")

    monkeypatch.setattr(pipeline.agents, "run_agents", boom)
    events = _sse_events(api_client.post("/v1/query/stream", json=_q()).text)
    assert [e for e, _ in events][-2:] == ["error", "done"]
    assert "qdrant exploded" not in json.dumps(events)  # no internals leaked


# --- exports & feedback ----------------------------------------------------------


@pytest.mark.parametrize("path", ["/v1/audit/export", "/v1/learning/export"])
def test_exports_reject_client_paths(api_client, path):
    # Regression: both endpoints wrote to any client-supplied filesystem path.
    r = api_client.post(path, json={"path": "/app/main.py"})
    assert r.status_code == 422


def test_audit_export_download_and_save(api_client):
    store.log_query_result("u1", "q1", "i", "a1", [], 0.5)
    store.log_query_result("u2", "q2", "i", "a2", [], 0.5)
    r = api_client.post("/v1/audit/export", json={})
    assert r.headers["content-type"].startswith("application/x-ndjson")
    assert "attachment" in r.headers["content-disposition"]
    assert [json.loads(line)["query"] for line in r.text.splitlines()] == ["q2", "q1"]
    saved = api_client.post("/v1/audit/export", json={"save": True}).json()
    assert saved == {"exported": 2, "path": settings.audit_export_path}


def test_learning_export_endpoint_only_rated(api_client):
    store.log_query_result("u1", "good", "i", "a", [], 0.5, query_id="g")
    store.log_query_result("u1", "bad", "i", "a", [], 0.5, query_id="b")
    store.log_feedback("u1", "good", 5, "", query_id="g")
    store.log_feedback("u1", "bad", 2, "", query_id="b")
    r = api_client.post("/v1/learning/export")
    assert [json.loads(line)["query"] for line in r.text.splitlines()] == ["good"]


def test_feedback_validation_and_query_id(api_client):
    bad = api_client.post("/v1/feedback", json={"user_id": "u1", "query": "q", "rating": 9})
    assert bad.status_code == 422
    ok = api_client.post(
        "/v1/feedback", json={"user_id": "u1", "query": "q", "rating": 4, "query_id": "abc"}
    )
    assert ok.json() == {"status": "ok"}
    assert api_client.get("/v1/feedback/list").json()["results"][0]["query_id"] == "abc"


# --- ingestion -------------------------------------------------------------------


def test_ingest_starts_a_background_job(api_client, monkeypatch):
    started = []
    monkeypatch.setattr(
        api.ingest_jobs, "start_job", lambda job_id, fn, *args: started.append((job_id, args))
    )
    monkeypatch.setattr(settings, "tenant_isolation", True)
    r = api_client.post("/v1/ingest", json={"tenant": "acme"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert started == [(job_id, ("/data/docs", "acme"))]


def test_ingest_status_hidden_from_other_tenants(api_client, monkeypatch):
    _enable_auth(monkeypatch)
    job_id = ingest_jobs.create_job("ingest", meta={"tenant": "acme"})
    assert api_client.get(f"/v1/ingest/{job_id}", headers={"x-api-key": "key-b"}).status_code == 404
    assert api_client.get(f"/v1/ingest/{job_id}", headers={"x-api-key": "key-a"}).status_code == 200


def test_delete_source_uses_the_callers_tenant(api_client, monkeypatch):
    # Regression: delete ignored the tenant and removed the file everywhere.
    seen = []
    monkeypatch.setattr(
        api,
        "delete_source",
        lambda source, tenant: seen.append((source, tenant)) or {"source": source},
    )
    _enable_auth(monkeypatch)
    r = api_client.delete(
        "/v1/ingest/source/letters.txt?tenant=globex", headers={"x-api-key": "key-a"}
    )
    assert r.status_code == 200
    assert seen == [("letters.txt", "acme")]


# --- middleware, metrics, misc -------------------------------------------------


def test_admin_html_page_is_gone(api_client):
    # Regression: /ui rendered user-controlled text with innerHTML (stored XSS).
    assert api_client.get("/ui").status_code == 404


def test_rate_limit_ignores_forwarded_for_unless_trusted(api_client, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)
    codes = [
        api_client.get("/v1/feedback/list", headers={"x-forwarded-for": f"10.0.0.{i}"}).status_code
        for i in range(3)
    ]
    assert codes == [200, 200, 429]
    limited = api_client.get("/v1/feedback/list")
    assert "retry-after" in limited.headers

    import main

    main._rate_state.clear()
    monkeypatch.setattr(settings, "trusted_proxy_hops", 1)
    codes = [
        api_client.get(
            "/v1/feedback/list", headers={"x-forwarded-for": f"spoof, 10.0.0.{i}"}
        ).status_code
        for i in range(3)
    ]
    assert codes == [200, 200, 200]


def test_rate_limit_is_per_tenant_and_runs_after_auth(api_client, monkeypatch):
    _enable_auth(monkeypatch)
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)
    a, b = {"x-api-key": "key-a"}, {"x-api-key": "key-b"}
    assert api_client.get("/v1/feedback/list", headers=a).status_code == 200
    assert api_client.get("/v1/feedback/list", headers=b).status_code == 200
    assert api_client.get("/v1/feedback/list", headers=a).status_code == 429
    # Unauthenticated requests are rejected before they reach the limiter.
    assert api_client.get("/v1/feedback/list", headers={"x-api-key": "nope"}).status_code == 401


def test_metrics_use_route_templates(api_client):
    # Regression: raw paths were labels, one series per job id / scanner URL.
    api_client.get("/v1/ingest/abc")
    api_client.get("/v1/ingest/def")
    api_client.get("/definitely/not/a/route")
    text = api_client.get("/metrics").text
    assert 'sag_rag_requests_total{method="GET",path="/v1/ingest/{job_id}",status="404"} 2' in text
    assert 'path="unmatched"' in text
    assert "abc" not in text and "definitely" not in text


def test_ready_reports_dependency_status(api_client, monkeypatch):
    import agents
    from llm_client import llm

    class ES:
        def ping(self):
            return True

    def broken_qdrant():
        raise ConnectionError("down")

    async def ollama_ok():
        return True

    monkeypatch.setattr(agents, "_get_qdrant", broken_qdrant)
    monkeypatch.setattr(agents, "_get_es", lambda: ES())
    monkeypatch.setattr(llm, "ping", ollama_ok)
    r = api_client.get("/ready")
    assert r.status_code == 503
    checks = r.json()["checks"]
    assert (
        checks["qdrant"]["ok"] is False and checks["elasticsearch"]["ok"] and checks["ollama"]["ok"]
    )


def test_openapi_documents_request_bodies(api_client):
    spec = api_client.get("/openapi.json").json()
    body = spec["paths"]["/v1/query"]["post"]["requestBody"]["content"]["application/json"][
        "schema"
    ]
    assert body["$ref"].endswith("/QueryRequest")
    assert "QueryResponse" in spec["components"]["schemas"]


def test_domains_endpoint(api_client, monkeypatch):
    monkeypatch.setattr(settings, "domain_keywords", {"finance": ["revenue"]})
    assert api_client.get("/v1/domains").json() == {"domains": ["finance", "stoicism"]}


def test_verification_reaches_both_endpoints(api_client, fake_pipeline, monkeypatch):
    calls = []

    async def fake_verify(answer, evidence, provenance):
        calls.append(answer)
        return {
            "claims": [
                {
                    "text": answer,
                    "status": "supported",
                    "support_id": "c1",
                    "support_text": "Virtue is the only good, said the Stoics.",
                    "entailment": 0.9,
                    "contradiction": 0.0,
                }
            ],
            "supported_ratio": 1.0,
        }

    monkeypatch.setattr(settings, "verify_enabled", True)
    monkeypatch.setattr(pipeline.verification, "verify", fake_verify)
    body = api_client.post("/v1/query", json=_q()).json()
    assert body["hallucination_risk_method"] == "nli" and body["hallucination_risk"] == 0.0
    assert body["claims"][0]["support_id"] == "c1"
    # Regression: undeclared Claim fields were silently dropped by the response model.
    assert body["claims"][0]["support_text"] == "Virtue is the only good, said the Stoics."
    assert "verify" in body["timings_ms"]

    events = _sse_events(api_client.post("/v1/query/stream", json=_q()).text)
    stages = [(d["stage"], d["state"]) for e, d in events if e == "stage"]
    assert stages[-2:] == [("verify", "running"), ("verify", "done")]
    final_claim = dict(events)["final"]["claims"][0]
    assert final_claim["status"] == "supported"
    assert final_claim["support_text"] == "Virtue is the only good, said the Stoics."
    assert calls == ["Virtue is the only good.", "Virtue is the only good."]
