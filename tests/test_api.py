import os
import sys
from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
APP_DIR = os.path.join(ROOT, "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from app.main import app
import app.api as api

client = TestClient(app)

def test_query_happy_path(monkeypatch):
    async def _plan_query(_query):
        return {"intent": "test", "queries": ["q1"], "constraints": {}}
    async def _run_agents(_queries):
        return [{
            "id": "doc_1",
            "score": 0.9,
            "payload": {"text": "alpha beta", "source": "file.txt"},
            "agent": "vector",
            "elapsed_ms": 5,
        }]
    async def _rerank(_query, docs):
        return docs
    async def _judge(_query, _evidence, _graph):
        return {"confidence": 0.8, "trusted_ids": ["doc_1"], "notes": "ok"}
    async def _synth(_query, _evidence, _judge):
        return {"answer": "ok", "provenance": [{"id": "doc_1"}], "confidence": 0.8, "explain_trace": "t"}

    monkeypatch.setattr(api, "plan_query", _plan_query)
    monkeypatch.setattr(api, "run_agents", _run_agents)
    monkeypatch.setattr(api, "rerank", _rerank)
    monkeypatch.setattr(api, "judge_evidence", _judge)
    monkeypatch.setattr(api, "synthesize_answer", _synth)

    resp = client.post("/v1/query", json={"user_id": "u1", "query": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "ok"
    assert body["judge"]["confidence"] == 0.8
    assert body["results"][0]["id"] == "doc_1"

def test_query_missing_fields():
    resp = client.post("/v1/query", json={"query": "hello"})
    assert resp.status_code == 422
