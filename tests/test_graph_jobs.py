"""Pure graph helpers (no Neo4j) and ingest job status reporting."""

import asyncio

import graph
import ingest_jobs


def test_tenant_key():
    assert graph.tenant_key(None) == ""
    assert graph.tenant_key("acme") == "acme"
    assert graph.tenant_key("user@example.com").startswith("h-")


def test_find_contradictions_requires_negation_flip_and_overlap():
    claim_meta = [("new", "Seneca does not fear death because death is natural")]
    recent = [
        {"id": "old1", "text": "Seneca does fear death because death is natural"},
        {
            "id": "old2",
            "text": "Seneca does not fear death because death is natural",
        },  # same polarity
        {"id": "old3", "text": "Completely unrelated sentence about markets"},
    ]
    assert graph.find_contradictions(claim_meta, recent, ["Seneca"]) == [("new", "old1")]


def test_summarize_reasoning_flags_conflicts_and_scores():
    out = graph.summarize_reasoning(
        claims=[{"id": "c1", "text": "x"}],
        contra=[{"id": "c1", "contradict_count": 2}],
        entities=[],
        chunk_scores=[
            {"chunk_id": "k1", "support_count": 3, "contradict_count": 1, "entity_count": 10}
        ],
        paths=[],
        relations=[
            {"src": "A", "predicate": "is", "dst": "B", "rel_count": 2},
            {"src": "A", "predicate": "has", "dst": "B", "rel_count": 1},
        ],
    )
    assert out["claims"][0]["contradict_count"] == 2
    assert out["relation_strength"] == [{"relation": "A|is|B", "count": 2}]
    assert out["relation_conflicts"] == [{"pair": "A|B", "predicates": ["has", "is"]}]
    assert out["evidence_scores"][0]["score"] == 3.5


async def test_job_reports_partial_failure():
    job_id = ingest_jobs.create_job("ingest")
    await ingest_jobs.run_job(
        job_id, lambda: {"ingested_chunks": 3, "error_count": 1, "errors": [{"stage": "es_bulk"}]}
    )
    assert ingest_jobs.get_job(job_id)["status"] == "completed_with_errors"


async def test_start_job_keeps_a_reference_until_done():
    # The loop only holds weak references to tasks; start_job must hold a
    # strong one until the job finishes.
    job_id = ingest_jobs.create_job("ingest")
    task = ingest_jobs.start_job(job_id, lambda: {"ok": True})
    assert task in ingest_jobs._TASKS
    await task
    await asyncio.sleep(0)
    assert task not in ingest_jobs._TASKS
    assert ingest_jobs.get_job(job_id)["status"] == "succeeded"
