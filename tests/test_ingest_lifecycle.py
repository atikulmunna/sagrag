"""Tests for the ingestion lifecycle additions: name derivation, batched
Qdrant upserts, and the background job registry.

The store-touching paths (ES/Qdrant/graph delete + bulk) need live services;
here we cover the pure name logic, batching with a fake client, and the job
registry state machine.
"""

import ingest_jobs
import ingestion


# --- _target_names ----------------------------------------------------------


def test_target_names_base():
    assert ingestion._target_names(None, None) == ("docs", "docs_index")


def test_target_names_tenant():
    assert ingestion._target_names("acme", None) == ("docs_acme", "docs_index_acme")


def test_target_names_domain_resets_off_base():
    # A domain resets names off the base, dropping the tenant suffix (preserved
    # legacy behavior).
    assert ingestion._target_names("acme", "stoicism") == ("docs_stoicism", "docs_index_stoicism")


# --- _batch_upsert_qdrant ---------------------------------------------------


class FakeQdrant:
    def __init__(self):
        self.upserts = []

    def upsert(self, collection_name, points):
        self.upserts.append((collection_name, list(points)))


def test_batch_upsert_splits_into_batches():
    fake = FakeQdrant()
    points = [{"id": i} for i in range(5)]
    ingestion._batch_upsert_qdrant(fake, "docs", points, batch_size=2)
    sizes = [len(pts) for _, pts in fake.upserts]
    assert sizes == [2, 2, 1]
    assert all(coll == "docs" for coll, _ in fake.upserts)


def test_batch_upsert_swallows_errors():
    class Boom:
        def upsert(self, **kwargs):
            raise RuntimeError("qdrant down")

    # Should not raise (best-effort ingest).
    ingestion._batch_upsert_qdrant(Boom(), "docs", [{"id": 1}], batch_size=10)


# --- ingest_jobs ------------------------------------------------------------


def test_job_create_and_get():
    job_id = ingest_jobs.create_job("ingest", meta={"tenant": "t1"})
    job = ingest_jobs.get_job(job_id)
    assert job["status"] == "queued"
    assert job["meta"] == {"tenant": "t1"}
    assert ingest_jobs.get_job("missing") is None


async def test_run_job_success():
    job_id = ingest_jobs.create_job("ingest")

    def work(a, b):
        return {"sum": a + b}

    await ingest_jobs.run_job(job_id, work, 2, 3)
    job = ingest_jobs.get_job(job_id)
    assert job["status"] == "succeeded"
    assert job["result"] == {"sum": 5}
    assert job["finished_at"] is not None


async def test_run_job_failure_is_captured():
    job_id = ingest_jobs.create_job("ingest")

    def boom():
        raise ValueError("bad file")

    await ingest_jobs.run_job(job_id, boom)
    job = ingest_jobs.get_job(job_id)
    assert job["status"] == "failed"
    assert "bad file" in job["error"]
