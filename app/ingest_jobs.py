# app/ingest_jobs.py
"""In-process registry for background ingest jobs.

Ingestion is heavy and synchronous, so the endpoint enqueues a job and returns
immediately; the work runs off the event loop via ``asyncio.to_thread`` and its
status is polled via a status endpoint. The registry is per-process (single-node
assumption); a Redis-backed store for multi-worker visibility is a future
enhancement — the job payload is also best-effort enqueued to Redis so a
distributed worker can be added later.
"""
import asyncio
import time
import uuid

_JOBS: dict = {}
# Cap the registry so a long-running server doesn't grow unbounded.
_MAX_JOBS = 200


def create_job(kind: str, meta: dict | None = None) -> str:
    job_id = uuid.uuid4().hex
    _JOBS[job_id] = {
        "id": job_id,
        "kind": kind,
        "status": "queued",
        "meta": meta or {},
        "result": None,
        "error": None,
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
    }
    _prune()
    return job_id


def get_job(job_id: str):
    return _JOBS.get(job_id)


def _prune():
    if len(_JOBS) <= _MAX_JOBS:
        return
    # Drop the oldest finished jobs first.
    finished = sorted(
        (j for j in _JOBS.values() if j["finished_at"]),
        key=lambda j: j["finished_at"],
    )
    for job in finished[: len(_JOBS) - _MAX_JOBS]:
        _JOBS.pop(job["id"], None)


async def run_job(job_id: str, fn, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` in a worker thread, tracking status."""
    job = _JOBS.get(job_id)
    if job is None:
        return
    job["status"] = "running"
    job["started_at"] = time.time()
    try:
        job["result"] = await asyncio.to_thread(fn, *args, **kwargs)
        job["status"] = "succeeded"
    except Exception as e:
        job["error"] = str(e)
        job["status"] = "failed"
    finally:
        job["finished_at"] = time.time()
    return job
