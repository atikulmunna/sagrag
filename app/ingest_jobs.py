# app/ingest_jobs.py
"""In-process registry for background ingest jobs.

Ingestion is heavy and synchronous, so the endpoint enqueues a job and returns
immediately; the work runs off the event loop via ``asyncio.to_thread`` and its
status is polled via a status endpoint. Jobs run one at a time per process (two
concurrent re-ingests of the same source would race on delete-then-upsert).
The registry is per-process (single-node assumption); the job payload is also
best-effort enqueued to Redis so a distributed worker can be added later.
"""
import asyncio
import logging
import time
import uuid

_LOG = logging.getLogger(__name__)
_JOBS: dict = {}
# Cap the registry so a long-running server doesn't grow unbounded.
_MAX_JOBS = 200
# Strong references: the event loop only keeps weak references to tasks, so an
# unreferenced background task can be garbage-collected mid-run.
_TASKS: set = set()
_RUN_LOCK = None
_RUN_LOCK_LOOP = None


def _run_lock():
    global _RUN_LOCK, _RUN_LOCK_LOOP
    loop = asyncio.get_running_loop()
    if _RUN_LOCK is None or _RUN_LOCK_LOOP is not loop:
        _RUN_LOCK, _RUN_LOCK_LOOP = asyncio.Lock(), loop
    return _RUN_LOCK


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
    async with _run_lock():
        job["status"] = "running"
        job["started_at"] = time.time()
        try:
            result = await asyncio.to_thread(fn, *args, **kwargs)
            job["result"] = result
            has_errors = isinstance(result, dict) and result.get("error_count")
            job["status"] = "completed_with_errors" if has_errors else "succeeded"
        except Exception as e:
            _LOG.exception("ingest job %s failed", job_id)
            job["error"] = str(e)
            job["status"] = "failed"
        finally:
            job["finished_at"] = time.time()
    return job


def start_job(job_id: str, fn, *args, **kwargs) -> asyncio.Task:
    """Schedule ``run_job`` in the background, keeping a reference to the task."""
    task = asyncio.create_task(run_job(job_id, fn, *args, **kwargs))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return task
