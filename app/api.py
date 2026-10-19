# app/api.py
"""HTTP routes. Request handling only: validation comes from schemas.py, the
query pipeline lives in pipeline.py, and every read/write is scoped to the
caller's tenant when tenant isolation is on."""
import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

import agents
import ingest_jobs
import namespaces
import pipeline
import redis_client
from config import settings
from continuous_learning import build_training_rows
from ingestion import delete_source, ingest_folder
from judge import build_graph_reasoning
from metrics import render_prometheus
from schemas import (
    AuditExportRequest,
    FeedbackRequest,
    GraphSummaryRequest,
    IngestRequest,
    LearningExportRequest,
    QueryRequest,
    QueryResponse,
)
from store import ANY_TENANT, fetch_audit_logs, fetch_feedback, log_feedback, write_jsonl
from synthesis import synthesize_answer_stream

_LOG = logging.getLogger(__name__)
router = APIRouter()
INGEST_FOLDER = "/data/docs"


# --- tenancy -----------------------------------------------------------------


def _resolve_tenant(request, body):
    """Tenant for a request. When the request is authenticated, the key's
    tenant (set on request.state by the auth middleware) is authoritative and
    any body-supplied tenant is ignored; this is the server-side isolation
    boundary. Otherwise fall back to the body tenant."""
    auth_tenant = getattr(getattr(request, "state", None), "tenant", None)
    if auth_tenant is not None:
        return auth_tenant
    if isinstance(body, dict):
        return body.get("tenant")
    return None


def request_tenant(request, body_tenant=None, fallback=None):
    """Storage tenant for this request, or None when isolation is off.

    Authenticated: the key's tenant. Unauthenticated (dev mode): the supplied
    tenant, then ``fallback`` (usually the user id), then "default".
    """
    if not settings.tenant_isolation:
        return None
    raw = _resolve_tenant(request, {"tenant": body_tenant}) or fallback or "default"
    return namespaces.tenant_slug(raw)


def _store_scope(tenant):
    """Store filter: all rows when isolation is off, else only the tenant's."""
    return tenant if settings.tenant_isolation else ANY_TENANT


def _ndjson(rows, filename: str) -> Response:
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return Response(
        content=body,
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --- ingestion ---------------------------------------------------------------


@router.post("/ingest", status_code=202)
async def ingest_endpoint(request: Request, body: IngestRequest | None = None):
    """Start a background ingest of the docs folder; poll GET /ingest/{job_id}."""
    tenant = request_tenant(request, body.tenant if body else None)
    job_id = ingest_jobs.create_job("ingest", meta={"tenant": tenant, "folder": INGEST_FOLDER})
    # Best-effort enqueue for observability / a future distributed worker.
    await redis_client.enqueue_json(
        settings.ingest_queue_name, {"job_id": job_id, "tenant": tenant, "folder": INGEST_FOLDER}
    )
    ingest_jobs.start_job(job_id, ingest_folder, INGEST_FOLDER, tenant)
    return {"status": "accepted", "job_id": job_id}


@router.get("/ingest/{job_id}")
async def ingest_status_endpoint(job_id: str, request: Request, tenant: str | None = None):
    job = ingest_jobs.get_job(job_id)
    if job is None or (settings.tenant_isolation and job["meta"].get("tenant") != request_tenant(request, tenant)):
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.delete("/ingest/source/{source}")
async def delete_source_endpoint(source: str, request: Request, tenant: str | None = None):
    """Delete one source from the caller's namespaces only."""
    res = await asyncio.to_thread(delete_source, source, request_tenant(request, tenant))
    return {"status": "ok", **res}


# --- query -------------------------------------------------------------------


def _prepare(body: QueryRequest, request: Request):
    tenant = request_tenant(request, body.tenant, body.user_id)
    try:
        prefs = pipeline.validated_preferences(body.preferences.model_dump(), tenant)
    except pipeline.UnknownDomain as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    cache_key = pipeline.query_cache_key(body.query, prefs, tenant) if settings.redis_cache_enabled else None
    return tenant, prefs, cache_key


async def _cached_response(cache_key, body: QueryRequest, tenant):
    """A cached answer re-issued under a fresh query_id (and audited)."""
    if cache_key is None:
        return None
    cached = await redis_client.cache_get_json(cache_key)
    if not isinstance(cached, dict) or "answer" not in cached:
        return None
    cached.update({"cache": "hit", "user_id": body.user_id, "query_id": uuid.uuid4().hex})
    await pipeline.audit(cached, tenant)
    return cached


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest, request: Request):
    tenant, prefs, cache_key = _prepare(body, request)
    cached = await _cached_response(cache_key, body, tenant)
    if cached is not None:
        return cached
    ctx = await pipeline.run_stages(pipeline.QueryContext(query=body.query, prefs=prefs, tenant=tenant))
    synthesis = await pipeline.synthesize(ctx)
    await pipeline.verify_stage(ctx, synthesis)
    response = pipeline.finalize(ctx, synthesis, user_id=body.user_id, query_id=uuid.uuid4().hex)
    await pipeline.audit(response, tenant)
    if cache_key is not None:
        response["cache"] = "miss"
        await redis_client.cache_set_json(cache_key, response, settings.query_cache_ttl_s)
    return response


def _sse(event: str, data) -> str:
    """Format a Server-Sent Events message."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@router.post("/query/stream")
async def query_stream_endpoint(body: QueryRequest, request: Request):
    """Streaming variant of /query.

    Events: ``stage`` (pipeline progress), ``meta`` (domain/intent before the
    answer), ``token`` (answer deltas), ``final`` (the same body /query
    returns), ``error``, and ``done``.
    """
    tenant, prefs, cache_key = _prepare(body, request)

    async def event_stream():
        cached = await _cached_response(cache_key, body, tenant)
        if cached is not None:
            yield _sse("stage", {"stage": "cache", "state": "done", "ms": 0})
            yield _sse("meta", {k: cached.get(k) for k in ("query_id", "domain", "intent", "author_terms", "author_gap")})
            yield _sse("token", {"text": cached.get("answer", "")})
            yield _sse("final", cached)
            yield _sse("done", {})
            return
        ctx = pipeline.QueryContext(query=body.query, prefs=prefs, tenant=tenant)
        query_id = uuid.uuid4().hex
        try:
            async for event in pipeline.iter_stages(ctx):
                yield _sse("stage", event)
            yield _sse("meta", {
                "query_id": query_id,
                "domain": ctx.domain,
                "intent": ctx.plan.get("intent"),
                "author_terms": ctx.author_terms,
                "author_gap": ctx.author_gap,
            })
            yield _sse("stage", {"stage": "synthesize", "state": "running"})
            loop = asyncio.get_running_loop()
            started = loop.time()
            parts = []
            if settings.enable_synthesis:
                async for delta in synthesize_answer_stream(
                    ctx.query, ctx.reranked, ctx.judge_output, ctx.author_terms, ctx.author_gap
                ):
                    parts.append(delta)
                    yield _sse("token", {"text": delta})
            ctx.timings_ms["synthesize"] = int((loop.time() - started) * 1000)
            yield _sse("stage", {"stage": "synthesize", "state": "done", "ms": ctx.timings_ms["synthesize"]})
            synthesis = pipeline.stream_synthesis_result(ctx, "".join(parts).strip())
            if settings.verify_enabled:
                yield _sse("stage", {"stage": "verify", "state": "running"})
                await pipeline.verify_stage(ctx, synthesis)
                yield _sse("stage", {"stage": "verify", "state": "done", "ms": ctx.timings_ms.get("verify", 0)})
            response = pipeline.finalize(ctx, synthesis, user_id=body.user_id, query_id=query_id)
            response = QueryResponse.model_validate(response).model_dump(mode="json")
            await pipeline.audit(response, tenant)
            if cache_key is not None:
                response["cache"] = "miss"
                await redis_client.cache_set_json(cache_key, response, settings.query_cache_ttl_s)
            yield _sse("final", response)
        except Exception:
            # The 200 status is already sent; report the failure in-band.
            _LOG.exception("query_stream_failed")
            yield _sse("error", {"detail": "the query failed; see server logs"})
        yield _sse("done", {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/domains")
async def domains_endpoint(request: Request, tenant: str | None = None):
    """Domains the caller can pass as ``preferences.domain``."""
    scoped = request_tenant(request, tenant)
    return {"domains": sorted(await asyncio.to_thread(agents.known_domains, scoped))}


# --- audit, feedback, learning -----------------------------------------------


@router.get("/audit")
async def audit_endpoint(
    request: Request,
    user_id: str | None = None,
    limit: int = 50,
    cursor: str | None = None,
    tenant: str | None = None,
):
    limit = max(1, min(limit, 1000))
    scope = _store_scope(request_tenant(request, tenant))
    results = await asyncio.to_thread(fetch_audit_logs, limit=limit, user_id=user_id, cursor=cursor, tenant=scope)
    next_token = results[-1]["cursor"] if results else None
    return {"results": results, "next_token": next_token}


@router.post("/audit/export")
async def audit_export_endpoint(body: AuditExportRequest, request: Request):
    """Download the caller's audit log as NDJSON, or (``save``) write it to the
    server's configured export path. The client never chooses a path."""
    scope = _store_scope(request_tenant(request, body.tenant))
    rows = await asyncio.to_thread(fetch_audit_logs, limit=body.limit, user_id=body.user_id, tenant=scope)
    if body.save:
        return await asyncio.to_thread(write_jsonl, rows, settings.audit_export_path)
    return _ndjson(rows, "audit-export.jsonl")


@router.post("/feedback")
async def feedback_endpoint(body: FeedbackRequest, request: Request):
    tenant = request_tenant(request, body.tenant, body.user_id)
    stored = await asyncio.to_thread(
        log_feedback, user_id=body.user_id, query=body.query, rating=body.rating,
        comment=body.comment, tenant=tenant, query_id=body.query_id,
    )
    return {"status": "ok" if stored else "not_stored"}


@router.get("/feedback/list")
async def feedback_list_endpoint(request: Request, limit: int = 50, tenant: str | None = None):
    limit = max(1, min(limit, 1000))
    scope = _store_scope(request_tenant(request, tenant))
    return {"results": await asyncio.to_thread(fetch_feedback, limit=limit, tenant=scope)}


@router.post("/learning/export")
async def learning_export_endpoint(request: Request, body: LearningExportRequest | None = None):
    """Highly rated interactions as training data (NDJSON download, or
    ``save`` to the configured learning export path)."""
    body = body or LearningExportRequest()
    scope = _store_scope(request_tenant(request, body.tenant))
    min_rating = body.min_rating if body.min_rating is not None else settings.learning_min_rating
    rows = await asyncio.to_thread(build_training_rows, limit=body.limit, min_rating=min_rating, tenant=scope)
    if body.save:
        return await asyncio.to_thread(write_jsonl, rows, settings.learning_export_path)
    return _ndjson(rows, "training-data.jsonl")


@router.post("/graph/summary")
async def graph_summary_endpoint(body: GraphSummaryRequest, request: Request):
    tenant = request_tenant(request, body.tenant)
    return {"graph_reasoning": await asyncio.to_thread(build_graph_reasoning, body.chunk_ids, tenant)}


@router.get("/metrics")
def metrics_endpoint():
    return Response(content=render_prometheus(), media_type="text/plain; version=0.0.4")
