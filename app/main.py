# app/main.py
import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.routing import Match

import redis_client
import security
import store
from api import router
from config import settings
from llm_client import llm
from metrics import record_request
from otel import setup_tracing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sag_rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(store.init_db)
    await asyncio.to_thread(store.init_feedback_db)
    if settings.graph_enabled:
        try:
            import graph
            await asyncio.to_thread(graph.ensure_schema)
        except Exception as e:
            logger.warning("neo4j schema setup failed: %s", e)
    if settings.warm_models_on_startup:
        # Load models in the background; startup does not wait for them.
        app.state.model_warmup = asyncio.get_running_loop().run_in_executor(None, warm_models)
    if settings.tenant_isolation and not settings.auth_enabled:
        logger.warning(
            "TENANT_ISOLATION is on without AUTH_ENABLED: callers choose their own tenant, "
            "so isolation is not a security boundary in this mode."
        )
    yield
    await llm.aclose()
    await redis_client.close()
    if settings.graph_enabled:
        import graph
        graph.close_driver()


def warm_models():
    """Load the embedding, reranker and (when enabled) NLI models. Best effort."""
    import embeddings
    import reranker

    steps = [("embeddings", embeddings.get_model), ("reranker", reranker._load_reranker)]
    if settings.verify_enabled:
        import verification

        steps.append(("verification", verification.warm_up))
    for name, step in steps:
        try:
            step()
            logger.info("model warm-up done: %s", name)
        except Exception as e:  # pragma: no cover - logged, never fatal
            logger.warning("model warm-up failed for %s: %s", name, e)


app = FastAPI(title="SAG-RAG Backend", lifespan=lifespan)
app.include_router(router, prefix="/v1")
# Root aliases kept for backwards compatibility.
app.include_router(router)

setup_tracing(app)

_rate_state = {}


def _in_memory_count(key: str, window: int) -> int:
    """Fixed-window counter in process memory (single-worker fallback)."""
    bucket = _rate_state.setdefault(window, {})
    count = bucket.get(key, 0) + 1
    bucket[key] = count
    # prune older windows
    for ts in list(_rate_state.keys()):
        if ts < window:
            _rate_state.pop(ts, None)
    return count


def client_ip(request: Request) -> str:
    """Caller address. X-Forwarded-For is only honored for the configured
    number of trusted proxy hops (each proxy appends the address it saw, so the
    entry ``hops`` from the right is the one our own proxy recorded)."""
    hops = settings.trusted_proxy_hops
    if hops > 0:
        chain = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if len(chain) >= hops:
            return chain[-hops]
    return request.client.host if request.client else "unknown"


def rate_limit_key(request: Request) -> str:
    tenant = getattr(request.state, "tenant", None)
    return f"t:{tenant}" if tenant else f"ip:{client_ip(request)}"


def _full_template(route, path: str) -> str | None:
    """Matched route template including any router prefix.

    Newer FastAPI includes routers lazily, so a route only knows its own path
    ("/ingest/{job_id}"); the static prefix segments ("/v1") are taken from
    the request path, which routing has already matched against them.
    """
    template = getattr(route, "path", None)
    if not template:
        return None
    if ":path}" in template:
        return template
    actual = [s for s in path.split("/") if s]
    own = [s for s in template.split("/") if s]
    extra = len(actual) - len(own)
    return "/" + "/".join(actual[:extra] + own) if extra > 0 else template


def route_template(request: Request) -> str:
    """Bounded-cardinality path label: the matched route template, not the raw
    path (so /ingest/{job_id} is one series, and scanner 404s are one series)."""
    path = request.scope.get("path", "")
    template = _full_template(request.scope.get("route"), path)
    if template:
        return template
    # Requests answered by middleware (401/429) never reached the router.
    for r in request.app.router.routes:
        match, child = r.matches(request.scope)
        if match == Match.FULL:
            return _full_template(child.get("route", r), path) or "unmatched"
    return "unmatched"


# Middleware registration order matters: the last registered runs first.
# Effective order per request: metrics -> auth -> rate limit -> route.


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path in ("/health", "/ready", "/metrics", "/v1/metrics"):
        return await call_next(request)
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return await call_next(request)
    key = rate_limit_key(request)
    now = int(time.time())
    window = now - (now % 60)
    # Prefer a shared Redis counter so the limit holds across workers/replicas;
    # fall back to the in-process counter if Redis is disabled/unreachable.
    count = None
    if settings.redis_rate_limit_enabled:
        count = await redis_client.incr_fixed_window(f"sagrag:rl:{window}:{key}", 60)
    if count is None:
        count = _in_memory_count(key, window)
    if count > limit:
        # Return (don't raise): HTTPException raised inside a BaseHTTPMiddleware
        # escapes FastAPI's exception handler and surfaces as a 500.
        return JSONResponse(
            status_code=429,
            content={"detail": "rate limit exceeded"},
            headers={"Retry-After": str(60 - (now % 60))},
        )
    return await call_next(request)


@app.middleware("http")
async def authenticate(request: Request, call_next):
    """API-key gate. No-op unless settings.auth_enabled. Exempt paths (health,
    readiness, metrics, OpenAPI docs) are always open. On success, binds the
    key's tenant to request.state so endpoints enforce isolation server-side."""
    if settings.auth_enabled and request.method != "OPTIONS" and not security.is_exempt(request.url.path):
        tenant = security.authenticate(request.headers.get(security.API_KEY_HEADER))
        if tenant is None:
            return JSONResponse(status_code=401, content={"detail": "invalid or missing api key"})
        request.state.tenant = tenant
    return await call_next(request)


@app.middleware("http")
async def request_metrics(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = int((time.monotonic() - start) * 1000)
    path = route_template(request)
    record_request(request.method, path, response.status_code, elapsed_ms)
    logger.info(json.dumps({
        "event": "request",
        "method": request.method,
        "path": path,
        "status": response.status_code,
        "elapsed_ms": elapsed_ms,
    }))
    return response


_cors = [o.strip() for o in (settings.cors_origins or "").split(",") if o.strip()]
if _cors:
    # Registered last so it is outermost: preflight requests are answered
    # before the auth gate sees them.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["content-type", security.API_KEY_HEADER],
    )


@app.get("/health")
def health():
    """Liveness: the process is up."""
    return {"status": "ok"}


async def _check(fn, timeout_s=3.0):
    try:
        result = await asyncio.wait_for(fn(), timeout=timeout_s)
        return {"ok": bool(result) if result is not None else True}
    except Exception as e:
        return {"ok": False, "error": type(e).__name__}


@app.get("/ready")
async def ready():
    """Readiness: required dependencies answer. 503 if any required one fails."""
    import agents

    checks = {
        "qdrant": await _check(lambda: asyncio.to_thread(lambda: agents._get_qdrant().get_collections())),
        "elasticsearch": await _check(lambda: asyncio.to_thread(lambda: agents._get_es().ping())),
        "ollama": await _check(llm.ping),
    }
    required = ["qdrant", "elasticsearch", "ollama"]
    if settings.graph_enabled:
        import graph
        checks["neo4j"] = await _check(lambda: asyncio.to_thread(lambda: graph.get_driver().verify_connectivity()))
        required.append("neo4j")
    if settings.redis_url:
        checks["redis"] = await _check(redis_client.ping)  # optional: degrades gracefully
    ok = all(checks[name]["ok"] for name in required)
    return JSONResponse(status_code=200 if ok else 503, content={"status": "ready" if ok else "degraded", "checks": checks})
