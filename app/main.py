# app/main.py
import json
import logging
import time
from fastapi import FastAPI, Request, HTTPException
from api import router
from ui import router as ui_router
from metrics import record_request
from otel import setup_tracing
from config import settings
import redis_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sag_rag")

app = FastAPI(title="SAG Backend")
app.include_router(router, prefix="/v1")
app.include_router(router)
app.include_router(ui_router)

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

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path in ["/health", "/metrics", "/v1/metrics"]:
        return await call_next(request)
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return await call_next(request)
    key = request.headers.get("x-forwarded-for") or request.client.host
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
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    return await call_next(request)

@app.middleware("http")
async def request_metrics(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = int((time.monotonic() - start) * 1000)
    record_request(request.method, request.url.path, response.status_code, elapsed_ms)
    logger.info(json.dumps({
        "event": "request",
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "elapsed_ms": elapsed_ms,
    }))
    return response

@app.get("/health")
def health():
    return {"status": "ok"}
