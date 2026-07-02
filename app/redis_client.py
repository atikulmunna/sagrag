# app/redis_client.py
"""Best-effort async Redis helpers.

Every operation degrades to a no-op / ``None`` / fallback when Redis is
disabled (empty ``redis_url``) or unreachable, so the application never fails
because of Redis. The client is created lazily and cached; connections are
lazy in redis-py, so importing this module has no side effects and tests that
never touch a live Redis don't need the ``redis`` package installed.
"""
import json
import logging

from config import settings

_LOG = logging.getLogger(__name__)
_client = None


async def get_redis():
    """Return a cached async Redis client, or None if disabled/unavailable."""
    global _client
    if not settings.redis_url:
        return None
    if _client is None:
        try:
            import redis.asyncio as aioredis

            _client = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=0.5,
                socket_timeout=0.5,
            )
        except Exception as e:  # pragma: no cover - import/config failure
            _LOG.warning("redis_init_failed: %s", e)
            return None
    return _client


async def incr_fixed_window(key: str, ttl_s: int):
    """Increment a fixed-window counter; return the count, or None on failure."""
    client = await get_redis()
    if client is None:
        return None
    try:
        count = await client.incr(key)
        if count == 1:
            await client.expire(key, ttl_s)
        return int(count)
    except Exception as e:
        _LOG.warning("redis_incr_failed: %s", e)
        return None


async def cache_get_json(key: str):
    """Return a cached JSON value, or None if missing/unavailable."""
    client = await get_redis()
    if client is None:
        return None
    try:
        raw = await client.get(key)
        return json.loads(raw) if raw else None
    except Exception as e:
        _LOG.warning("redis_get_failed: %s", e)
        return None


async def cache_set_json(key: str, value, ttl_s: int) -> bool:
    """Best-effort cache write. Returns True on success, False otherwise."""
    client = await get_redis()
    if client is None:
        return False
    try:
        await client.set(key, json.dumps(value), ex=ttl_s)
        return True
    except Exception as e:
        _LOG.warning("redis_set_failed: %s", e)
        return False


async def enqueue_json(queue: str, payload) -> bool:
    """Best-effort RPUSH of a JSON payload onto a list. Returns success."""
    client = await get_redis()
    if client is None:
        return False
    try:
        await client.rpush(queue, json.dumps(payload))
        return True
    except Exception as e:
        _LOG.warning("redis_enqueue_failed: %s", e)
        return False
