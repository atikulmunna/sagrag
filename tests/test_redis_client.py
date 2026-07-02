"""Tests for the best-effort Redis helpers.

No live Redis: the happy path uses a fake async client injected into the
module-level `_client`, and the degradation paths assert that a disabled or
failing Redis returns None/False instead of raising.
"""

import json

import redis_client


class FakeAsyncRedis:
    def __init__(self, raises=False):
        self.store = {}
        self.expires = {}
        self.lists = {}
        self.raises = raises

    async def incr(self, key):
        if self.raises:
            raise RuntimeError("redis down")
        self.store[key] = int(self.store.get(key, 0)) + 1
        return self.store[key]

    async def expire(self, key, ttl):
        if self.raises:
            raise RuntimeError("redis down")
        self.expires[key] = ttl
        return True

    async def get(self, key):
        if self.raises:
            raise RuntimeError("redis down")
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        if self.raises:
            raise RuntimeError("redis down")
        self.store[key] = value
        if ex:
            self.expires[key] = ex
        return True

    async def rpush(self, key, value):
        if self.raises:
            raise RuntimeError("redis down")
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])


def _use_fake(monkeypatch, fake):
    monkeypatch.setattr(redis_client.settings, "redis_url", "redis://fake:6379/0")
    monkeypatch.setattr(redis_client, "_client", fake)


# --- disabled / degradation -------------------------------------------------


async def test_get_redis_none_when_url_empty(monkeypatch):
    monkeypatch.setattr(redis_client.settings, "redis_url", "")
    monkeypatch.setattr(redis_client, "_client", None)
    assert await redis_client.get_redis() is None


async def test_helpers_return_fallback_when_disabled(monkeypatch):
    monkeypatch.setattr(redis_client.settings, "redis_url", "")
    monkeypatch.setattr(redis_client, "_client", None)
    assert await redis_client.incr_fixed_window("k", 60) is None
    assert await redis_client.cache_get_json("k") is None
    assert await redis_client.cache_set_json("k", {"a": 1}, 60) is False
    assert await redis_client.enqueue_json("q", {"a": 1}) is False


async def test_helpers_swallow_errors(monkeypatch):
    _use_fake(monkeypatch, FakeAsyncRedis(raises=True))
    assert await redis_client.incr_fixed_window("k", 60) is None
    assert await redis_client.cache_get_json("k") is None
    assert await redis_client.cache_set_json("k", {"a": 1}, 60) is False
    assert await redis_client.enqueue_json("q", {"a": 1}) is False


# --- happy path -------------------------------------------------------------


async def test_incr_fixed_window_counts_and_sets_ttl(monkeypatch):
    fake = FakeAsyncRedis()
    _use_fake(monkeypatch, fake)
    assert await redis_client.incr_fixed_window("win", 60) == 1
    assert fake.expires["win"] == 60  # ttl set on first increment
    assert await redis_client.incr_fixed_window("win", 60) == 2


async def test_cache_json_round_trip(monkeypatch):
    fake = FakeAsyncRedis()
    _use_fake(monkeypatch, fake)
    assert await redis_client.cache_set_json("k", {"answer": "hi"}, 300) is True
    assert fake.expires["k"] == 300
    assert await redis_client.cache_get_json("k") == {"answer": "hi"}


async def test_cache_get_missing_is_none(monkeypatch):
    _use_fake(monkeypatch, FakeAsyncRedis())
    assert await redis_client.cache_get_json("absent") is None


async def test_enqueue_json_pushes(monkeypatch):
    fake = FakeAsyncRedis()
    _use_fake(monkeypatch, fake)
    assert await redis_client.enqueue_json("jobs", {"source": "x"}) is True
    assert json.loads(fake.lists["jobs"][0]) == {"source": "x"}
