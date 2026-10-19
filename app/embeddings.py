# app/embeddings.py
"""The one place the sentence-transformers embedding model is loaded.

Retrieval, ingestion, and caching all go through here so the model is loaded
once per process and its dimension comes from the model rather than a
hard-coded constant.
"""
import asyncio
import hashlib
import logging
import threading

import redis_client
from config import settings

_LOG = logging.getLogger(__name__)
_MODEL = None
_LOCK = threading.Lock()


def get_model():
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                from sentence_transformers import SentenceTransformer

                _MODEL = SentenceTransformer(settings.embedding_model_name)
    return _MODEL


def dimension() -> int:
    return int(get_model().get_sentence_embedding_dimension())


def encode(texts: list[str]) -> list[list[float]]:
    """Synchronous batch encode (call from a worker thread)."""
    if not texts:
        return []
    return [v.tolist() for v in get_model().encode(list(texts))]


def _cache_key(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sagrag:embed:{settings.embedding_model_name}:{digest}"


async def encode_cached(texts: list[str]) -> list[list[float]]:
    """Encode texts, reusing Redis-cached vectors when available.

    Misses are computed in one batch off the event loop. Redis is best-effort:
    when it is disabled or down every text is simply computed.
    """
    texts = [str(t) for t in texts]
    vectors: list = [None] * len(texts)
    if settings.redis_cache_enabled:
        for i, t in enumerate(texts):
            cached = await redis_client.cache_get_json(_cache_key(t))
            if isinstance(cached, list):
                vectors[i] = cached
    missing = [i for i, v in enumerate(vectors) if v is None]
    if missing:
        computed = await asyncio.to_thread(encode, [texts[i] for i in missing])
        for i, vec in zip(missing, computed):
            vectors[i] = vec
            if settings.redis_cache_enabled:
                await redis_client.cache_set_json(_cache_key(texts[i]), vec, settings.embed_cache_ttl_s)
    return vectors
