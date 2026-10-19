import asyncio
import logging
import threading

from config import settings

_LOG = logging.getLogger(__name__)
_RERANKER = None
_LOCK = threading.Lock()


def _load_reranker():
    global _RERANKER
    if _RERANKER is None:
        with _LOCK:
            if _RERANKER is None:
                from sentence_transformers import CrossEncoder
                _RERANKER = CrossEncoder(settings.reranker_model_name)
    return _RERANKER


async def rerank(query, docs, top_k=None):
    if not docs:
        return []
    if top_k is None:
        top_k = settings.reranker_top_k

    def _sync_rerank():
        model = _load_reranker()
        pairs = [(query, d["text"]) for d in docs]
        scores = model.predict(pairs)
        scored = []
        for d, s in zip(docs, scores):
            item = dict(d)
            item["rerank_score"] = float(s)
            scored.append(item)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]

    try:
        return await asyncio.to_thread(_sync_rerank)
    except Exception as e:
        # fallback: return original docs (trimmed) if model load/predict fails
        _LOG.warning("rerank_failed, using retrieval order: %r", e)
        return [dict(d) for d in docs[:top_k]]
