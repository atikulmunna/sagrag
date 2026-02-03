# app/llm_client.py
import asyncio
import random
import contextlib
from typing import List
import httpx

from config import settings
try:
    from opentelemetry import trace
    _TRACER = trace.get_tracer(__name__)
except Exception:
    _TRACER = None

# Local embedder (fast, offline, deterministic)
_EMBED_MODEL = None
def _load_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(settings.embedding_model_name)
    return _EMBED_MODEL

class OllamaREST:
    """Minimal Ollama REST wrapper (generate endpoint)."""
    def __init__(self):
        self.base_url = settings.ollama_url.rstrip("/")
        self.model = settings.ollama_model
        self._sema = asyncio.Semaphore(settings.llm_max_concurrent)

    async def completion(self, prompt: str, max_tokens: int = 512) -> str:
        span_cm = _TRACER.start_as_current_span("llm.completion") if _TRACER else contextlib.nullcontext()
        with span_cm as span:
            if span:
                span.set_attribute("llm.model", self.model)
                span.set_attribute("llm.max_tokens", max_tokens)
                span.set_attribute("llm.provider", "ollama")
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens}
            }
            attempt = 0
            while True:
                attempt += 1
                async with self._sema:
                    async with httpx.AsyncClient(timeout=60) as client:
                        resp = await client.post(f"{self.base_url}/api/generate", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("response", "")
                    return text
                if attempt >= settings.llm_max_retries:
                    if span:
                        span.set_attribute("http.status_code", resp.status_code)
                    raise RuntimeError(f"Ollama error: {resp.status_code} {resp.text}")
                backoff = settings.llm_retry_base_s * (2 ** (attempt - 1))
                backoff = backoff + random.uniform(0, 0.25)
                await asyncio.sleep(backoff)

class LLMClient:
    """Facade: generation (Ollama REST) + local embeddings (SentenceTransformer)."""
    def __init__(self):
        self.gen = OllamaREST()
        self._embed_model = None

    # GENERATION
    async def completion(self, prompt: str, max_tokens: int = 512) -> str:
        return await self.gen.completion(prompt, max_tokens=max_tokens)

    # EMBEDDINGS (local)
    async def embed(self, text: str):
        # run embedding in thread pool because sentence-transformers is sync heavy
        def _sync_embed(t):
            model = _load_embed_model()
            vec = model.encode(t)
            return vec.tolist()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_embed, text)

    async def embed_many(self, texts: List[str]):
        def _sync_embed_many(ts):
            model = _load_embed_model()
            mats = model.encode(ts)
            return [v.tolist() for v in mats]
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_embed_many, texts)

# single shared client instance
llm = LLMClient()
