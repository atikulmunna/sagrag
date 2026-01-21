# app/llm_client.py
import json
import asyncio
import random
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

class GeminiREST:
    """Minimal Gemini REST wrapper for text generation (generateContent endpoint)."""
    def __init__(self):
        self.api_key = settings.gemini_api_key
        self.model = settings.gemini_model_name
        self.base_url = f"https://generativelanguage.googleapis.com/v1/models/{self.model}:generateContent"
        self.enabled = bool(self.api_key)

    async def completion(self, prompt: str, max_tokens: int = 512) -> str:
        if not self.enabled:
            raise RuntimeError("Gemini API key not configured. Set GEMINI_API_KEY in .env.")
        span = _TRACER.start_as_current_span("llm.completion") if _TRACER else None
        if span:
            span.__enter__()
            span.set_attribute("llm.model", self.model)
            span.set_attribute("llm.max_tokens", max_tokens)
        try:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens}
        }
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        attempt = 0
        while True:
            attempt += 1
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(self.base_url, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                try:
                    if span:
                        span.set_attribute("http.status_code", resp.status_code)
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                except Exception:
                    if span:
                        span.set_attribute("http.status_code", resp.status_code)
                    return json.dumps(data, indent=2)
            if attempt >= settings.llm_max_retries:
                if span:
                    span.set_attribute("http.status_code", resp.status_code)
                raise RuntimeError(f"Gemini API error: {resp.status_code} {resp.text}")
            backoff = settings.llm_retry_base_s * (2 ** (attempt - 1))
            backoff = backoff + random.uniform(0, 0.25)
            await asyncio.sleep(backoff)
        finally:
            if span:
                span.__exit__(None, None, None)

class LLMClient:
    """Facade: generation (Gemini REST) + local embeddings (SentenceTransformer)."""
    def __init__(self):
        self.gen = GeminiREST()
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
