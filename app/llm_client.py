# app/llm_client.py
import asyncio
import contextlib
import json
import random

import httpx

from config import settings
try:
    from opentelemetry import trace
    _TRACER = trace.get_tracer(__name__)
except Exception:
    _TRACER = None

# Statuses worth retrying; any other non-200 (bad model name, bad request) fails fast.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=settings.llm_connect_timeout_s,
        read=settings.llm_read_timeout_s,
        write=10.0,
        pool=10.0,
    )


class OllamaREST:
    """Minimal Ollama REST wrapper (generate endpoint).

    One pooled httpx client and one concurrency semaphore per event loop. Every
    request has connect/read timeouts, and streamed answers also have an overall
    deadline, so a stalled Ollama can never hold a concurrency slot forever.
    """
    def __init__(self):
        self.base_url = settings.ollama_url.rstrip("/")
        self.model = settings.ollama_model
        self._loop = None
        self._sema = None
        self._client = None

    def _resources(self):
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop = loop
            self._sema = asyncio.Semaphore(settings.llm_max_concurrent)
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=_timeout())
        return self._sema, self._client

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()
        self._client = None
        self._loop = None

    async def completion(
        self,
        prompt: str,
        max_tokens: int = 512,
        format: str | dict | None = None,
        temperature: float | None = None,
    ) -> str:
        span_cm = _TRACER.start_as_current_span("llm.completion") if _TRACER else contextlib.nullcontext()
        with span_cm as span:
            if span:
                span.set_attribute("llm.model", self.model)
                span.set_attribute("llm.max_tokens", max_tokens)
                span.set_attribute("llm.provider", "ollama")
                if format is not None:
                    span.set_attribute("llm.format", "json" if isinstance(format, str) else "schema")
                if temperature is not None:
                    span.set_attribute("llm.temperature", temperature)
            options: dict = {"num_predict": max_tokens}
            if temperature is not None:
                options["temperature"] = temperature
            payload: dict = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": options,
            }
            # Ollama structured-output: "json" forces valid JSON; a dict is a JSON schema.
            if format is not None:
                payload["format"] = format
            sema, client = self._resources()
            attempt = 0
            while True:
                attempt += 1
                try:
                    async with sema:
                        resp = await client.post("/api/generate", json=payload)
                except httpx.TransportError as e:
                    err, retryable = e, True
                else:
                    if resp.status_code == 200:
                        return resp.json().get("response", "")
                    if span:
                        span.set_attribute("http.status_code", resp.status_code)
                    err = RuntimeError(f"Ollama error: {resp.status_code} {resp.text[:200]}")
                    retryable = resp.status_code in _RETRY_STATUS
                if not retryable or attempt >= settings.llm_max_retries:
                    raise err
                backoff = settings.llm_retry_base_s * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
                await asyncio.sleep(backoff)

    async def completion_stream(self, prompt: str, max_tokens: int = 512, temperature: float | None = None):
        """Yield response text deltas from Ollama's streaming generate endpoint.

        Free-text only (no JSON contract): callers stream plain prose. Errors
        (including the overall deadline) propagate to the caller, which is
        expected to handle fallback.
        """
        options: dict = {"num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": options,
        }
        span_cm = _TRACER.start_as_current_span("llm.completion_stream") if _TRACER else contextlib.nullcontext()
        with span_cm as span:
            if span:
                span.set_attribute("llm.model", self.model)
                span.set_attribute("llm.provider", "ollama")
                span.set_attribute("llm.stream", True)
            sema, client = self._resources()
            loop = asyncio.get_running_loop()
            started = loop.time()
            async with sema:
                # The read timeout bounds a stall between chunks; the deadline
                # bounds a slow trickle.
                async with client.stream("POST", "/api/generate", json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if loop.time() - started > settings.synthesis_stream_timeout_s:
                            raise TimeoutError("llm stream deadline exceeded")
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except ValueError:
                            continue
                        chunk = data.get("response")
                        if chunk:
                            yield chunk
                        if data.get("done"):
                            break

    async def ping(self) -> bool:
        _, client = self._resources()
        resp = await client.get("/api/tags", timeout=3.0)
        return resp.status_code == 200


class LLMClient:
    """Facade over the generation backend (Ollama REST). Embeddings live in
    embeddings.py."""
    def __init__(self):
        self.gen = OllamaREST()

    async def completion(
        self,
        prompt: str,
        max_tokens: int = 512,
        format: str | dict | None = None,
        temperature: float | None = None,
    ) -> str:
        return await self.gen.completion(
            prompt, max_tokens=max_tokens, format=format, temperature=temperature
        )

    async def completion_stream(self, prompt: str, max_tokens: int = 512, temperature: float | None = None):
        async for chunk in self.gen.completion_stream(prompt, max_tokens=max_tokens, temperature=temperature):
            yield chunk

    async def ping(self) -> bool:
        return await self.gen.ping()

    async def aclose(self):
        await self.gen.aclose()


# single shared client instance
llm = LLMClient()
