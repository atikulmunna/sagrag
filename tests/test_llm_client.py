"""OllamaREST transport behavior, driven through httpx.MockTransport."""

import asyncio
import json

import httpx
import pytest

import llm_client


def _install(monkeypatch, handler):
    client = llm_client.OllamaREST()
    monkeypatch.setattr(llm_client.settings, "llm_retry_base_s", 0.0)
    monkeypatch.setattr(llm_client.settings, "llm_max_retries", 3)
    client._loop = asyncio.get_running_loop()
    client._sema = asyncio.Semaphore(2)
    client._client = httpx.AsyncClient(
        base_url="http://ollama", transport=httpx.MockTransport(handler)
    )
    return client


async def test_retries_transport_errors_then_succeeds(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"response": "hello"})

    client = _install(monkeypatch, handler)
    assert await client.completion("p") == "hello"
    assert len(calls) == 3


async def test_does_not_retry_client_errors(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(404, text="model not found")

    client = _install(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="404"):
        await client.completion("p")
    assert len(calls) == 1


async def test_retries_server_errors_up_to_the_limit(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, text="busy")

    client = _install(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="503"):
        await client.completion("p")
    assert len(calls) == 3


async def test_request_payload(monkeypatch):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"response": "{}"})

    client = _install(monkeypatch, handler)
    await client.completion("prompt", max_tokens=7, format="json", temperature=0)
    assert seen["format"] == "json" and seen["options"] == {"num_predict": 7, "temperature": 0}


def _stream_body(*chunks):
    lines = [json.dumps({"response": c, "done": False}) for c in chunks]
    lines.append(json.dumps({"response": "", "done": True}))
    return "\n".join(lines).encode()


async def test_stream_yields_deltas(monkeypatch):
    client = _install(
        monkeypatch, lambda request: httpx.Response(200, content=_stream_body("Hel", "lo"))
    )
    monkeypatch.setattr(llm_client.settings, "synthesis_stream_timeout_s", 60)
    assert [c async for c in client.completion_stream("p")] == ["Hel", "lo"]


async def test_stream_deadline_is_enforced(monkeypatch):
    # Regression: the stream had no timeout at all, so a stalled model held an
    # LLM concurrency slot forever.
    client = _install(
        monkeypatch, lambda request: httpx.Response(200, content=_stream_body("a", "b"))
    )
    monkeypatch.setattr(llm_client.settings, "synthesis_stream_timeout_s", -1)
    with pytest.raises(TimeoutError):
        async for _ in client.completion_stream("p"):
            pass
    # The semaphore slot was released.
    assert not client._sema.locked()
