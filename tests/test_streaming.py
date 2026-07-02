"""Tests for the streaming synthesis path and its server-side provenance.

The streaming LLM boundary (`llm.completion_stream`) is monkeypatched with the
FakeLLM async generator, so no live Ollama is needed.
"""

import api
import synthesis
from conftest import FakeLLM

_EVIDENCE = [
    {
        "id": "a",
        "text": "Fear is often worse than the danger itself, according to Seneca.",
        "source": "seneca.txt",
        "offset_start": 0,
        "offset_end": 60,
    },
    {
        "id": "b",
        "text": "Reason dissolves imagined fears.",
        "source": "seneca_letters.txt",
        "offset_start": 5,
        "offset_end": 40,
    },
]


# --- _pick_evidence / build_stream_provenance -------------------------------


def test_pick_evidence_prefers_trusted_ids():
    picks = synthesis._pick_evidence(_EVIDENCE, {"trusted_ids": ["b"]})
    assert [p["id"] for p in picks] == ["b"]


def test_pick_evidence_falls_back_to_top():
    picks = synthesis._pick_evidence(_EVIDENCE, {})
    assert [p["id"] for p in picks] == ["a", "b"]


def test_build_stream_provenance_filters_incomplete():
    evidence = _EVIDENCE + [{"id": "c", "text": "no offsets", "source": "x.txt"}]
    prov = synthesis.build_stream_provenance(evidence, {"trusted_ids": ["a", "c"]})
    # "c" lacks offsets → dropped; "a" kept.
    assert [p["id"] for p in prov] == ["a"]
    assert prov[0]["source"] == "seneca.txt"


# --- synthesize_answer_stream -----------------------------------------------


async def test_stream_yields_token_deltas(monkeypatch):
    fake = FakeLLM(stream_chunks=["Fear ", "is ", "imagined."])
    monkeypatch.setattr(synthesis.llm, "completion_stream", fake.completion_stream)
    chunks = [
        c async for c in synthesis.synthesize_answer_stream("q", _EVIDENCE, {"trusted_ids": ["a"]})
    ]
    assert chunks == ["Fear ", "is ", "imagined."]
    assert fake.calls[0]["stream"] is True


async def test_stream_falls_back_when_empty(monkeypatch):
    # No chunks emitted → a formatted fallback from the evidence is yielded.
    fake = FakeLLM(stream_chunks=[])
    monkeypatch.setattr(synthesis.llm, "completion_stream", fake.completion_stream)
    chunks = [
        c async for c in synthesis.synthesize_answer_stream("q", _EVIDENCE, {"trusted_ids": ["a"]})
    ]
    assert chunks  # non-empty fallback
    assert "fear" in " ".join(chunks).lower()


async def test_stream_falls_back_on_error(monkeypatch):
    fake = FakeLLM(raises=RuntimeError("ollama down"))
    monkeypatch.setattr(synthesis.llm, "completion_stream", fake.completion_stream)
    chunks = [
        c async for c in synthesis.synthesize_answer_stream("q", _EVIDENCE, {"trusted_ids": ["a"]})
    ]
    # Error is swallowed; a fallback answer is still produced.
    assert chunks


# --- SSE formatter ----------------------------------------------------------


def test_sse_format():
    msg = api._sse("token", {"text": "hi"})
    assert msg == 'event: token\ndata: {"text": "hi"}\n\n'
