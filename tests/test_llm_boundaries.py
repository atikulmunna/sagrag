"""Async tests exercising the LLM fallback ladders in the pipeline stages.

The shared `llm.completion` coroutine is monkeypatched per test so no live
Ollama is required; each test drives a specific success/non-JSON/error branch.
"""

import asyncio

import judge
import speculative
import synthesis
from conftest import FakeLLM


# --- speculative.plan_query -------------------------------------------------


async def test_plan_query_success(monkeypatch):
    fake = FakeLLM(
        responses=['{"intent": "lookup", "hypotheses": [], "queries": ["q1"], "constraints": {}}']
    )
    monkeypatch.setattr(speculative.llm, "completion", fake.completion)
    plan = await speculative.plan_query("some query")
    assert plan["intent"] == "lookup"
    assert plan["queries"] == ["q1"]


async def test_plan_query_falls_back_on_error(monkeypatch):
    fake = FakeLLM(raises=RuntimeError("ollama down"))
    monkeypatch.setattr(speculative.llm, "completion", fake.completion)
    plan = await speculative.plan_query("I feel a lot of fear lately")
    # Rule-based fallback recognizes the emotional-guidance intent.
    assert plan["intent"] == "stoic-guidance"
    assert plan["queries"]


async def test_plan_query_requests_structured_output(monkeypatch):
    fake = FakeLLM(responses=['{"intent": "lookup", "queries": ["q1"]}'])
    monkeypatch.setattr(speculative.llm, "completion", fake.completion)
    await speculative.plan_query("some query")
    assert fake.calls[0]["format"] == "json"
    assert fake.calls[0]["temperature"] == 0


# --- judge.judge_evidence ---------------------------------------------------


async def test_judge_success(monkeypatch):
    fake = FakeLLM(responses=['{"confidence": 0.7, "trusted_ids": ["a"], "notes": "ok"}'])
    monkeypatch.setattr(judge.llm, "completion", fake.completion)
    out = await judge.judge_evidence("q", [{"id": "a", "text": "x"}], [], None, None)
    assert out["confidence"] == 0.7
    assert out["trusted_ids"] == ["a"]


async def test_judge_non_json_fallback(monkeypatch):
    fake = FakeLLM(responses=["this is not json at all"])
    monkeypatch.setattr(judge.llm, "completion", fake.completion)
    out = await judge.judge_evidence(
        "q", [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}], [], None, None
    )
    assert out["confidence"] == 0.4
    assert out["trusted_ids"] == ["a", "b"]
    assert out["notes"] == "judge_non_json"


async def test_judge_error_fallback(monkeypatch):
    fake = FakeLLM(raises=asyncio.TimeoutError())
    monkeypatch.setattr(judge.llm, "completion", fake.completion)
    out = await judge.judge_evidence("q", [{"id": "a", "text": "x"}], [], None, None)
    assert out["confidence"] == 0.3
    assert out["notes"] == "judge_error_fallback"


async def test_judge_requests_structured_output(monkeypatch):
    fake = FakeLLM(responses=['{"confidence": 0.7, "trusted_ids": ["a"], "notes": "ok"}'])
    monkeypatch.setattr(judge.llm, "completion", fake.completion)
    await judge.judge_evidence("q", [{"id": "a", "text": "x"}], [], None, None)
    assert fake.calls[0]["format"] == "json"
    assert fake.calls[0]["temperature"] == 0


# --- synthesis.synthesize_answer --------------------------------------------

_EVIDENCE = [
    {
        "id": "a",
        "text": "Fear is often worse than the danger itself, according to Seneca.",
        "source": "seneca.txt",
        "offset_start": 0,
        "offset_end": 60,
    }
]


async def test_synthesize_success(monkeypatch):
    answer = (
        "Seneca argues that fear is largely imagined and that reason dissolves it, "
        "so we should question our fears calmly."
    )
    fake = FakeLLM(
        responses=[
            '{"answer": "'
            + answer
            + '", "provenance": [{"id": "a"}], "confidence": 0.6, "explain_trace": "ok"}'
        ]
    )
    monkeypatch.setattr(synthesis.llm, "completion", fake.completion)
    out = await synthesis.synthesize_answer(
        "q", _EVIDENCE, {"confidence": 0.6, "trusted_ids": ["a"]}
    )
    assert out["answer"]
    assert out["confidence"] == 0.6
    assert len(out["provenance"]) == 1
    assert out["provenance"][0]["id"] == "a"


async def test_synthesize_requests_structured_output(monkeypatch):
    answer = (
        "Seneca argues that fear is largely imagined and that reason dissolves it, "
        "so we should question our fears calmly."
    )
    fake = FakeLLM(
        responses=[
            '{"answer": "'
            + answer
            + '", "provenance": [{"id": "a"}], "confidence": 0.6, "explain_trace": "ok"}'
        ]
    )
    monkeypatch.setattr(synthesis.llm, "completion", fake.completion)
    await synthesis.synthesize_answer("q", _EVIDENCE, {"confidence": 0.6, "trusted_ids": ["a"]})
    # The main synthesis call requests JSON with deterministic temperature.
    assert fake.calls[0]["format"] == "json"
    assert fake.calls[0]["temperature"] == 0


async def test_naturalizer_stays_free_text(monkeypatch):
    # Non-JSON model output forces the naturalizer path; that call must NOT
    # request JSON format (it produces plain prose).
    fake = FakeLLM(responses=["this is plain prose, not json, about fear and reason and calm."])
    monkeypatch.setattr(synthesis.llm, "completion", fake.completion)
    await synthesis.synthesize_answer("q", _EVIDENCE, {"confidence": 0.3, "trusted_ids": ["a"]})
    # First call is the structured synthesis attempt; a later naturalizer call
    # (if any) is free-text.
    assert fake.calls[0]["format"] == "json"
    naturalizer_calls = [c for c in fake.calls[1:]]
    assert all(c.get("format") is None for c in naturalizer_calls)


async def test_synthesize_timeout_falls_back(monkeypatch):
    fake = FakeLLM(raises=asyncio.TimeoutError())
    monkeypatch.setattr(synthesis.llm, "completion", fake.completion)
    out = await synthesis.synthesize_answer(
        "q", _EVIDENCE, {"confidence": 0.3, "trusted_ids": ["a"]}
    )
    assert out["explain_trace"] == "synthesis_timeout"
    assert out["answer"]  # formatted from evidence
    assert len(out["provenance"]) == 1


async def test_synthesize_non_json(monkeypatch):
    prose = "Fear tends to be worse in anticipation than in reality, and reason helps us see that clearly."
    fake = FakeLLM(responses=[prose])
    monkeypatch.setattr(synthesis.llm, "completion", fake.completion)
    out = await synthesis.synthesize_answer(
        "q", _EVIDENCE, {"confidence": 0.3, "trusted_ids": ["a"]}
    )
    assert out["explain_trace"] == "synthesis_non_json"
    assert out["answer"]
