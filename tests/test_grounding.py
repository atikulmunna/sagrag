"""Grounding guarantees: citations come only from retrieved evidence, model
output can't inject response fields, and model scores are sanitized."""

import json

import judge
import speculative
import synthesis
from conftest import FakeLLM

_ANSWER = (
    "Virtue is the only true good for the Stoics, and externals are indifferent. "
    "Seneca develops this idea at length in his letters."
)
_EVIDENCE = [
    {
        "id": "real-1",
        "text": "Virtue is the only good.",
        "source": "real.txt",
        "offset_start": 0,
        "offset_end": 24,
    },
    {
        "id": "real-2",
        "text": "Externals are indifferent.",
        "source": "real.txt",
        "offset_start": 30,
        "offset_end": 56,
    },
]


def _model_reply(**overrides):
    reply = {"answer": _ANSWER, "provenance": [], "confidence": 0.6, "explain_trace": "ok"}
    reply.update(overrides)
    return json.dumps(reply)


async def test_fabricated_citation_is_dropped(monkeypatch):
    # Regression: an invented id with invented source/offsets used to pass.
    fake = FakeLLM(
        responses=[
            _model_reply(
                provenance=[
                    {
                        "id": "made-up",
                        "source": "nonexistent.txt",
                        "offset_start": 1,
                        "offset_end": 2,
                    },
                ]
            )
        ]
    )
    monkeypatch.setattr(synthesis.llm, "completion", fake.completion)
    out = await synthesis.synthesize_answer("q", _EVIDENCE, {"trusted_ids": ["real-1"]})
    assert [p["source"] for p in out["provenance"]] == ["real.txt"]
    assert all(p["id"] in {"real-1", "real-2"} for p in out["provenance"])


async def test_citation_offsets_come_from_evidence_not_model(monkeypatch):
    fake = FakeLLM(
        responses=[
            _model_reply(
                provenance=[
                    {"id": "real-2", "source": "evil.txt", "offset_start": 999, "offset_end": 1000},
                ]
            )
        ]
    )
    monkeypatch.setattr(synthesis.llm, "completion", fake.completion)
    out = await synthesis.synthesize_answer("q", _EVIDENCE, {"trusted_ids": []})
    assert out["provenance"] == [
        {"id": "real-2", "source": "real.txt", "offset_start": 30, "offset_end": 56}
    ]


async def test_model_cannot_inject_response_fields(monkeypatch):
    # Regression: the parsed model JSON was spread into the API response, so
    # extra keys could overwrite fields such as domain or retrieval_failures.
    fake = FakeLLM(
        responses=[_model_reply(domain="hijacked", retrieval_failures=[], user_id="admin")]
    )
    monkeypatch.setattr(synthesis.llm, "completion", fake.completion)
    out = await synthesis.synthesize_answer("q", _EVIDENCE, {"trusted_ids": ["real-1"]})
    assert set(out) == {"answer", "provenance", "confidence", "explain_trace"}


async def test_model_confidence_is_coerced(monkeypatch):
    fake = FakeLLM(responses=[_model_reply(confidence="0.7")])
    monkeypatch.setattr(synthesis.llm, "completion", fake.completion)
    out = await synthesis.synthesize_answer("q", _EVIDENCE, {"confidence": 0.4, "trusted_ids": []})
    assert out["confidence"] == 0.7

    fake = FakeLLM(responses=[_model_reply(confidence="very high")])
    monkeypatch.setattr(synthesis.llm, "completion", fake.completion)
    out = await synthesis.synthesize_answer("q", _EVIDENCE, {"confidence": 0.4, "trusted_ids": []})
    assert out["confidence"] == 0.4  # falls back to the judge's score


def test_validate_provenance_dedupes_and_ignores_garbage():
    prov = [{"id": "real-1"}, {"id": "real-1"}, "junk", {"chunk_id": "real-2"}, {"id": None}]
    out = synthesis.validate_provenance(prov, _EVIDENCE)
    assert [p["id"] for p in out] == ["real-1", "real-2"]
    assert synthesis.validate_provenance("[{'id': 'real-1'}]", _EVIDENCE) == []


async def test_judge_filters_invented_trusted_ids(monkeypatch):
    fake = FakeLLM(
        responses=['{"confidence": "0.9", "trusted_ids": ["real-1", "ghost"], "notes": "ok"}']
    )
    monkeypatch.setattr(judge.llm, "completion", fake.completion)
    out = await judge.judge_evidence("q", _EVIDENCE, [], None, None)
    assert out["trusted_ids"] == ["real-1"]
    assert out["confidence"] == 0.9


async def test_judge_clamps_out_of_range_confidence(monkeypatch):
    fake = FakeLLM(responses=['{"confidence": 7, "trusted_ids": [], "notes": "ok"}'])
    monkeypatch.setattr(judge.llm, "completion", fake.completion)
    out = await judge.judge_evidence("q", _EVIDENCE, [], None, None)
    assert out["confidence"] == 1.0


# --- planner output is untrusted ---------------------------------------------


def test_sanitize_plan_drops_domain_and_policy_constraints():
    # Regression: the planner's constraints could set the search domain (and
    # policy lists), which made the query text able to choose a namespace.
    plan = speculative.sanitize_plan(
        {
            "queries": ["q1"],
            "constraints": {"domain": "acme", "allowlist": ["x"], "freshness_days": 30},
        },
        "user question",
    )
    assert plan["constraints"] == {"freshness_days": 30}


def test_sanitize_plan_bounds_freshness():
    for bad in (0, -5, "soon", 10**9, None):
        plan = speculative.sanitize_plan({"constraints": {"freshness_days": bad}}, "q")
        assert plan["constraints"] == {}


def test_sanitize_plan_keeps_original_first_dedupes_and_caps(monkeypatch):
    monkeypatch.setattr(speculative.settings, "planner_max_queries", 3)
    plan = speculative.sanitize_plan(
        {"queries": ["User Question", "a", "a", "b", "c", "d"], "intent": "x" * 200},
        "user question",
    )
    assert plan["queries"] == ["user question", "a", "b"]
    assert len(plan["intent"]) == 64


def test_sanitize_plan_tolerates_garbage():
    plan = speculative.sanitize_plan(
        {"queries": "single", "hypotheses": "nope", "intent": {"x": 1}}, "q"
    )
    assert plan == {
        "intent": "general",
        "hypotheses": [],
        "queries": ["q", "single"],
        "constraints": {},
    }
    assert speculative.sanitize_plan(None, "q")["queries"] == ["q"]


# --- short evidence references --------------------------------------------------


def test_refs_resolve_labels_ids_and_drop_unknowns():
    import refs

    ev = [{"id": "uuid-a"}, {"id": "uuid-b"}, {"id": "uuid-c"}]
    labelled = refs.label(ev, 2)
    assert refs.resolve(["E2", "[e1]", "E9", "uuid-c", "made-up", 2], labelled, ev) == [
        "uuid-b",
        "uuid-a",
        "uuid-c",
    ]
    assert refs.resolve("E1", labelled, ev) == []


async def test_judge_maps_labels_back_to_chunk_ids(monkeypatch):
    # Regression: echoing 36-char UUIDs overflowed JUDGE_MAX_TOKENS, the JSON
    # was cut off, and the judge's real confidence was replaced by a fallback.
    fake = FakeLLM(responses=['{"confidence": 0.8, "trusted": ["E2", "E7"], "notes": "ok"}'])
    monkeypatch.setattr(judge.llm, "completion", fake.completion)
    out = await judge.judge_evidence("q", _EVIDENCE, [], None, None)
    assert out["trusted_ids"] == ["real-2"] and out["confidence"] == 0.8
    prompt = fake.calls[0]["prompt"]
    assert "[E1] (real.txt)" in prompt and "real-1" not in prompt


async def test_synthesis_cites_by_label(monkeypatch):
    fake = FakeLLM(responses=[json.dumps({"answer": _ANSWER, "cited": ["E2"], "confidence": 0.7})])
    monkeypatch.setattr(synthesis.llm, "completion", fake.completion)
    out = await synthesis.synthesize_answer(
        "q", _EVIDENCE, {"confidence": 0.5, "trusted_ids": ["real-2"]}
    )
    assert [p["id"] for p in out["provenance"]] == ["real-2"]
    assert out["explain_trace"] == "synthesis_json"
    assert "Judge assessment:\nconfidence 0.5; trusted passages: E2" in fake.calls[0]["prompt"]


async def test_truncated_json_answer_is_salvaged_without_another_call(monkeypatch):
    truncated = '{"answer": "Virtue is the only good for the Stoics, and externals are indifferent. Seneca adds that'
    fake = FakeLLM(responses=[truncated, "a naturalizer reply that must not be needed"])
    monkeypatch.setattr(synthesis.llm, "completion", fake.completion)
    out = await synthesis.synthesize_answer("q", _EVIDENCE, {"confidence": 0.5, "trusted_ids": []})
    assert out["answer"] == "Virtue is the only good for the Stoics, and externals are indifferent."
    assert out["explain_trace"] == "synthesis_json_truncated"
    assert len(fake.calls) == 1
