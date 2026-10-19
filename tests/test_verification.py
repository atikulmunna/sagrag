"""Claim-level verification (NLI), abstention, and NLI-confirmed graph
contradictions. The NLI model is faked at the `nli_probabilities` boundary."""

import types

import graph
import metrics
import pipeline
import verification

_EVIDENCE = [
    {
        "id": "e1",
        "text": "Some things are in our control and others not.",
        "source": "ench.txt",
        "offset_start": 0,
        "offset_end": 46,
    },
    {
        "id": "e2",
        "text": "If thou art pained by any external thing, it is thy judgement.",
        "source": "med.txt",
        "offset_start": 0,
        "offset_end": 61,
    },
]
_PROV = [{"id": "e1"}, {"id": "e2"}]


def test_split_sentences():
    text = "Epictetus divides the world. Our opinions are ours! Is rain up to us? No. That is the whole point."
    assert verification.split_sentences(text) == [
        "Epictetus divides the world.",
        "Our opinions are ours!",
        "Is rain up to us? No.",  # fragments under 12 characters join the previous sentence
        "That is the whole point.",
    ]
    assert verification.split_sentences("  ") == []


def test_label_index_reads_model_config():
    model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            config=types.SimpleNamespace(
                id2label={0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
            )
        )
    )
    assert verification._label_index(model) == {"entailment": 0, "neutral": 1, "contradiction": 2}
    assert verification._label_index(object())["entailment"] == 1  # cross-encoder/nli default


def test_classify_statuses():
    sentences = [
        "Opinion is in our control.",
        "Rain helps travellers.",
        "Property is in our control.",
    ]
    passages = [
        ("e1", "Opinion and desire are in our control."),
        ("e2", "Property is not in our control."),
    ]
    probs = [
        {"entailment": 0.9, "contradiction": 0.0},
        {"entailment": 0.2, "contradiction": 0.1},
        {"entailment": 0.1, "contradiction": 0.9},
        {"entailment": 0.1, "contradiction": 0.9},
        {"entailment": 0.1, "contradiction": 0.3},
        {"entailment": 0.2, "contradiction": 0.8},
    ]
    claims = verification.classify(sentences, passages, probs, threshold=0.5)
    # The unrelated sentence's high contradiction scores are ignored (no shared content words).
    assert [c["status"] for c in claims] == ["supported", "unsupported", "contradicted"]
    assert claims[0]["support_id"] == "e1"
    assert claims[0]["support_text"] == "Opinion and desire are in our control."
    assert claims[1]["support_id"] is None and claims[1]["support_text"] is None


def test_strip_attribution():
    assert (
        verification.strip_attribution("Epictetus says our opinions are ours.")
        == "Our opinions are ours."
    )
    assert verification.strip_attribution("He argues that fear is imagined.") == "Fear is imagined."
    assert (
        verification.strip_attribution("According to Marcus Aurelius, judgement disturbs us.")
        == "Judgement disturbs us."
    )
    assert (
        verification.strip_attribution("The Stoic Seneca teaches that time is short.")
        == "Time is short."
    )
    # Nothing to strip, or stripping would leave too little.
    assert verification.strip_attribution("Virtue is the only good.") == "Virtue is the only good."
    assert verification.strip_attribution("He says no.") == "He says no."


def test_premise_units_are_sentences_and_pairs():
    units = verification.premise_units(
        "First sentence here. Second sentence here; third clause here."
    )
    assert units == [
        "First sentence here.",
        "Second sentence here;",
        "third clause here.",
        "First sentence here. Second sentence here;",
        "Second sentence here; third clause here.",
    ]
    assert verification.premise_units("") == []


def test_related_requires_shared_content_words():
    assert verification.related(
        "Our body is in our control.", "Things not in our control are body, property."
    )
    assert not verification.related(
        "Seneca wrote letters to Lucilius.", "We suffer more in imagination."
    )
    assert verification.related("Things disturb us.", "It is not this thing that disturbs thee.")


def test_verify_sync_checks_every_sentence_against_cited_passages(monkeypatch):
    seen = []

    def fake_nli(pairs):
        seen.extend(pairs)
        return [
            {
                "entailment": 0.95 if "control" in prem and "control" in hyp else 0.05,
                "contradiction": 0.0,
            }
            for prem, hyp in pairs
        ]

    monkeypatch.setattr(verification, "nli_probabilities", fake_nli)
    out = verification.verify_sync(
        "Some things are in our control. Rain is always good.", _EVIDENCE, _PROV
    )
    assert [c["status"] for c in out["claims"]] == ["supported", "unsupported"]
    assert out["supported_ratio"] == 0.5
    # Only premise units sharing a content word with a sentence reach the model:
    # both passages for the first sentence, none for "Rain is always good."
    assert len(seen) == 2
    assert all(hyp == "Some things are in our control." for _, hyp in seen)


def test_verify_sync_without_cited_passages_marks_everything_unsupported():
    out = verification.verify_sync("One sentence here. Another sentence here.", _EVIDENCE, [])
    assert out["supported_ratio"] == 0.0
    assert {c["status"] for c in out["claims"]} == {"unsupported"}


async def test_verify_is_off_by_default_and_fails_soft(monkeypatch):
    assert await verification.verify("Text.", _EVIDENCE, _PROV) is None
    monkeypatch.setattr(verification.settings, "verify_enabled", True)

    def boom(*args):
        raise OSError("model download failed")

    monkeypatch.setattr(verification, "verify_sync", boom)
    assert await verification.verify("Text.", _EVIDENCE, _PROV) is None


def _ctx(verification_result):
    ctx = pipeline.QueryContext(query="q", prefs={})
    ctx.domain = "stoicism"
    ctx.reranked = [dict(e, rerank_score=3.0) for e in _EVIDENCE]
    ctx.verification = verification_result
    return ctx


_VERIFIED = {
    "claims": [
        {"text": "A.", "status": "supported", "support_id": "e1"},
        {"text": "B.", "status": "unsupported", "support_id": None},
        {"text": "C.", "status": "contradicted", "support_id": None},
    ],
    "supported_ratio": 1 / 3,
}
_SYNTH = {
    "answer": "A. B. C.",
    "provenance": [{"id": "e1", "source": "ench.txt", "offset_start": 0, "offset_end": 46}],
    "confidence": 0.9,
    "explain_trace": "ok",
}


def test_finalize_uses_measured_risk_when_verified(reset_metrics):
    out = pipeline.finalize(_ctx(_VERIFIED), dict(_SYNTH), user_id="u", query_id="q")
    assert out["hallucination_risk_method"] == "nli"
    assert abs(out["hallucination_risk"] - 2 / 3) < 1e-9
    assert out["claims"] == _VERIFIED["claims"] and out["abstained"] is False
    assert {"unsupported_claims", "contradicted_claims"} <= set(out["retrieval_failures"])
    text = metrics.render_prometheus()
    assert 'sag_rag_claims_total{status="contradicted"} 1' in text


def test_finalize_heuristic_without_verification(reset_metrics):
    out = pipeline.finalize(_ctx(None), dict(_SYNTH), user_id="u", query_id="q")
    assert out["hallucination_risk_method"] == "heuristic"
    assert out["claims"] is None and out["supported_ratio"] is None


def test_finalize_abstains_below_support(reset_metrics, monkeypatch):
    monkeypatch.setattr(pipeline.settings, "abstain_below_support", 0.5)
    out = pipeline.finalize(_ctx(_VERIFIED), dict(_SYNTH), user_id="u", query_id="q")
    assert out["abstained"] is True
    assert out["answer"] == pipeline.ABSTAIN_ANSWER
    assert out["unverified_answer"] == "A. B. C."
    assert "abstained" in out["retrieval_failures"]


def test_graph_contradictions_confirmed_by_nli():
    claim_meta = [("new", "Seneca says death is natural and should not be feared by the wise")]
    recent = [
        {"id": "old1", "text": "Seneca says death is natural and should be feared by the wise"},
        {"id": "old2", "text": "Seneca says death is natural and the wise accept it calmly"},
    ]

    def fake_nli(pairs):
        # Contradiction only between "should be feared" and "should not be feared".
        def contra(a, b):
            return ("should be feared" in a and "not be feared" in b) or (
                "should be feared" in b and "not be feared" in a
            )

        return [
            {"entailment": 0.1, "contradiction": 0.9 if contra(a, b) else 0.05} for a, b in pairs
        ]

    assert graph.find_contradictions(claim_meta, recent, ["Seneca"], nli=fake_nli) == [
        ("new", "old1")
    ]

    def broken(pairs):
        raise RuntimeError("no model")

    # Falls back to the negation heuristic.
    assert graph.find_contradictions(claim_meta, recent, ["Seneca"], nli=broken) == [
        ("new", "old1"),
        ("new", "old2"),
    ]


def test_rhetorical_questions_cannot_contradict():
    question = "For what is baser than to chase after pleasure in the very midst of mourning?"
    claim = "Seneca advises against pursuing pleasure during mourning."
    assert verification.is_question(question)
    assert verification.is_question("Is grief natural?\u201d")
    assert not verification.is_question("Grief is natural.")
    assert verification.related(claim, question)
    assert not verification.can_contradict(claim, question)
    probs = [{"entailment": 0.05, "contradiction": 0.95}]
    out = verification.classify_one(claim, [("p1", question)], probs, threshold=0.5)
    assert out["status"] == "unsupported"


def test_source_author_uses_configured_authors_only():
    assert verification.source_author("seneca_moral_letters.txt") == "Seneca"
    assert verification.source_author("marcus_meditations.txt") == "Marcus"
    assert verification.source_author("cleanthes_hymn_to_zeus.txt") is None
    assert verification.source_author("quarterly_report.csv") is None
    assert verification.source_author(None) is None


def test_premises_carry_the_source_author(monkeypatch):
    seen = []

    def fake_nli(pairs):
        seen.extend(pairs)
        return [{"entailment": 0.9, "contradiction": 0.0} for _ in pairs]

    monkeypatch.setattr(verification, "nli_probabilities", fake_nli)
    evidence = [
        {
            "id": "s1",
            "text": "We suffer more often in imagination than in reality.",
            "source": "seneca_moral_letters.txt",
        }
    ]
    out = verification.verify_sync(
        "Seneca says we suffer more in imagination.", evidence, [{"id": "s1"}]
    )
    assert seen and all(premise.startswith("Seneca writes: ") for premise, _ in seen)
    # The UI highlight still points at the raw passage text, without the prefix.
    assert (
        out["claims"][0]["support_text"] == "We suffer more often in imagination than in reality."
    )
