"""Unit tests for the pure confidence-adjustment helpers in judge.py."""

import judge


def test_extract_contradictions_filters_zero():
    gr = {"claims": [{"id": "c1", "contradict_count": 2}, {"id": "c2", "contradict_count": 0}]}
    assert judge.extract_contradictions(gr) == [{"id": "c1", "contradict_count": 2}]
    assert judge.extract_contradictions(None) == []


def test_apply_contradiction_penalty_caps(monkeypatch):
    monkeypatch.setattr(judge.settings, "judge_contradiction_confidence_cap", 0.6)
    monkeypatch.setattr(judge.settings, "judge_contradiction_penalty_per_claim", 0.1)
    monkeypatch.setattr(judge.settings, "judge_contradiction_penalty_max", 0.5)
    out = judge.apply_contradiction_penalty({"confidence": 0.9}, 2)
    # 0.9 * (1 - 0.2) = 0.72, then capped at 0.6.
    assert out["confidence"] == 0.6
    assert "contradiction" in out["notes"]


def test_apply_relation_boost(monkeypatch):
    monkeypatch.setattr(judge.settings, "judge_relation_boost_per_relation", 0.05)
    monkeypatch.setattr(judge.settings, "judge_relation_boost_max", 0.2)
    gr = {"relation_strength": [{"relation": "a"}, {"relation": "b"}]}
    out = judge.apply_relation_boost({"confidence": 0.5}, gr, contradiction_count=0)
    assert abs(out["confidence"] - 0.6) < 1e-9  # +min(0.2, 0.1)
    # Contradictions present -> no boost applied.
    out2 = judge.apply_relation_boost({"confidence": 0.5}, gr, contradiction_count=1)
    assert out2["confidence"] == 0.5


def test_apply_relation_conflict_penalty(monkeypatch):
    monkeypatch.setattr(judge.settings, "judge_relation_conflict_penalty", 0.15)
    gr = {"relation_conflicts": [{"pair": "a|b"}]}
    out = judge.apply_relation_conflict_penalty({"confidence": 0.8}, gr)
    assert abs(out["confidence"] - 0.68) < 1e-9  # 0.8 * (1 - 0.15)
    # No conflicts -> unchanged.
    assert judge.apply_relation_conflict_penalty({"confidence": 0.8}, {})["confidence"] == 0.8
