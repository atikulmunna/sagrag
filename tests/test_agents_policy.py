"""Unit tests for pure routing/policy/parsing helpers in agents.py."""

import agents


def test_point_field_dict_and_object():
    assert agents._point_field({"id": 5}, "id") == 5

    class P:
        score = 0.9

    assert agents._point_field(P(), "score") == 0.9
    assert agents._point_field({}, "missing", default="d") == "d"


def test_qdrant_points_variants():
    assert agents._qdrant_points(None) == []
    assert agents._qdrant_points([1, 2]) == [1, 2]
    assert agents._qdrant_points({"result": [{"id": 1}]}) == [{"id": 1}]
    assert agents._qdrant_points({"result": {"points": [{"id": 2}]}}) == [{"id": 2}]


def test_route_domain_prefers_constraint(monkeypatch):
    monkeypatch.setattr(agents, "_list_domain_indices", lambda: [])
    assert agents.route_domain("anything", constraints={"domain": "Finance"}) == "finance"


def test_route_domain_keyword_scoring(monkeypatch):
    monkeypatch.setattr(agents, "_list_domain_indices", lambda: [])
    monkeypatch.setattr(agents.settings, "domain_keywords", {"stoicism": ["seneca", "stoic"]})
    monkeypatch.setattr(agents.settings, "domain_aliases", {})
    monkeypatch.setattr(agents.settings, "domain_min_keyword_hits", 2)
    # Two keyword hits meets the threshold.
    assert agents.route_domain("what did seneca the stoic say") == "stoicism"
    # A single hit falls below the threshold -> no domain.
    assert agents.route_domain("what did seneca say") is None


def test_apply_policy_filter_blocklist():
    docs = [{"text": "clean text"}, {"text": "contains SPOILER here"}]
    out = agents.apply_policy_filter(docs, blocklist=["spoiler"])
    assert out == [{"text": "clean text"}]


def test_match_rule_domain_and_contains():
    rule = {"domains": ["stoicism"], "contains": ["fear"]}
    assert agents._match_rule(rule, "a note about fear", "txt", "stoicism") is True
    assert agents._match_rule(rule, "a note about fear", "txt", "finance") is False
    assert agents._match_rule(rule, "no keyword", "txt", "stoicism") is False


def test_apply_policy_rules_deny():
    results = [
        {"text": "spoiler alert", "domain": "stoicism", "source_type": "txt"},
        {"text": "safe content", "domain": "stoicism", "source_type": "txt"},
    ]
    rules = [{"action": "deny", "domains": ["stoicism"], "contains": ["spoiler"]}]
    out = agents.apply_policy_rules(results, rules=rules)
    assert len(out) == 1 and out[0]["text"] == "safe content"


def test_apply_freshness_filter():
    import time

    now = time.time()
    results = [
        {"text": "fresh", "timestamp": now},
        {"text": "stale", "timestamp": now - 10 * 86400},
    ]
    out = agents.apply_freshness_filter(results, freshness_days=5)
    assert [r["text"] for r in out] == ["fresh"]
    # No filter when freshness_days is None.
    assert agents.apply_freshness_filter(results, freshness_days=None) == results


def test_extract_structured_lines_keeps_delimited():
    text = "plain prose line\nName: Seneca\nrandom words only\nrevenue, 100, usd"
    lines = agents._extract_structured_lines(text, tokens=[], max_lines=10)
    assert "Name: Seneca" in lines
    assert "revenue, 100, usd" in lines
    assert "plain prose line" not in lines
