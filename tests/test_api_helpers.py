"""Unit tests for pure request-path helpers in api.py."""

import api


def test_parse_blocklist_variants():
    assert api._parse_blocklist("a, B ,c") == ["a", "b", "c"]
    assert api._parse_blocklist(["X", " y ", ""]) == ["x", "y"]
    assert api._parse_blocklist(None) == []


def test_query_cache_key_stable_and_order_independent():
    # Same inputs (prefs dict order aside) → same key.
    k1 = api._query_cache_key("Fear and how to handle it", {"a": 1, "b": 2}, "t1")
    k2 = api._query_cache_key("  fear   AND how to Handle it ", {"b": 2, "a": 1}, "t1")
    assert k1 == k2
    assert k1.startswith("sagrag:query:")


def test_query_cache_key_varies_with_inputs():
    base = api._query_cache_key("q", {}, None)
    assert base != api._query_cache_key("different", {}, None)
    assert base != api._query_cache_key("q", {"freshness_days": 7}, None)
    assert base != api._query_cache_key("q", {}, "tenant-a")


def test_policy_signature_changes_with_policy(monkeypatch):
    before = api._policy_signature()
    monkeypatch.setattr(api.settings, "policy_blocklist", "spam.txt")
    assert api._policy_signature() != before


def test_merged_term_synonyms_merges_config(monkeypatch):
    monkeypatch.setattr(api.settings, "query_term_synonyms", {"grief": ["sorrow"]})
    merged = api._merged_term_synonyms()
    assert "fear" in merged  # built-in retained
    assert merged["grief"] == ["sorrow"]  # config merged in


def test_extract_author_mentions(monkeypatch):
    monkeypatch.setattr(api.settings, "domain_keywords", {"stoicism": ["seneca"]})
    terms = api._extract_author_mentions("What does Seneca say about fear?")
    assert "seneca" in terms
    # "what" is a stopword and must not be treated as an author.
    assert "what" not in terms


def test_extract_query_terms_drops_authors_and_expands(monkeypatch):
    monkeypatch.setattr(api.settings, "query_term_synonyms", {})
    terms = api._extract_query_terms("What does Seneca say about fear", ["seneca"])
    assert "seneca" not in terms  # author term removed
    assert "fear" in terms
    assert "anxiety" in terms  # expanded via built-in synonyms


def test_apply_author_bias(monkeypatch):
    monkeypatch.setattr(api.settings, "domain_keywords", {"stoicism": ["seneca"]})
    results = [
        {"source": "seneca_letters.txt", "text": "on the shortness of life"},
        {"source": "other.txt", "text": "seneca once wrote about time"},
        {"source": "other.txt", "text": "unrelated content"},
    ]
    biased, terms = api._apply_author_bias(results, "What did Seneca say?", bias=0.8)
    assert terms == ["seneca"]
    assert biased[0]["author_bias"] == 0.8  # filename hit -> full bias
    assert biased[1]["author_bias"] == 0.4  # text-only hit -> half bias
    assert biased[2]["author_bias"] == 0.0  # no hit


def test_author_matches_by_source_and_text():
    assert api._author_matches({"source": "seneca.txt", "text": ""}, ["seneca"]) is True
    assert api._author_matches({"source": "x.txt", "text": "seneca wrote"}, ["seneca"]) is True
    assert api._author_matches({"source": "x.txt", "text": "nothing"}, ["seneca"]) is False
    assert api._author_matches({"source": "seneca.txt"}, []) is False
