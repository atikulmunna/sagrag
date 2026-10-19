"""Pure helpers in pipeline.py (cache keys, failure tags, finalize) and the
author helpers in authors.py."""

import authors
import pipeline


def test_query_cache_key_stable_and_order_independent():
    # Same inputs (prefs dict order aside) -> same key.
    k1 = pipeline.query_cache_key("Fear and how to handle it", {"a": 1, "b": 2}, "t1")
    k2 = pipeline.query_cache_key("  fear   AND how to Handle it ", {"b": 2, "a": 1}, "t1")
    assert k1 == k2
    assert k1.startswith(pipeline.CACHE_PREFIX)


def test_query_cache_key_varies_with_inputs():
    base = pipeline.query_cache_key("q", {}, None)
    assert base != pipeline.query_cache_key("different", {}, None)
    assert base != pipeline.query_cache_key("q", {"freshness_days": 7}, None)
    assert base != pipeline.query_cache_key("q", {}, "tenant-a")


def test_policy_signature_changes_with_policy(monkeypatch):
    before = pipeline.policy_signature()
    monkeypatch.setattr(pipeline.settings, "policy_blocklist", "spam.txt")
    assert pipeline.policy_signature() != before


def test_parse_list_variants():
    assert pipeline._parse_list("a, B ,c") == ["a", "b", "c"]
    assert pipeline._parse_list(["X", " y ", ""]) == ["x", "y"]
    assert pipeline._parse_list(None) == []


def test_normalize_hit_keeps_zero_offsets():
    # Regression: `payload.get(...) or source.get(...)` turned offset 0 (the
    # first chunk of every file) into None, dropping its provenance.
    hit = {
        "id": "a",
        "score": 1.0,
        "agent": "vector",
        "payload": {"text": "t", "source": "s.txt", "offset_start": 0, "offset_end": 5},
    }
    item = pipeline._normalize_hit(hit)
    assert (item["offset_start"], item["offset_end"]) == (0, 5)


def test_validated_preferences_rejects_unknown_domain(monkeypatch):
    monkeypatch.setattr(pipeline.agents, "known_domains", lambda tenant=None: {"stoicism"})
    assert pipeline.validated_preferences({"domain": "stoicism", "blocklist": []}, None) == {
        "domain": "stoicism"
    }
    for bad in ("finance", "t_acme", "../x"):
        try:
            pipeline.validated_preferences({"domain": bad}, None)
        except pipeline.UnknownDomain:
            continue
        raise AssertionError(f"{bad!r} should be rejected")


def _ctx(**kw):
    ctx = pipeline.QueryContext(query="q", prefs={})
    for k, v in kw.items():
        setattr(ctx, k, v)
    return ctx


def test_retrieval_failure_tags():
    ctx = _ctx(
        domain=None,
        raw_results=[{"id": 1}],
        normalized=[],
        agent_diagnostics={
            "vector": {"attempted": 1, "ok": 0, "timeout": 1},
            "lexical": {"attempted": 1, "ok": 1},
            "structured": {"attempted": 1, "ok": 0, "zero_hits": 1},
        },
    )
    tags = pipeline.retrieval_failures(ctx, [], "synthesis_timeout")
    assert tags == [
        "no_domain",
        "policy_blocked",
        "vector_timeout",
        "structured_zero_hits",
        "no_results",
        "synthesis_timeout",
        "low_result_count",
    ]


def test_finalize_shape_and_risk(reset_metrics):
    results = [
        {
            "id": "a",
            "text": "t",
            "source": "s.txt",
            "offset_start": 0,
            "offset_end": 1,
            "rerank_score": 2.0,
        }
    ]
    ctx = _ctx(
        domain="stoicism", reranked=results, plan={"intent": "lookup"}, timings_ms={"plan": 5}
    )
    synthesis = {
        "answer": "ans",
        "provenance": [{"id": "a", "source": "s.txt", "offset_start": 0, "offset_end": 1}],
        "confidence": 0.8,
        "explain_trace": "ok",
    }
    out = pipeline.finalize(ctx, synthesis, user_id="u", query_id="qid")
    assert (
        out["query_id"] == "qid" and out["intent"] == "lookup" and out["timings_ms"] == {"plan": 5}
    )
    assert abs(out["hallucination_risk"] - 0.2) < 1e-9
    assert out["evidence_coverage"] == 1.0
    assert "low_result_count" in out["retrieval_failures"]


def test_finalize_clamps_confidence(reset_metrics):
    out = pipeline.finalize(
        _ctx(), {"answer": "", "provenance": [], "confidence": 5}, user_id="u", query_id="q"
    )
    assert out["confidence"] == 1.0 and out["hallucination_risk"] == 0.2


# --- authors ---------------------------------------------------------------


def test_merged_term_synonyms_merges_config(monkeypatch):
    monkeypatch.setattr(authors.settings, "query_term_synonyms", {"grief": ["sorrow"]})
    merged = authors.merged_term_synonyms()
    assert "fear" in merged  # built-in retained
    assert merged["grief"] == ["sorrow"]  # config merged in


def test_extract_author_mentions(monkeypatch):
    monkeypatch.setattr(authors.settings, "domain_keywords", {"stoicism": ["seneca"]})
    terms = authors.extract_author_mentions("What does Seneca say about fear?")
    assert "seneca" in terms
    # "what" is a stopword and must not be treated as an author.
    assert "what" not in terms


def test_extract_query_terms_drops_authors_and_expands(monkeypatch):
    monkeypatch.setattr(authors.settings, "query_term_synonyms", {})
    terms = authors.extract_query_terms("What does Seneca say about fear", ["seneca"])
    assert "seneca" not in terms  # author term removed
    assert "fear" in terms
    assert "anxiety" in terms  # expanded via built-in synonyms


def test_apply_author_bias(monkeypatch):
    monkeypatch.setattr(authors.settings, "domain_keywords", {"stoicism": ["seneca"]})
    results = [
        {"source": "seneca_letters.txt", "text": "on the shortness of life"},
        {"source": "other.txt", "text": "seneca once wrote about time"},
        {"source": "other.txt", "text": "unrelated content"},
    ]
    biased, terms = authors.apply_author_bias(results, "What did Seneca say?", bias=0.8)
    assert terms == ["seneca"]
    assert biased[0]["author_bias"] == 0.8  # filename hit -> full bias
    assert biased[1]["author_bias"] == 0.4  # text-only hit -> half bias
    assert biased[2]["author_bias"] == 0.0  # no hit


def test_author_matches_by_source_and_text(tmp_store):
    assert authors.author_matches({"source": "seneca.txt", "text": ""}, ["seneca"]) is True
    assert authors.author_matches({"source": "x.txt", "text": "seneca wrote"}, ["seneca"]) is True
    assert authors.author_matches({"source": "x.txt", "text": "nothing"}, ["seneca"]) is False
    assert authors.author_matches({"source": "seneca.txt"}, []) is False


def test_author_matches_index_is_case_insensitive(tmp_store, monkeypatch):
    (tmp_store / "author_index.json").write_text(
        '{"cato": ["Letters_Of_Utica.txt"]}', encoding="utf-8"
    )
    monkeypatch.setattr(authors, "_AUTHOR_INDEX_CACHE", None)
    assert authors.author_matches({"source": "Letters_Of_Utica.txt", "text": ""}, ["cato"]) is True


def test_fuse_candidates_keeps_each_retrievers_best_hits():
    items = [{"id": f"v{i}", "agent": "vector", "score": 1 - i / 100} for i in range(30)]
    items += [{"id": f"l{i}", "agent": "lexical", "score": 50 - i} for i in range(30)]
    items += [{"id": "a0", "agent": "lexical_author", "score": 3.0}]
    kept = pipeline.fuse_candidates(items, limit=10)
    ids = {item["id"] for item in kept}
    assert len(kept) == 10
    assert {"v0", "l0", "a0"} <= ids  # every retriever's top hit survives
    assert "v29" not in ids and "l29" not in ids
    assert pipeline.fuse_candidates(items[:5], limit=10) == items[:5]
