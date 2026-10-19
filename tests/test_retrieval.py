"""Retrieval fan-out, structured extraction, routing, and domain discovery."""

import agents


def test_structured_rows_are_not_duplicated():
    # Regression: table-like rows were appended twice.
    assert agents._extract_structured_lines("a,b,c\nx: 1", tokens=[], max_lines=6) == [
        "a,b,c",
        "x: 1",
    ]


def test_structured_from_lexical_offsets_point_at_the_line():
    text = "Intro prose here.\nName: Seneca\nEra: Roman"
    hits = [
        {"id": "c1", "score": 2.0, "source": {"text": text, "source": "s.txt", "offset_start": 100}}
    ]
    out = agents.structured_from_lexical(hits, "seneca name")
    assert len(out) == 1
    line = out[0]["source"]
    assert line["text"] == "Name: Seneca"
    assert (line["offset_start"], line["offset_end"]) == (
        100 + text.find("Name: Seneca"),
        100 + text.find("Name: Seneca") + 12,
    )
    assert out[0]["id"] == "c1::line::0" and out[0]["agent"] == "structured"


async def test_run_agents_embeds_once_and_derives_structured(monkeypatch):
    calls = {"encode": [], "vector": [], "lexical": []}

    async def fake_encode(texts):
        calls["encode"].append(list(texts))
        return [[0.1, 0.2] for _ in texts]

    async def fake_vector(q, domain=None, tenant=None, return_status=False, vector=None, **kw):
        calls["vector"].append((q, domain, vector))
        return [
            {"id": f"v-{q}-{domain}", "score": 0.5, "payload": {"text": q}, "agent": "vector"}
        ], "ok"

    async def fake_lexical(q, domain=None, tenant=None, return_status=False, **kw):
        calls["lexical"].append((q, domain))
        hit = {
            "id": f"l-{q}-{domain}",
            "score": 3.0,
            "agent": "lexical",
            "source": {"text": f"key: {q}", "source": "s.csv", "offset_start": 0},
        }
        return [hit], "ok"

    monkeypatch.setattr(agents.embeddings, "encode_cached", fake_encode)
    monkeypatch.setattr(agents, "vector_search", fake_vector)
    monkeypatch.setattr(agents, "lexical_search", fake_lexical)
    results, diag = await agents.run_agents(
        ["alpha", "beta"], domain="d1", fallback_domains=["d2"], tenant=None
    )
    # One batched embedding call for all sub-queries, reused across domains.
    assert calls["encode"] == [["alpha", "beta"]]
    assert all(vec == [0.1, 0.2] for _, _, vec in calls["vector"])
    # One lexical query per (query, domain); structured adds no extra ES call.
    assert len(calls["lexical"]) == 4
    assert diag["structured"]["attempted"] == 4 and diag["structured"]["ok"] == 4
    assert {r["agent"] for r in results} == {"vector", "lexical", "structured"}


async def test_run_agents_structured_inherits_lexical_failure(monkeypatch):
    async def fake_encode(texts):
        return [None for _ in texts]

    async def fake_vector(*a, **kw):
        return [], "zero_hits"

    async def fake_lexical(*a, **kw):
        return [], "timeout"

    monkeypatch.setattr(agents.embeddings, "encode_cached", fake_encode)
    monkeypatch.setattr(agents, "vector_search", fake_vector)
    monkeypatch.setattr(agents, "lexical_search", fake_lexical)
    _, diag = await agents.run_agents(["q"])
    assert diag["structured"] == {"attempted": 1, "ok": 0, "zero_hits": 0, "timeout": 1, "error": 0}


def test_route_domain_matches_word_starts_only(monkeypatch):
    monkeypatch.setattr(agents, "_list_domain_indices", lambda tenant=None: ["art"])
    monkeypatch.setattr(agents.settings, "domain_keywords", {})
    assert agents.route_domain("how do I start a fire") is None
    assert agents.route_domain("famous artists and art history") == "art"


def test_known_domains_merges_config_packs_and_discovery(monkeypatch):
    monkeypatch.setattr(agents.settings, "domain_keywords", {"Finance": ["revenue"]})
    monkeypatch.setattr(agents.settings, "domain_aliases", {})
    monkeypatch.setattr(agents.settings, "domain_fallbacks", [])
    monkeypatch.setattr(agents.settings, "domain_index_map", {})
    monkeypatch.setattr(
        agents, "_list_domain_indices", lambda tenant=None: ["history"] if tenant == "acme" else []
    )
    assert agents.known_domains("acme") == {"finance", "stoicism", "history"}
    assert "history" not in agents.known_domains("globex")


def test_list_domain_indices_is_tenant_aware(monkeypatch):
    class FakeIndices:
        def get(self, index, ignore_unavailable=True):
            return {
                "docs_index_stoicism": {},
                "docs_index_t_acme__d_finance": {},
                "docs_index_t_acme": {},
            }

    class FakeES:
        indices = FakeIndices()

    monkeypatch.setattr(agents, "_get_es", lambda: FakeES())
    agents._DOMAIN_INDEX_CACHE.clear()
    assert agents._list_domain_indices(None) == ["stoicism"]
    assert agents._list_domain_indices("acme") == ["finance"]
    agents._DOMAIN_INDEX_CACHE.clear()


def test_es_targets_ignore_index_map_for_tenants(monkeypatch):
    monkeypatch.setattr(agents.settings, "domain_index_map", {"stoicism": "custom_index"})
    assert agents._es_targets(None, "stoicism") == ["custom_index", "docs_index"]
    assert agents._es_targets("acme", "stoicism") == ["docs_index_t_acme__d_stoicism"]


def test_freshness_filter_tolerates_bad_timestamps():
    import time

    results = [
        {"text": "a", "timestamp": "not-a-number"},
        {"text": "b", "timestamp": time.time() - 99 * 86400},
    ]
    assert [r["text"] for r in agents.apply_freshness_filter(results, freshness_days=5)] == ["a"]


def test_structured_lines_skip_prose_paragraphs():
    prose = (
        "You have asked me, Lucilius, why, if the world be ruled by providence, so many evils befall "
        "good men? The answer is that God is amongst us: but, since you wish it, I will plead the cause."
    )
    text = "\n".join([prose, "Name: Seneca", "a.b.c: 1", "h1: 1 | h2: 2", "revenue, 100, usd"])
    lines = agents._extract_structured_lines(text, tokens=[], max_lines=10)
    assert lines == ["Name: Seneca", "a.b.c: 1", "h1: 1 | h2: 2", "revenue, 100, usd"]
