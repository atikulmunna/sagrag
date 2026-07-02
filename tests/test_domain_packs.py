"""Tests for the domain-pack loader and its integration with api helpers."""

import json

import domain_packs


def _write_pack(tmp_path, monkeypatch, data):
    (tmp_path / "test_pack.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(domain_packs.settings, "domain_packs_path", str(tmp_path))
    domain_packs._PACKS_CACHE = None
    domain_packs._PACKS_MTIME = None


def test_shipped_stoicism_pack_loads():
    # The autouse fixture points at data/domain_packs; the stoicism pack ships.
    assert "seneca" in domain_packs.authors()
    assert "fear" in domain_packs.term_synonyms()
    assert "anxiety" in domain_packs.term_synonyms()["fear"]
    rules = domain_packs.planner_rules()
    assert any("fear" in (r.get("triggers") or []) for r in rules)


def test_stopwords_include_base_and_pack(tmp_path, monkeypatch):
    _write_pack(tmp_path, monkeypatch, {"author_stopwords": ["custom"], "query_stopwords": ["foo"]})
    assert "the" in domain_packs.author_stopwords()  # base
    assert "custom" in domain_packs.author_stopwords()  # pack
    assert "foo" in domain_packs.query_stopwords()


def test_term_synonyms_merge_and_dedup(tmp_path, monkeypatch):
    _write_pack(tmp_path, monkeypatch, {"term_synonyms": {"Grief": ["Sorrow", "sorrow", "loss"]}})
    syn = domain_packs.term_synonyms()
    assert syn["grief"] == ["sorrow", "loss"]  # lowercased + deduped


def test_missing_dir_degrades_to_base(tmp_path, monkeypatch):
    monkeypatch.setattr(domain_packs.settings, "domain_packs_path", str(tmp_path / "nope"))
    domain_packs._PACKS_CACHE = None
    domain_packs._PACKS_MTIME = None
    assert domain_packs.authors() == set()
    assert domain_packs.term_synonyms() == {}
    assert domain_packs.planner_rules() == []
    # base stopwords still present
    assert "what" in domain_packs.author_stopwords()
