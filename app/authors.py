# app/authors.py
"""Author-aware retrieval helpers: detect author mentions, expand query terms,
and bias/filter evidence toward the named author."""
import json
import os
import re

import domain_packs
from config import settings

_AUTHOR_INDEX_CACHE = None
_AUTHOR_INDEX_MTIME = 0.0
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")


def merged_term_synonyms() -> dict[str, list[str]]:
    merged = domain_packs.term_synonyms()
    extra = settings.query_term_synonyms or {}
    if isinstance(extra, dict):
        for k, v in extra.items():
            key = str(k).strip().lower()
            if not key:
                continue
            vals = [str(x).strip().lower() for x in (v or []) if str(x).strip()]
            if not vals:
                continue
            merged.setdefault(key, [])
            merged[key].extend(vals)
    return merged


def load_author_index() -> dict[str, list[str]]:
    global _AUTHOR_INDEX_CACHE, _AUTHOR_INDEX_MTIME
    path = settings.author_index_path
    try:
        mtime = os.path.getmtime(path)
        if _AUTHOR_INDEX_CACHE is not None and mtime == _AUTHOR_INDEX_MTIME:
            return _AUTHOR_INDEX_CACHE
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            _AUTHOR_INDEX_CACHE = {str(k).lower(): list(v) for k, v in data.items() if k}
        else:
            _AUTHOR_INDEX_CACHE = {}
        _AUTHOR_INDEX_MTIME = mtime
    except (OSError, ValueError):
        _AUTHOR_INDEX_CACHE = _AUTHOR_INDEX_CACHE or {}
    return _AUTHOR_INDEX_CACHE or {}


def extract_author_mentions(query: str) -> list[str]:
    if not query:
        return []
    q = query.lower()
    stopwords = domain_packs.author_stopwords()
    # Primary: explicitly configured authors (domain packs + author index) and
    # domain keywords present in the query.
    configured = set()
    known = domain_packs.authors() | set(load_author_index().keys())
    for name in known:
        name_l = str(name).strip().lower()
        if len(name_l) >= 3 and name_l not in stopwords and name_l in q:
            configured.add(name_l)
    for keywords in (settings.domain_keywords or {}).values():
        for kw in keywords:
            kw_l = str(kw).strip().lower()
            if not kw_l or len(kw_l) < 3 or kw_l in stopwords:
                continue
            if kw_l in q:
                configured.add(kw_l)
    if configured:
        return sorted(configured)
    # Fallback (low-confidence): capitalized tokens, only when nothing is
    # configured; this is a heuristic, not the primary signal.
    fallback = set()
    for token in _TOKEN_RE.findall(query):
        if token[0].isupper():
            t = token.lower()
            if len(t) >= 3 and t not in stopwords:
                fallback.add(t)
    return sorted(fallback)


def extract_query_terms(query: str, author_terms: list[str]) -> list[str]:
    if not query:
        return []
    stopwords = domain_packs.query_stopwords()
    author_set = {t.lower() for t in author_terms}
    terms = []
    for token in _TOKEN_RE.findall(query):
        t = token.lower()
        if len(t) < 3 or t in stopwords or t in author_set:
            continue
        terms.append(t)
    # expand with simple synonyms, keeping a stable order without duplicates
    synonyms = merged_term_synonyms()
    out = []
    seen = set()
    for t in terms:
        for candidate in [t, *synonyms.get(t, [])]:
            if candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
    return out


def apply_author_bias(results, query: str, bias: float = 0.8):
    terms = extract_author_mentions(query)
    if not terms:
        return results, terms
    out = []
    for r in results:
        source = (r.get("source") or "").lower()
        text = (r.get("text") or "").lower()
        r2 = dict(r)
        if any(t in source for t in terms):
            r2["author_bias"] = bias
        elif any(t in text for t in terms):
            r2["author_bias"] = bias * 0.5
        else:
            r2["author_bias"] = 0.0
        out.append(r2)
    return out, terms


def author_matches(result: dict, author_terms: list[str]) -> bool:
    if not author_terms:
        return False
    source = (result.get("source") or "").lower()
    text = (result.get("text") or "").lower()
    author_index = load_author_index()
    if author_index:
        for t in author_terms:
            sources = author_index.get(t)
            if sources and source in {str(s).lower() for s in sources}:
                return True
    return any(t in source or t in text for t in author_terms)
