# app/domain_packs.py
"""Domain packs: per-domain JSON that externalizes domain knowledge (authors,
stopwords, synonyms, planner hints) out of the request path so the engine is
multi-domain and configurable rather than hard-coded to Stoicism.

Packs live in ``settings.domain_packs_path`` (one JSON file per domain). They
are loaded lazily and cached, with the cache keyed on the directory's mtime
signature so edits are picked up without a restart. A missing/unreadable
directory degrades to base behavior (generic stopwords only).
"""
import json
import logging
import os

from config import settings

_LOG = logging.getLogger(__name__)

# Generic English function/question words shared across every domain (not
# domain-specific). Domain packs add their own stopwords on top of these.
_BASE_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "what", "why", "how", "who", "where", "when", "which", "whom", "whose",
}

_PACKS_CACHE = None
_PACKS_MTIME = None


def _dir_signature(path):
    """Max mtime across the dir and its *.json files, or None if unreadable."""
    try:
        sig = os.path.getmtime(path)
        for name in os.listdir(path):
            if name.endswith(".json"):
                sig = max(sig, os.path.getmtime(os.path.join(path, name)))
        return sig
    except Exception:
        return None


def _load_packs():
    global _PACKS_CACHE, _PACKS_MTIME
    path = settings.domain_packs_path
    sig = _dir_signature(path)
    if _PACKS_CACHE is not None and sig == _PACKS_MTIME:
        return _PACKS_CACHE
    packs = []
    try:
        for name in sorted(os.listdir(path)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(path, name), "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                packs.append(data)
    except Exception as e:
        _LOG.warning("domain_packs_load_failed: %s", e)
    _PACKS_CACHE = packs
    _PACKS_MTIME = sig
    return packs


def _collect_lower_set(key: str) -> set:
    out = set()
    for pack in _load_packs():
        for v in pack.get(key) or []:
            v = str(v).strip().lower()
            if v:
                out.add(v)
    return out


def author_stopwords() -> set:
    return _BASE_STOPWORDS | _collect_lower_set("author_stopwords")


def query_stopwords() -> set:
    return _BASE_STOPWORDS | _collect_lower_set("query_stopwords")


def authors() -> set:
    """Explicitly configured author names across all packs (primary signal for
    author detection; the capitalized-token heuristic is only a fallback)."""
    return _collect_lower_set("authors")


def term_synonyms() -> dict:
    """Merged query-term synonyms across all packs."""
    merged: dict = {}
    for pack in _load_packs():
        syn = pack.get("term_synonyms") or {}
        if not isinstance(syn, dict):
            continue
        for k, v in syn.items():
            key = str(k).strip().lower()
            if not key:
                continue
            merged.setdefault(key, [])
            for val in (v or []):
                val = str(val).strip().lower()
                if val and val not in merged[key]:
                    merged[key].append(val)
    return merged


def planner_rules() -> list:
    """Rule-based planner hints (trigger -> intent/hypotheses/queries) used as a
    fallback when the LLM planner is unavailable."""
    rules = []
    for pack in _load_packs():
        for rule in pack.get("planner_hints") or []:
            if isinstance(rule, dict) and rule.get("triggers"):
                rules.append(rule)
    return rules
