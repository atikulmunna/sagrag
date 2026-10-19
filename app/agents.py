# app/agents.py
import asyncio
import contextlib
import logging
import math
import re
import time

import domain_packs
import embeddings
import namespaces
from config import settings
try:
    from opentelemetry import trace
    _TRACER = trace.get_tracer(__name__)
except Exception:
    _TRACER = None

_LOG = logging.getLogger(__name__)

# Kept for backwards compatibility with code that imported these names.
QDRANT_COLLECTION = namespaces.BASE_COLLECTION
ELASTIC_INDEX = namespaces.BASE_INDEX

_QDRANT = None
_ES = None
_DOMAIN_INDEX_CACHE: dict = {}


def _get_qdrant():
    global _QDRANT
    if _QDRANT is None:
        from qdrant_client import QdrantClient
        # Client-side timeout so a search abandoned by asyncio.wait_for also
        # ends its worker thread instead of holding it indefinitely.
        _QDRANT = QdrantClient(url=settings.qdrant_url, timeout=int(math.ceil(settings.retriever_timeout_s)))
    return _QDRANT


def _get_es():
    global _ES
    if _ES is None:
        from elasticsearch import Elasticsearch
        _ES = Elasticsearch(settings.elastic_url, request_timeout=settings.retriever_timeout_s)
    return _ES


def _list_domain_indices(tenant: str | None = None):
    """Domains with an ES index in the caller's layout (cached with a TTL)."""
    key = tenant or ""
    hit = _DOMAIN_INDEX_CACHE.get(key)
    now = time.monotonic()
    if hit and now - hit[0] < settings.domain_cache_ttl_s:
        return hit[1]
    try:
        data = _get_es().indices.get(index=f"{ELASTIC_INDEX}_*", ignore_unavailable=True)
        domains = namespaces.domains_from_index_names(list(data.keys()), tenant)
    except Exception as e:
        _LOG.info("domain_index_listing_failed: %s", e)
        domains = []
    _DOMAIN_INDEX_CACHE[key] = (now, domains)
    return domains


def _configured_domains() -> list[str]:
    names = list((settings.domain_keywords or {}).keys())
    names += list((settings.domain_aliases or {}).keys())
    names += list(settings.domain_fallbacks or [])
    names += list((settings.domain_index_map or {}).keys())
    names += domain_packs.pack_names()
    out = []
    for n in names:
        slug = namespaces.slugify(n)
        if slug and slug not in out:
            out.append(slug)
    return out


def known_domains(tenant: str | None = None) -> set[str]:
    """Every domain a caller may select: configured ones plus those discovered
    in the caller's own index layout."""
    return set(_configured_domains()) | set(_list_domain_indices(tenant))


def _point_field(point, name, default=None):
    try:
        if isinstance(point, dict):
            return point.get(name, default)
        return getattr(point, name, default)
    except Exception:
        return default


def _qdrant_points(resp):
    if resp is None:
        return []
    if isinstance(resp, list):
        return resp
    pts = _point_field(resp, "points")
    if isinstance(pts, list):
        return pts
    if isinstance(resp, dict):
        result = resp.get("result")
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            pts2 = result.get("points")
            if isinstance(pts2, list):
                return pts2
    return []


def _es_targets(tenant: str | None, domain: str | None) -> list[str]:
    targets = [index for _, index in namespaces.search_targets(tenant, domain)]
    # Explicit index overrides only apply to the shared layout.
    if domain and not tenant and domain in (settings.domain_index_map or {}):
        targets[0] = settings.domain_index_map[domain]
    return targets


def _span(name: str):
    return _TRACER.start_as_current_span(name) if _TRACER else contextlib.nullcontext()


async def _run_retriever(name, sync_fn, timeout_s, k, domain):
    """Run a blocking search in a thread; returns (results, status)."""
    with _span(name) as span:
        if span:
            span.set_attribute("retriever.k", k)
            span.set_attribute("retriever.domain", domain or "")
        try:
            return await asyncio.wait_for(asyncio.to_thread(sync_fn), timeout=timeout_s), None
        except asyncio.TimeoutError:
            return [], "timeout"
        except Exception as e:
            _LOG.warning("%s_failed: %r", name, e)
            return [], "error"


async def vector_search(
    query_text: str,
    k: int = 6,
    timeout_s: float | None = None,
    domain: str | None = None,
    tenant: str | None = None,
    return_status: bool = False,
    vector: list[float] | None = None,
):
    query_text = str(query_text)
    timeout_s = float(timeout_s or settings.retriever_timeout_s)
    start = time.monotonic()

    def _sync_search():
        q_emb = vector if vector is not None else embeddings.encode([query_text])[0]
        qdrant = _get_qdrant()
        last_err = None
        for collection, _ in namespaces.search_targets(tenant, domain):
            try:
                try:
                    resp = qdrant.query_points(collection_name=collection, query=q_emb, limit=k)
                except AttributeError:
                    # Older client without query_points.
                    resp = qdrant.search(collection_name=collection, query_vector=q_emb, limit=k)
                break
            except Exception as e:
                last_err = e
        else:
            if last_err is not None:
                raise last_err
            return []
        results = []
        for r in _qdrant_points(resp):
            rid = _point_field(r, "id")
            if rid is None:
                continue
            try:
                score = float(_point_field(r, "score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            payload = _point_field(r, "payload", {}) or {}
            results.append({"id": rid, "score": score, "payload": payload})
        return results

    results, status = await _run_retriever("retriever.vector", _sync_search, timeout_s, k, domain)
    elapsed_ms = int((time.monotonic() - start) * 1000)
    for r in results:
        r["agent"] = "vector"
        r["elapsed_ms"] = elapsed_ms
    status = status or ("ok" if results else "zero_hits")
    return (results, status) if return_status else results


def _es_search(targets: list[str], **kwargs):
    """Search the first index that answers; raise the last error if none do."""
    es = _get_es()
    last_err = None
    for index in targets:
        try:
            return es.search(index=index, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err if last_err is not None else RuntimeError("no search targets")


async def lexical_search(
    query_text: str,
    k: int = 6,
    timeout_s: float | None = None,
    domain: str | None = None,
    tenant: str | None = None,
    return_status: bool = False,
):
    query_text = str(query_text)
    timeout_s = float(timeout_s or settings.retriever_timeout_s)
    start = time.monotonic()

    def _sync_search():
        resp = _es_search(_es_targets(tenant, domain), query={"match": {"text": {"query": query_text}}}, size=k)
        hits = resp["hits"]["hits"]
        return [{"id": h["_id"], "score": float(h["_score"]), "source": h["_source"]} for h in hits]

    results, status = await _run_retriever("retriever.lexical", _sync_search, timeout_s, k, domain)
    elapsed_ms = int((time.monotonic() - start) * 1000)
    for r in results:
        r["agent"] = "lexical"
        r["elapsed_ms"] = elapsed_ms
    status = status or ("ok" if results else "zero_hits")
    return (results, status) if return_status else results


async def author_lexical_search(
    query_text: str,
    author_terms: list[str],
    required_terms: list[str] | None = None,
    k: int = 6,
    timeout_s: float | None = None,
    domain: str | None = None,
    tenant: str | None = None,
    return_status: bool = False,
):
    if not author_terms:
        return ([], "zero_hits") if return_status else []
    query_text = str(query_text)
    timeout_s = float(timeout_s or settings.retriever_timeout_s)

    def _sync_search():
        # Match author terms against source.keyword with a wildcard so
        # "seneca" finds "seneca_letters.txt".
        author_filters = [{"wildcard": {"source.keyword": f"*{t}*"}} for t in author_terms]

        def _query(must_terms: bool):
            must_clauses = []
            if must_terms and required_terms:
                must_clauses.append({
                    "simple_query_string": {
                        "query": " ".join(required_terms),
                        "fields": ["text"],
                        "default_operator": "or",
                    }
                })
            return {
                "bool": {
                    "filter": [{"bool": {"should": author_filters, "minimum_should_match": 1}}],
                    "must": must_clauses,
                    "should": [{"match": {"text": {"query": query_text}}}],
                }
            }

        targets = _es_targets(tenant, domain)
        resp = _es_search(targets, query=_query(must_terms=True), size=k)
        hits = resp["hits"]["hits"]
        if required_terms and not hits:
            # Fallback: relax term requirement if it yields zero results.
            hits = _es_search(targets, query=_query(must_terms=False), size=k)["hits"]["hits"]
        return [{"id": h["_id"], "score": float(h["_score"]), "source": h["_source"]} for h in hits]

    results, status = await _run_retriever("retriever.lexical.author", _sync_search, timeout_s, k, domain)
    for r in results:
        r["agent"] = "lexical_author"
    status = status or ("ok" if results else "zero_hits")
    return (results, status) if return_status else results


_KEY_VALUE_RE = re.compile(r"^[\w .\-/\[\]]{1,40}:\s*\S")
_SENTENCE_BREAK_RE = re.compile(r"[.!?]\s")


def _looks_structured(line: str) -> bool:
    """Table rows, key/value lines, or short comma-separated records; not prose.

    Prose paragraphs also contain commas and colons, so a line only counts as
    a record when its shape says so: 2+ pipes or a tab, a short key before a
    colon ("Name: Seneca", flattened JSON "a.b: 1"), or 3+ short comma cells
    with no sentence break.
    """
    if line.count("|") >= 2 or "\t" in line:
        return True
    if _KEY_VALUE_RE.match(line):
        return True
    cells = [c.strip() for c in line.split(",")]
    return (
        len(cells) >= 3
        and len(line) <= 200
        and max(len(c) for c in cells) <= 40
        and not _SENTENCE_BREAK_RE.search(line)
    )


def _extract_structured_lines(text: str, tokens: list[str], max_lines: int = 6):
    lines = []
    for line in text.splitlines():
        l = line.strip()
        if not l or not _looks_structured(l):
            continue
        if tokens and not any(t in l.lower() for t in tokens):
            continue
        lines.append(l[:400])
        if len(lines) >= max_lines:
            break
    return lines


def structured_from_lexical(lexical_results, query_text: str, k: int = 6):
    """Structured (table/key-value) lines pulled out of lexical hits.

    Derived in-process from results we already have, so the structured agent
    costs no extra Elasticsearch round trip. Offsets point at the exact line.
    """
    tokens = [t for t in str(query_text).lower().split() if len(t) > 2]
    out = []
    for r in lexical_results:
        source = r.get("source") or {}
        text = source.get("text") or ""
        chunk_start = source.get("offset_start")
        for idx, line in enumerate(_extract_structured_lines(text, tokens, max_lines=3)):
            offset_start = offset_end = None
            pos = text.find(line)
            if isinstance(chunk_start, int) and pos >= 0:
                offset_start = chunk_start + pos
                offset_end = offset_start + len(line)
            out.append({
                "id": f"{r.get('id')}::line::{idx}",
                "score": r.get("score"),
                "source": {
                    "text": line,
                    "source": source.get("source"),
                    "timestamp": source.get("timestamp"),
                    "source_type": source.get("source_type"),
                    "domain": source.get("domain"),
                    "offset_start": offset_start,
                    "offset_end": offset_end,
                },
                "agent": "structured",
                "elapsed_ms": r.get("elapsed_ms"),
            })
            if len(out) >= k:
                return out
    return out


async def structured_search(
    query_text: str,
    k: int = 6,
    timeout_s: float | None = None,
    domain: str | None = None,
    tenant: str | None = None,
    return_status: bool = False,
):
    base, base_status = await lexical_search(
        query_text, k=k, timeout_s=timeout_s, domain=domain, tenant=tenant, return_status=True
    )
    out = structured_from_lexical(base, query_text, k=k)
    status = "ok" if out else base_status
    return (out, status) if return_status else out


def _bump(diagnostics: dict, agent: str, status: str):
    entry = diagnostics.setdefault(agent, {"attempted": 0, "ok": 0, "zero_hits": 0, "timeout": 0, "error": 0})
    entry["attempted"] += 1
    entry[status] = entry.get(status, 0) + 1


async def run_agents(queries, domain: str | None = None, fallback_domains: list[str] | None = None, tenant: str | None = None):
    search_domains = [domain] if domain else []
    for d in fallback_domains or []:
        if d and d not in search_domains:
            search_domains.append(d)
    if not search_domains:
        search_domains = [None]
    # Embed every sub-query once (Redis-cached) instead of once per domain.
    try:
        vectors = await asyncio.wait_for(embeddings.encode_cached(list(queries)), timeout=settings.retriever_timeout_s)
    except Exception as e:
        _LOG.warning("query_embedding_failed: %r", e)
        vectors = [None] * len(queries)
    tasks = []
    task_meta = []
    for q, vec in zip(queries, vectors):
        for d in search_domains:
            tasks.append(vector_search(q, domain=d, tenant=tenant, return_status=True, vector=vec))
            task_meta.append(("vector", q))
            tasks.append(lexical_search(q, domain=d, tenant=tenant, return_status=True))
            task_meta.append(("lexical", q))
    res = await asyncio.gather(*tasks)
    diagnostics: dict = {}
    flattened = []
    for (agent_name, q), (results, status) in zip(task_meta, res):
        flattened.extend(results)
        _bump(diagnostics, agent_name, status)
        if agent_name == "lexical":
            structured = structured_from_lexical(results, q)
            flattened.extend(structured)
            if structured:
                _bump(diagnostics, "structured", "ok")
            else:
                _bump(diagnostics, "structured", status if status in ("timeout", "error") else "zero_hits")
    seen = set()
    out = []
    for r in flattened:
        rid = r.get("id")
        if rid not in seen:
            seen.add(rid)
            out.append(r)
    return out, diagnostics


def _mentions(term: str, text: str) -> bool:
    """Word-start match: "stoic" matches "stoicism" but "art" does not match "start"."""
    return bool(term) and re.search(r"\b" + re.escape(term.lower()), text) is not None


def route_domain(query: str, constraints: dict | None = None, preferences: dict | None = None, tenant: str | None = None):
    """Pick a domain. Explicit constraint/preference values must already be
    validated by the caller (see pipeline); keyword routing only ever returns
    configured or discovered domains."""
    if constraints and isinstance(constraints, dict):
        if constraints.get("domain"):
            return str(constraints.get("domain")).lower()
    if preferences and isinstance(preferences, dict):
        if preferences.get("domain"):
            return str(preferences.get("domain")).lower()
    q = (query or "").lower()
    # Prefer available index domains when a direct mention appears.
    for d in _list_domain_indices(tenant):
        if _mentions(d, q):
            return d
    best_domain = None
    best_score = 0
    for domain, keywords in (settings.domain_keywords or {}).items():
        slug = namespaces.slugify(domain)
        if not slug:
            continue
        score = sum(1 for kw in keywords if kw and _mentions(kw, q))
        score += 2 * sum(1 for alias in (settings.domain_aliases or {}).get(domain, []) if alias and _mentions(alias, q))
        if _mentions(slug, q):
            score += 2
        if score > best_score:
            best_score = score
            best_domain = slug
    if best_score >= settings.domain_min_keyword_hits:
        return best_domain
    return None


def apply_policy_filter(results, blocklist=None):
    if not blocklist:
        return results
    out = []
    for r in results:
        text = r.get("text", "")
        if any(b in text.lower() for b in blocklist):
            continue
        out.append(r)
    return out


def _match_rule(rule, text, source_type, domain):
    contains = rule.get("contains") or []
    not_contains = rule.get("not_contains") or []
    domains = rule.get("domains") or []
    source_types = rule.get("source_types") or []
    if domains and domain not in domains:
        return False
    if source_types and source_type not in source_types:
        return False
    if contains and not any(c in text.lower() for c in contains):
        return False
    if not_contains and any(c in text.lower() for c in not_contains):
        return False
    return True


def apply_policy_rules(results, allowlist=None, source_types_allow=None, source_types_block=None, domains_allow=None, domains_block=None, rules=None):
    """Filter results. Rules are evaluated in order and the first match wins;
    ``deny`` drops the result, ``allow`` keeps it, no match keeps it."""
    out = []
    for r in results:
        text = r.get("text", "")
        source_type = r.get("source_type") or ""
        domain = r.get("domain") or ""
        if allowlist and not any(a in text.lower() for a in allowlist):
            continue
        if source_types_allow and source_type and source_type not in source_types_allow:
            continue
        if source_types_block and source_type and source_type in source_types_block:
            continue
        if domains_allow and domain and domain not in domains_allow:
            continue
        if domains_block and domain and domain in domains_block:
            continue
        decision = None
        for rule in rules or []:
            action = (rule.get("action") or "").lower()
            if action in ("allow", "deny") and _match_rule(rule, text, source_type, domain):
                decision = action
                break
        if decision == "deny":
            continue
        out.append(r)
    return out


def apply_freshness_filter(results, freshness_days=None):
    if freshness_days is None:
        return results
    try:
        cutoff = time.time() - float(freshness_days) * 86400.0
    except (TypeError, ValueError):
        return results
    out = []
    for r in results:
        try:
            ts = float(r.get("timestamp")) if r.get("timestamp") is not None else None
        except (TypeError, ValueError):
            ts = None
        if ts is not None and ts < cutoff:
            continue
        out.append(r)
    return out
