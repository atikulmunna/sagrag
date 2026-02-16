# app/agents.py
import asyncio
import time
import contextlib
import logging
from config import settings
try:
    from opentelemetry import trace
    _TRACER = trace.get_tracer(__name__)
except Exception:
    _TRACER = None

_LOG = logging.getLogger(__name__)

QDRANT_COLLECTION = "docs"
ELASTIC_INDEX = "docs_index"

_QDRANT = None
_ES = None
_DOMAIN_INDEX_CACHE = None

def _get_qdrant():
    global _QDRANT
    if _QDRANT is None:
        from qdrant_client import QdrantClient
        _QDRANT = QdrantClient(url=settings.qdrant_url)
    return _QDRANT

def _get_es():
    global _ES
    if _ES is None:
        from elasticsearch import Elasticsearch
        _ES = Elasticsearch(settings.elastic_url)
    return _ES

def _list_domain_indices():
    global _DOMAIN_INDEX_CACHE
    if _DOMAIN_INDEX_CACHE is not None:
        return _DOMAIN_INDEX_CACHE
    es = _get_es()
    indices = []
    try:
        data = es.indices.get(index=f"{ELASTIC_INDEX}_*")
        indices = list(data.keys())
    except Exception:
        indices = []
    domains = []
    for idx in indices:
        if idx.startswith(f"{ELASTIC_INDEX}_"):
            domains.append(idx.replace(f"{ELASTIC_INDEX}_", ""))
    _DOMAIN_INDEX_CACHE = domains
    return domains

_EMB_MODEL = None
def _load_embed_model():
    global _EMB_MODEL
    if _EMB_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMB_MODEL = SentenceTransformer(settings.embedding_model_name)
    return _EMB_MODEL

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

async def vector_search(
    query_text: str,
    k: int = 6,
    timeout_s: float | None = None,
    domain: str | None = None,
    tenant: str | None = None,
    return_status: bool = False,
):
    if not isinstance(query_text, str):
        query_text = str(query_text)
    timeout_s = float(timeout_s or settings.retriever_timeout_s)
    start = time.monotonic()
    span_cm = _TRACER.start_as_current_span("retriever.vector") if _TRACER else contextlib.nullcontext()
    def _sync_search():
        model = _load_embed_model()
        q_emb = model.encode([query_text])[0].tolist()
        qdrant = _get_qdrant()
        collections = []
        if domain:
            collections.append(f"{QDRANT_COLLECTION}_{domain}")
        elif tenant:
            collections.append(f"{QDRANT_COLLECTION}_{tenant}")
        collections.append(QDRANT_COLLECTION)

        last_err = None
        resp = None
        for collection in collections:
            try:
                # Older client path.
                resp = qdrant.search(collection_name=collection, query_vector=q_emb, limit=k)
                break
            except Exception as e_search:
                last_err = e_search
                try:
                    # Newer client path (returns QueryResponse with .points).
                    resp = qdrant.query_points(collection_name=collection, query=q_emb, limit=k)
                    break
                except Exception as e_query:
                    last_err = e_query
                    resp = None
                    continue
        if resp is None and last_err is not None:
            raise last_err

        points = _qdrant_points(resp)
        results = []
        for r in points:
            rid = _point_field(r, "id")
            if rid is None:
                continue
            try:
                score = float(_point_field(r, "score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            payload = _point_field(r, "payload", {}) or {}
            results.append({"id": rid, "score": score, "payload": payload})
        return results
    with span_cm as span:
        if span:
            span.set_attribute("retriever.k", k)
            span.set_attribute("retriever.domain", domain or "")
        try:
            results = await asyncio.wait_for(asyncio.to_thread(_sync_search), timeout=timeout_s)
        except asyncio.TimeoutError:
            return ([], "timeout") if return_status else []
        except Exception as e:
            _LOG.warning("vector_search_failed", exc_info=e)
            return ([], "error") if return_status else []
    elapsed_ms = int((time.monotonic() - start) * 1000)
    for r in results:
        r["agent"] = "vector"
        r["elapsed_ms"] = elapsed_ms
    status = "ok" if results else "zero_hits"
    return (results, status) if return_status else results

async def lexical_search(
    query_text: str,
    k: int = 6,
    timeout_s: float | None = None,
    domain: str | None = None,
    tenant: str | None = None,
    return_status: bool = False,
):
    if not isinstance(query_text, str):
        query_text = str(query_text)
    timeout_s = float(timeout_s or settings.retriever_timeout_s)
    start = time.monotonic()
    span_cm = _TRACER.start_as_current_span("retriever.lexical") if _TRACER else contextlib.nullcontext()
    def _sync_search():
        body = {"query": {"match": {"text": {"query": query_text}}}, "size": k}
        es = _get_es()
        index = ELASTIC_INDEX
        if tenant:
            index = f"{index}_{tenant}"
        if domain:
            index = settings.domain_index_map.get(domain, f"{ELASTIC_INDEX}_{domain}")
        try:
            resp = es.search(index=index, body=body)
        except Exception:
            resp = es.search(index=ELASTIC_INDEX, body=body)
        hits = resp["hits"]["hits"]
        return [{"id": h["_id"], "score": float(h["_score"]), "source": h["_source"]} for h in hits]
    with span_cm as span:
        if span:
            span.set_attribute("retriever.k", k)
            span.set_attribute("retriever.domain", domain or "")
        try:
            results = await asyncio.wait_for(asyncio.to_thread(_sync_search), timeout=timeout_s)
        except asyncio.TimeoutError:
            return ([], "timeout") if return_status else []
        except Exception:
            return ([], "error") if return_status else []
    elapsed_ms = int((time.monotonic() - start) * 1000)
    for r in results:
        r["agent"] = "lexical"
        r["elapsed_ms"] = elapsed_ms
    status = "ok" if results else "zero_hits"
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
        return []
    if not isinstance(query_text, str):
        query_text = str(query_text)
    timeout_s = float(timeout_s or settings.retriever_timeout_s)
    span_cm = _TRACER.start_as_current_span("retriever.lexical.author") if _TRACER else contextlib.nullcontext()
    def _sync_search():
        # Match author terms against source.keyword with a wildcard so
        # "seneca" finds "seneca_letters.txt".
        author_filters = [{"wildcard": {"source.keyword": f"*{t}*"}} for t in author_terms]
        def _body(must_terms: bool):
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
                "query": {
                    "bool": {
                        "filter": [{
                            "bool": {
                                "should": author_filters,
                                "minimum_should_match": 1,
                            }
                        }],
                        "must": must_clauses,
                        "should": [{"match": {"text": {"query": query_text}}}],
                    }
                },
                "size": k,
            }
        es = _get_es()
        index = ELASTIC_INDEX
        if tenant:
            index = f"{index}_{tenant}"
        if domain:
            index = settings.domain_index_map.get(domain, f"{ELASTIC_INDEX}_{domain}")
        try:
            body = _body(must_terms=True)
            resp = es.search(index=index, body=body)
        except Exception:
            body = _body(must_terms=True)
            resp = es.search(index=ELASTIC_INDEX, body=body)
        hits = resp["hits"]["hits"]
        if required_terms and not hits:
            # Fallback: relax term requirement if it yields zero results.
            try:
                resp = es.search(index=index, body=_body(must_terms=False))
            except Exception:
                resp = es.search(index=ELASTIC_INDEX, body=_body(must_terms=False))
            hits = resp["hits"]["hits"]
        return [{"id": h["_id"], "score": float(h["_score"]), "source": h["_source"]} for h in hits]
    with span_cm as span:
        if span:
            span.set_attribute("retriever.k", k)
            span.set_attribute("retriever.domain", domain or "")
        try:
            results = await asyncio.wait_for(asyncio.to_thread(_sync_search), timeout=timeout_s)
        except asyncio.TimeoutError:
            return ([], "timeout") if return_status else []
        except Exception:
            return ([], "error") if return_status else []
    for r in results:
        r["agent"] = "lexical_author"
    status = "ok" if results else "zero_hits"
    return (results, status) if return_status else results

def _extract_structured_lines(text: str, tokens: list[str], max_lines: int = 6):
    lines = []
    for line in text.splitlines():
        l = line.strip()
        if not l:
            continue
        if ":" not in l and "|" not in l and "\t" not in l and "," not in l:
            continue
        if tokens and not any(t in l.lower() for t in tokens):
            continue
        # Prefer header-like rows for tables
        if l.count(",") >= 2 or l.count("|") >= 2 or "\t" in l:
            lines.append(l[:400])
            if len(lines) >= max_lines:
                break
        lines.append(l[:400])
        if len(lines) >= max_lines:
            break
    return lines

async def structured_search(
    query_text: str,
    k: int = 6,
    timeout_s: float | None = None,
    domain: str | None = None,
    tenant: str | None = None,
    return_status: bool = False,
):
    timeout_s = float(timeout_s or settings.retriever_timeout_s)
    span_cm = _TRACER.start_as_current_span("retriever.structured") if _TRACER else contextlib.nullcontext()
    with span_cm as span:
        if span:
            span.set_attribute("retriever.k", k)
            span.set_attribute("retriever.domain", domain or "")
        if not isinstance(query_text, str):
            query_text = str(query_text)
        tokens = [t for t in query_text.lower().split() if len(t) > 2]
        base, base_status = await lexical_search(
            query_text,
            k=k,
            timeout_s=timeout_s,
            domain=domain,
            tenant=tenant,
            return_status=True,
        )
        out = []
        for r in base:
            source = r.get("source") or {}
            text = source.get("text") or ""
            for idx, line in enumerate(_extract_structured_lines(text, tokens, max_lines=3)):
                out.append({
                    "id": f"{r.get('id')}::line::{idx}",
                    "score": r.get("score"),
                    "source": {
                        "text": line,
                        "source": source.get("source"),
                        "timestamp": source.get("timestamp"),
                        "source_type": source.get("source_type"),
                        "domain": source.get("domain"),
                    },
                    "offset_start": r.get("offset_start"),
                    "offset_end": r.get("offset_end"),
                    "agent": "structured",
                    "elapsed_ms": r.get("elapsed_ms"),
                })
                if len(out) >= k:
                    status = "ok" if out else base_status
                    return (out, status) if return_status else out
        status = "ok" if out else base_status
        return (out, status) if return_status else out

async def run_agents(queries, domain: str | None = None, fallback_domains: list[str] | None = None, tenant: str | None = None):
    tasks = []
    task_meta = []
    search_domains = [domain] if domain else []
    if fallback_domains:
        for d in fallback_domains:
            if d and d not in search_domains:
                search_domains.append(d)
    if not search_domains:
        search_domains = [None]
    for q in queries:
        for d in search_domains:
            tasks.append(vector_search(q, domain=d, tenant=tenant, return_status=True))
            task_meta.append("vector")
            tasks.append(lexical_search(q, domain=d, tenant=tenant, return_status=True))
            task_meta.append("lexical")
            tasks.append(structured_search(q, domain=d, tenant=tenant, return_status=True))
            task_meta.append("structured")
    res = await asyncio.gather(*tasks)
    diagnostics = {}
    flattened = []
    for agent_name, item in zip(task_meta, res):
        results, status = item
        flattened.extend(results)
        entry = diagnostics.setdefault(agent_name, {"attempted": 0, "ok": 0, "zero_hits": 0, "timeout": 0, "error": 0})
        entry["attempted"] += 1
        entry[status] = entry.get(status, 0) + 1
    seen = set()
    out = []
    for r in flattened:
        rid = r.get("id")
        if rid not in seen:
            seen.add(rid)
            out.append(r)
    return out, diagnostics

def route_domain(query: str, constraints: dict | None = None, preferences: dict | None = None):
    if constraints and isinstance(constraints, dict):
        if constraints.get("domain"):
            return str(constraints.get("domain")).lower()
    if preferences and isinstance(preferences, dict):
        if preferences.get("domain"):
            return str(preferences.get("domain")).lower()
    # Prefer available index domains when a direct mention appears.
    q = (query or "").lower()
    for d in _list_domain_indices():
        if d and d.lower() in q:
            return d
    best_domain = None
    best_score = 0
    for domain, keywords in (settings.domain_keywords or {}).items():
        score = 0
        for kw in keywords:
            if kw and kw.lower() in q:
                score += 1
        for alias in (settings.domain_aliases or {}).get(domain, []):
            if alias and alias.lower() in q:
                score += 2
        if domain and domain.lower() in q:
            score += 2
        if score > best_score:
            best_score = score
            best_domain = domain
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
        if rules:
            decision = None
            for rule in rules:
                action = (rule.get("action") or "").lower()
                if action not in ("allow", "deny"):
                    continue
                if _match_rule(rule, text, source_type, domain):
                    decision = action
                    break
            if decision == "deny":
                continue
            if decision == "allow":
                out.append(r)
                continue
        out.append(r)
    return out

def apply_freshness_filter(results, freshness_days=None):
    if freshness_days is None:
        return results
    try:
        cutoff = time.time() - float(freshness_days) * 86400.0
    except Exception:
        return results
    out = []
    for r in results:
        ts = r.get("timestamp")
        if ts is not None and ts < cutoff:
            continue
        out.append(r)
    return out
