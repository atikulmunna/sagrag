# app/api.py
from fastapi import APIRouter, Request, HTTPException, Response
import json
import os
import re
from ingestion import ingest_folder
from speculative import plan_query
from agents import run_agents, route_domain, apply_policy_filter, apply_freshness_filter, apply_policy_rules, author_lexical_search
from reranker import rerank
from judge import judge_evidence, build_graph_context, build_graph_reasoning
from synthesis import synthesize_answer
from config import settings
from store import log_query_result, fetch_audit_logs, export_audit_jsonl, log_feedback, fetch_feedback
from metrics import render_prometheus, record_author_gap, record_author_query, record_retrieval_failure, record_hallucination_risk
from continuous_learning import export_training_data, export_default_training_data

router = APIRouter()

def _parse_blocklist(value):
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip().lower() for v in value.split(",") if v.strip()]
    return []

_AUTHOR_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "what", "why", "how", "who", "where", "when", "which", "whom", "whose",
    "stoic", "stoics", "stoicism", "philosophy", "philosopher", "guidance",
    "ethics", "roman", "advice",
}
_QUERY_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "what", "why", "how", "who", "where", "when", "which", "whom", "whose",
    "does", "say", "about", "handle",
}
_TERM_SYNONYMS = {
    "fear": ["fear", "fears", "afraid", "anxiety", "anxious", "terror", "dread", "timor", "metus"],
    "death": ["death", "die", "dying", "mortality", "mors"],
}

def _merged_term_synonyms() -> dict[str, list[str]]:
    merged = dict(_TERM_SYNONYMS)
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

_AUTHOR_INDEX_CACHE = None
_AUTHOR_INDEX_MTIME = 0.0

def _load_author_index() -> dict[str, list[str]]:
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
    except Exception:
        _AUTHOR_INDEX_CACHE = _AUTHOR_INDEX_CACHE or {}
    return _AUTHOR_INDEX_CACHE or {}

def _extract_author_mentions(query: str) -> list[str]:
    if not query:
        return []
    q = query.lower()
    terms = set()
    for keywords in (settings.domain_keywords or {}).values():
        for kw in keywords:
            kw_l = str(kw).strip().lower()
            if not kw_l or len(kw_l) < 3 or kw_l in _AUTHOR_STOPWORDS:
                continue
            if kw_l in q:
                terms.add(kw_l)
    for token in re.findall(r"[A-Za-z][A-Za-z\\-]+", query):
        if token[0].isupper():
            t = token.lower()
            if len(t) >= 3 and t not in _AUTHOR_STOPWORDS:
                terms.add(t)
    return sorted(terms)

def _extract_query_terms(query: str, author_terms: list[str]) -> list[str]:
    if not query:
        return []
    terms = []
    author_set = {t.lower() for t in author_terms}
    for token in re.findall(r"[A-Za-z][A-Za-z\\-]+", query):
        t = token.lower()
        if len(t) < 3:
            continue
        if t in _QUERY_STOPWORDS or t in author_set:
            continue
        terms.append(t)
    # expand with simple synonyms
    expanded = []
    synonyms = _merged_term_synonyms()
    for t in terms:
        expanded.append(t)
        if t in synonyms:
            expanded.extend(synonyms[t])
    # keep stable order, remove dups
    seen = set()
    out = []
    for t in expanded:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out

def _apply_author_bias(results, query: str, bias: float = 0.8):
    terms = _extract_author_mentions(query)
    if not terms:
        return results, terms
    out = []
    for r in results:
        source = (r.get("source") or "").lower()
        text = (r.get("text") or "").lower()
        file_hit = any(t in source for t in terms)
        text_hit = any(t in text for t in terms)
        match = file_hit or text_hit
        r2 = dict(r)
        if file_hit:
            r2["author_bias"] = bias
        elif text_hit:
            r2["author_bias"] = bias * 0.5
        else:
            r2["author_bias"] = 0.0
        out.append(r2)
    return out, terms

def _author_matches(result: dict, author_terms: list[str]) -> bool:
    if not author_terms:
        return False
    source = (result.get("source") or "").lower()
    text = (result.get("text") or "").lower()
    author_index = _load_author_index()
    if author_index:
        for t in author_terms:
            sources = author_index.get(t)
            if sources and source in sources:
                return True
    for t in author_terms:
        if t in source or t in text:
            return True
    return False

@router.post("/ingest")
async def ingest_endpoint(request: Request):
    # NOTE: uses folder path inside container
    tenant = None
    try:
        body = await request.json()
        if isinstance(body, dict):
            tenant = body.get("tenant")
    except Exception:
        tenant = None
    if settings.tenant_isolation:
        tenant = tenant or "default"
    res = ingest_folder("/data/docs", tenant=tenant)
    return {"status": "ok", **res}

@router.post("/query")
async def query_endpoint(request: Request):
    body = await request.json()
    if not body or "query" not in body or "user_id" not in body:
        raise HTTPException(status_code=422, detail="body must contain 'user_id' and 'query'")
    user_id = body["user_id"]
    query = body["query"]
    plan = await plan_query(query)
    constraints = plan.get("constraints", {}) if isinstance(plan, dict) else {}
    prefs = body.get("preferences") if isinstance(body, dict) else {}
    freshness_days = None
    if isinstance(prefs, dict):
        freshness_days = prefs.get("freshness_days")
    if freshness_days is None and isinstance(constraints, dict):
        freshness_days = constraints.get("freshness_days") or constraints.get("freshness")
    if freshness_days is None:
        freshness_days = settings.default_freshness_days
    blocklist = []
    allowlist = []
    source_types_allow = []
    source_types_block = []
    domains_allow = []
    domains_block = []
    blocklist += _parse_blocklist(settings.policy_blocklist)
    allowlist += _parse_blocklist(settings.policy_allowlist)
    source_types_allow += _parse_blocklist(settings.policy_source_types_allow)
    source_types_block += _parse_blocklist(settings.policy_source_types_block)
    domains_allow += _parse_blocklist(settings.policy_domains_allow)
    domains_block += _parse_blocklist(settings.policy_domains_block)
    if isinstance(constraints, dict):
        blocklist += _parse_blocklist(constraints.get("blocklist"))
        allowlist += _parse_blocklist(constraints.get("allowlist"))
        source_types_allow += _parse_blocklist(constraints.get("source_types_allow"))
        source_types_block += _parse_blocklist(constraints.get("source_types_block"))
        domains_allow += _parse_blocklist(constraints.get("domains_allow"))
        domains_block += _parse_blocklist(constraints.get("domains_block"))
    if isinstance(prefs, dict):
        blocklist += _parse_blocklist(prefs.get("blocklist"))
        allowlist += _parse_blocklist(prefs.get("allowlist"))
        source_types_allow += _parse_blocklist(prefs.get("source_types_allow"))
        source_types_block += _parse_blocklist(prefs.get("source_types_block"))
        domains_allow += _parse_blocklist(prefs.get("domains_allow"))
        domains_block += _parse_blocklist(prefs.get("domains_block"))

    domain = route_domain(query, constraints=constraints, preferences=prefs)
    domain_source = "unknown"
    if isinstance(constraints, dict) and constraints.get("domain"):
        domain_source = "constraint"
    elif isinstance(prefs, dict) and prefs.get("domain"):
        domain_source = "preference"
    elif domain:
        domain_source = "keyword"
    fallback_domains = settings.domain_fallbacks or []
    if domain_source != "keyword":
        # When domain isn't inferred, search base + fallback domains for coverage.
        fallback_domains = list({d for d in fallback_domains if d})
    queries = plan.get("queries", [query])
    tenant = None
    if settings.tenant_isolation:
        tenant = body.get("tenant") if isinstance(body, dict) else None
        tenant = tenant or user_id
    raw_results = await run_agents(queries, domain=domain, fallback_domains=fallback_domains, tenant=tenant)

    # Normalize results
    normalized = []
    for r in raw_results:
        payload = r.get("payload") or {}
        source = r.get("source") or {}
        text = payload.get("text") or source.get("text") or ""
        if not text:
            continue
        normalized.append({
            "id": r.get("id"),
            "text": text,
            "source": payload.get("source") or source.get("source"),
            "timestamp": payload.get("timestamp") or source.get("timestamp"),
            "offset_start": payload.get("offset_start") or source.get("offset_start"),
            "offset_end": payload.get("offset_end") or source.get("offset_end"),
            "source_type": (payload.get("source_type") or source.get("source_type") or "").lower(),
            "domain": (payload.get("domain") or source.get("domain") or "").lower(),
            "score": r.get("score"),
            "agent": r.get("agent"),
            "elapsed_ms": r.get("elapsed_ms"),
        })

    normalized = apply_policy_filter(normalized, blocklist=blocklist)
    normalized = apply_policy_rules(
        normalized,
        allowlist=allowlist,
        source_types_allow=source_types_allow,
        source_types_block=source_types_block,
        domains_allow=domains_allow,
        domains_block=domains_block,
        rules=settings.policy_rules,
    )
    normalized = apply_freshness_filter(normalized, freshness_days=freshness_days)

    # Dedupe by normalized text
    seen = set()
    deduped = []
    for r in normalized:
        fp = " ".join(r["text"].lower().split())
        if fp in seen:
            continue
        seen.add(fp)
        deduped.append(r)

    deduped, author_terms = _apply_author_bias(deduped, query)
    if author_terms:
        query_terms = _extract_query_terms(query, author_terms)
        author_queries = [query]
        if query_terms:
            for t in author_terms:
                author_queries.append(f"{t} {' '.join(query_terms)}")
        author_hits = []
        for q in author_queries:
            author_hits += await author_lexical_search(
                q,
                author_terms,
                required_terms=query_terms,
                k=24,
                domain=domain,
                tenant=tenant,
            )
        for r in author_hits:
            payload = r.get("payload") or {}
            source = r.get("source") or {}
            text = payload.get("text") or source.get("text") or ""
            if not text:
                continue
            deduped.append({
                "id": r.get("id"),
                "text": text,
                "source": payload.get("source") or source.get("source"),
                "timestamp": payload.get("timestamp") or source.get("timestamp"),
                "offset_start": payload.get("offset_start") or source.get("offset_start"),
                "offset_end": payload.get("offset_end") or source.get("offset_end"),
                "source_type": (payload.get("source_type") or source.get("source_type") or "").lower(),
                "domain": (payload.get("domain") or source.get("domain") or "").lower(),
                "score": r.get("score"),
                "agent": r.get("agent"),
                "elapsed_ms": r.get("elapsed_ms"),
                "author_bias": 0.8,
            })
        # re-dedupe after author-guided expansion
        seen = set()
        tmp = []
        for r in deduped:
            fp = " ".join(r["text"].lower().split())
            if fp in seen:
                continue
            seen.add(fp)
            tmp.append(r)
        deduped = tmp
    reranked = await rerank(query, deduped)
    author_gap = False
    if author_terms:
        record_author_query(author_terms)
        for r in reranked:
            base = r.get("rerank_score")
            if base is None:
                base = r.get("score") or 0.0
            r["rerank_score"] = float(base) + float(r.get("author_bias") or 0.0)
        reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        query_terms = _extract_query_terms(query, author_terms)
        author_hits = [r for r in reranked if _author_matches(r, author_terms)]
        if author_hits:
            # Soft-hard: prefer author-only set when author is explicit.
            if query_terms:
                filtered = []
                for r in author_hits:
                    text_l = (r.get("text") or "").lower()
                    if any(t in text_l for t in query_terms):
                        filtered.append(r)
                if filtered:
                    author_hits = filtered
                    reranked = author_hits
                else:
                    # No author passages mention the query terms; allow fallback.
                    author_gap = True
                    record_author_gap(author_terms)
                    # Remove author bias so non-author relevance can surface.
                    for r in reranked:
                        rb = float(r.get("rerank_score") or 0.0)
                        rb -= float(r.get("author_bias") or 0.0)
                        r["rerank_score"] = rb
                    reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            else:
                reranked = author_hits

    graph_signals = []
    graph_subgraph = None
    graph_reasoning = None
    if settings.graph_enabled:
        try:
            ctx = build_graph_context(reranked)
            graph_signals = ctx.get("graph_signals", [])
            graph_subgraph = ctx.get("graph_subgraph")
            graph_reasoning = ctx.get("graph_reasoning")
        except Exception:
            graph_signals = []
            graph_subgraph = None
            graph_reasoning = None

    if settings.graph_enabled:
        pass

    judge_output = {}
    if settings.enable_judge:
        judge_output = await judge_evidence(query, reranked, graph_signals, graph_subgraph, graph_reasoning)
        if not isinstance(judge_output, dict):
            judge_output = {}
    judge_output["graph_reasoning"] = graph_reasoning

    synthesis = {"answer": "", "provenance": [], "confidence": judge_output.get("confidence", 0.3), "explain_trace": "synthesis_disabled"}
    if settings.enable_synthesis:
        synthesis = await synthesize_answer(query, reranked, judge_output, author_terms, author_gap)
        if not synthesis.get("answer") and synthesis.get("explain_trace") == "synthesis_unavailable":
            synthesis["answer"] = "I could not synthesize a confident answer from the available evidence."

    if author_gap and author_terms:
        non_author = [r for r in reranked if not _author_matches(r, author_terms)]
        if non_author:
            reranked = non_author

    retrieval_failures = []
    if not reranked:
        retrieval_failures.append("no_results")
    if author_gap:
        retrieval_failures.append("author_gap")
    if len(reranked) < settings.min_results_count:
        retrieval_failures.append("low_result_count")
    if reranked:
        top_score = reranked[0].get("rerank_score")
        if top_score is None:
            top_score = reranked[0].get("score")
        try:
            if top_score is not None and float(top_score) < settings.min_top_rerank_score:
                retrieval_failures.append("low_top_score")
        except Exception:
            pass
    for tag in retrieval_failures:
        record_retrieval_failure(tag)
    confidence = synthesis.get("confidence", 0.3) if isinstance(synthesis, dict) else 0.3
    try:
        hallucination_risk = 1.0 - float(confidence)
    except Exception:
        hallucination_risk = 0.7
    if not synthesis.get("provenance"):
        hallucination_risk += 0.2
    if "low_top_score" in retrieval_failures:
        hallucination_risk += 0.1
    if author_gap:
        hallucination_risk += 0.1
    hallucination_risk = max(0.0, min(1.0, hallucination_risk))
    record_hallucination_risk(hallucination_risk)

    response = {
        "user_id": user_id,
        "query": query,
        "domain": domain,
        "domain_source": domain_source,
        "author_terms": author_terms,
        "author_gap": author_gap,
        "retrieval_failures": retrieval_failures,
        "hallucination_risk": hallucination_risk,
        "intent": plan.get("intent"),
        "plan": plan,
        "results": reranked,
        "graph_signals": graph_signals,
        "graph_subgraph": graph_subgraph,
        "graph_reasoning": graph_reasoning,
        "judge": judge_output,
        **synthesis,
    }
    log_query_result(
        user_id=user_id,
        query=query,
        intent=plan.get("intent"),
        answer=response.get("answer", ""),
        provenance=response.get("provenance", []),
        confidence=response.get("confidence"),
        domain=domain or "domain_unknown",
        domain_source=domain_source,
    )
    return response

@router.get("/audit")
def audit_endpoint(user_id: str | None = None, limit: int = 50, cursor: str | None = None):
    limit = max(1, min(limit, 1000))
    results = fetch_audit_logs(limit=limit, user_id=user_id, cursor=cursor)
    next_token = results[-1]["cursor"] if results else None
    return {"results": results, "next_token": next_token}

@router.post("/audit/export")
async def audit_export_endpoint(request: Request):
    body = await request.json()
    path = body.get("path", "/data/audit/export.jsonl") if isinstance(body, dict) else "/data/audit/export.jsonl"
    limit = body.get("limit", 1000) if isinstance(body, dict) else 1000
    user_id = body.get("user_id") if isinstance(body, dict) else None
    return export_audit_jsonl(path=path, limit=limit, user_id=user_id)

@router.post("/feedback")
async def feedback_endpoint(request: Request):
    body = await request.json()
    if not isinstance(body, dict) or "user_id" not in body or "query" not in body or "rating" not in body:
        raise HTTPException(status_code=422, detail="body must contain 'user_id', 'query', 'rating'")
    user_id = body.get("user_id")
    query = body.get("query")
    rating = body.get("rating")
    comment = body.get("comment", "")
    log_feedback(user_id=user_id, query=query, rating=rating, comment=comment)
    return {"status": "ok"}

@router.get("/feedback/list")
def feedback_list_endpoint(limit: int = 50):
    limit = max(1, min(limit, 1000))
    return {"results": fetch_feedback(limit=limit)}

@router.post("/learning/export")
async def learning_export_endpoint(request: Request):
    body = await request.json()
    if not isinstance(body, dict):
        body = {}
    path = body.get("path")
    limit = body.get("limit", 1000)
    min_rating = body.get("min_rating")
    if path:
        return export_training_data(path=path, limit=limit, min_rating=min_rating)
    return export_default_training_data()

@router.post("/graph/summary")
async def graph_summary_endpoint(request: Request):
    body = await request.json()
    if not isinstance(body, dict) or "chunk_ids" not in body:
        raise HTTPException(status_code=422, detail="body must contain 'chunk_ids'")
    chunk_ids = body.get("chunk_ids") or []
    if not isinstance(chunk_ids, list):
        raise HTTPException(status_code=422, detail="'chunk_ids' must be a list")
    return {"graph_reasoning": build_graph_reasoning(chunk_ids)}

@router.get("/metrics")
def metrics_endpoint():
    return Response(content=render_prometheus(), media_type="text/plain")
