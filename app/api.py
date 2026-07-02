# app/api.py
from fastapi import APIRouter, Request, HTTPException, Response
from fastapi.responses import StreamingResponse
import asyncio
import hashlib
import json
import os
import re
import redis_client
import domain_packs
import ingest_jobs
from ingestion import ingest_folder, delete_source
from speculative import plan_query
from agents import run_agents, route_domain, apply_policy_filter, apply_freshness_filter, apply_policy_rules, author_lexical_search
from reranker import rerank
from judge import judge_evidence, build_graph_context, build_graph_reasoning
from synthesis import synthesize_answer, synthesize_answer_stream, build_stream_provenance
from config import settings
from store import log_query_result, fetch_audit_logs, export_audit_jsonl, log_feedback, fetch_feedback
from metrics import render_prometheus, record_author_gap, record_author_query, record_retrieval_failure, record_hallucination_risk, record_evidence_coverage
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

def _policy_signature() -> str:
    """Stable signature of the config-level policy so cached answers are
    invalidated when policy settings change."""
    parts = [
        settings.policy_blocklist,
        settings.policy_allowlist,
        settings.policy_source_types_allow,
        settings.policy_source_types_block,
        settings.policy_domains_allow,
        settings.policy_domains_block,
        settings.policy_rules,
        settings.default_freshness_days,
    ]
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]

def _query_cache_key(query, prefs, tenant) -> str:
    """Cache key over the inputs that determine a query's answer: normalized
    query text, per-request preferences, tenant, and the policy signature."""
    payload = {
        "q": " ".join((query or "").lower().split()),
        "prefs": prefs or {},
        "tenant": tenant,
        "policy": _policy_signature(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return f"sagrag:query:{digest}"

def _resolve_tenant(request, body):
    """Tenant for a request. When the request is authenticated, the key's
    tenant (set on request.state by the auth middleware) is authoritative and
    any body-supplied tenant is ignored — this is the server-side isolation
    boundary. Otherwise fall back to the body tenant."""
    auth_tenant = getattr(getattr(request, "state", None), "tenant", None)
    if auth_tenant is not None:
        return auth_tenant
    if isinstance(body, dict):
        return body.get("tenant")
    return None

def _merged_term_synonyms() -> dict[str, list[str]]:
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
    stopwords = domain_packs.author_stopwords()
    # Primary: explicitly configured authors (domain packs + author index) and
    # domain keywords present in the query.
    configured = set()
    known = domain_packs.authors() | set(_load_author_index().keys())
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
    # configured — this is a heuristic, not the primary signal.
    fallback = set()
    for token in re.findall(r"[A-Za-z][A-Za-z\-]+", query):
        if token[0].isupper():
            t = token.lower()
            if len(t) >= 3 and t not in stopwords:
                fallback.add(t)
    return sorted(fallback)

def _extract_query_terms(query: str, author_terms: list[str]) -> list[str]:
    if not query:
        return []
    terms = []
    stopwords = domain_packs.query_stopwords()
    author_set = {t.lower() for t in author_terms}
    for token in re.findall(r"[A-Za-z][A-Za-z\-]+", query):
        t = token.lower()
        if len(t) < 3:
            continue
        if t in stopwords or t in author_set:
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
    # NOTE: uses folder path inside container.
    # Ingestion is heavy; run it as a background job (off the event loop) and
    # return a job id immediately. Poll GET /ingest/{job_id} for status.
    tenant = None
    try:
        body = await request.json()
    except Exception:
        body = None
    if settings.tenant_isolation:
        tenant = _resolve_tenant(request, body) or "default"
    job_id = ingest_jobs.create_job("ingest", meta={"tenant": tenant, "folder": "/data/docs"})
    # Best-effort enqueue for observability / a future distributed worker.
    await redis_client.enqueue_json(
        settings.ingest_queue_name, {"job_id": job_id, "tenant": tenant, "folder": "/data/docs"}
    )
    asyncio.create_task(ingest_jobs.run_job(job_id, ingest_folder, "/data/docs", tenant))
    return {"status": "accepted", "job_id": job_id}

@router.get("/ingest/{job_id}")
async def ingest_status_endpoint(job_id: str):
    job = ingest_jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job

@router.delete("/ingest/source/{source}")
async def delete_source_endpoint(source: str, request: Request):
    tenant = None
    try:
        body = await request.json()
    except Exception:
        body = None
    if settings.tenant_isolation:
        tenant = _resolve_tenant(request, body) or "default"
    res = await asyncio.to_thread(delete_source, source, tenant)
    return {"status": "ok", **res}

async def _retrieve_and_judge(query, prefs, tenant):
    """Run plan -> retrieve -> rerank -> (graph) -> judge and return the
    pre-synthesis context shared by the buffered and streaming query
    endpoints. Behavior is identical to the original inline pipeline."""
    plan = await plan_query(query)
    constraints = plan.get("constraints", {}) if isinstance(plan, dict) else {}
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
    queries_raw = plan.get("queries", [query])
    if not isinstance(queries_raw, list):
        queries_raw = [queries_raw]
    queries = [str(q) for q in queries_raw if q is not None and str(q).strip()]
    if not queries:
        queries = [str(query)]
    raw_results, agent_diagnostics = await run_agents(queries, domain=domain, fallback_domains=fallback_domains, tenant=tenant)
    counts_raw = {}
    for r in raw_results:
        agent = r.get("agent") or "unknown"
        counts_raw[agent] = counts_raw.get(agent, 0) + 1

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
    counts_post_policy = {}
    for r in normalized:
        agent = r.get("agent") or "unknown"
        counts_post_policy[agent] = counts_post_policy.get(agent, 0) + 1
    distinct_domains = sorted({r.get("domain") for r in normalized if r.get("domain")})

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
    author_search_attempted = False
    author_diag = {"attempted": 0, "ok": 0, "zero_hits": 0, "timeout": 0, "error": 0}
    author_raw_count = 0
    if author_terms:
        author_search_attempted = True
        query_terms = _extract_query_terms(query, author_terms)
        author_queries = [query]
        if query_terms:
            for t in author_terms:
                author_queries.append(f"{t} {' '.join(query_terms)}")
        author_hits = []
        for q in author_queries:
            hits, status = await author_lexical_search(
                q,
                author_terms,
                required_terms=query_terms,
                k=24,
                domain=domain,
                tenant=tenant,
                return_status=True,
            )
            author_diag["attempted"] += 1
            author_diag[status] = author_diag.get(status, 0) + 1
            author_raw_count += len(hits)
            author_hits += hits
        if author_raw_count:
            counts_raw["lexical_author"] = counts_raw.get("lexical_author", 0) + author_raw_count
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
    return {
        "plan": plan,
        "domain": domain,
        "domain_source": domain_source,
        "raw_results": raw_results,
        "normalized": normalized,
        "counts_raw": counts_raw,
        "counts_post_policy": counts_post_policy,
        "distinct_domains": distinct_domains,
        "agent_diagnostics": agent_diagnostics,
        "author_terms": author_terms,
        "author_search_attempted": author_search_attempted,
        "author_diag": author_diag,
        "reranked": reranked,
        "author_gap": author_gap,
        "graph_signals": graph_signals,
        "graph_subgraph": graph_subgraph,
        "graph_reasoning": graph_reasoning,
        "judge_output": judge_output,
    }

@router.post("/query")
async def query_endpoint(request: Request):
    body = await request.json()
    if not body or "query" not in body or "user_id" not in body:
        raise HTTPException(status_code=422, detail="body must contain 'user_id' and 'query'")
    user_id = body["user_id"]
    query = body["query"]
    prefs = body.get("preferences") if isinstance(body, dict) else {}
    if not isinstance(prefs, dict):
        prefs = {}
    tenant = None
    if settings.tenant_isolation:
        tenant = _resolve_tenant(request, body) or user_id
    # Best-effort response cache: identical query+prefs+tenant+policy short-
    # circuits the whole pipeline. Never fatal — misses/errors just recompute.
    cache_key = _query_cache_key(query, prefs, tenant) if settings.redis_cache_enabled else None
    if cache_key is not None:
        cached = await redis_client.cache_get_json(cache_key)
        if isinstance(cached, dict):
            cached["cache"] = "hit"
            # Echo the current caller (key is shared across users), and still
            # audit the hit so the trail stays complete.
            cached["user_id"] = user_id
            log_query_result(
                user_id=user_id,
                query=query,
                intent=cached.get("intent"),
                answer=cached.get("answer", ""),
                provenance=cached.get("provenance", []),
                confidence=cached.get("confidence"),
                domain=cached.get("domain") or "domain_unknown",
                domain_source=cached.get("domain_source"),
            )
            return cached
    ctx = await _retrieve_and_judge(query, prefs, tenant)
    plan = ctx["plan"]
    domain = ctx["domain"]
    domain_source = ctx["domain_source"]
    raw_results = ctx["raw_results"]
    normalized = ctx["normalized"]
    counts_raw = ctx["counts_raw"]
    counts_post_policy = ctx["counts_post_policy"]
    distinct_domains = ctx["distinct_domains"]
    agent_diagnostics = ctx["agent_diagnostics"]
    author_terms = ctx["author_terms"]
    author_search_attempted = ctx["author_search_attempted"]
    author_diag = ctx["author_diag"]
    reranked = ctx["reranked"]
    author_gap = ctx["author_gap"]
    graph_signals = ctx["graph_signals"]
    graph_subgraph = ctx["graph_subgraph"]
    graph_reasoning = ctx["graph_reasoning"]
    judge_output = ctx["judge_output"]

    synthesis = {"answer": "", "provenance": [], "confidence": judge_output.get("confidence", 0.3), "explain_trace": "synthesis_disabled"}
    if settings.enable_synthesis:
        synthesis = await synthesize_answer(query, reranked, judge_output, author_terms, author_gap)
        if not synthesis.get("answer") and synthesis.get("explain_trace") in ("synthesis_unavailable", "synthesis_timeout", "synthesis_error"):
            synthesis["answer"] = "I could not synthesize a confident answer from the available evidence."

    if author_gap and author_terms:
        non_author = [r for r in reranked if not _author_matches(r, author_terms)]
        if non_author:
            reranked = non_author

    retrieval_failures = []
    if not domain:
        retrieval_failures.append("no_domain")
    if raw_results and not normalized:
        retrieval_failures.append("policy_blocked")
    def _append_agent_failure(agent_name: str, diag: dict | None):
        d = diag or {}
        attempted = int(d.get("attempted") or 0)
        if attempted == 0:
            return
        if int(d.get("ok") or 0) > 0:
            return
        if int(d.get("timeout") or 0) > 0:
            retrieval_failures.append(f"{agent_name}_timeout")
            return
        if int(d.get("error") or 0) > 0:
            retrieval_failures.append(f"{agent_name}_error")
            return
        retrieval_failures.append(f"{agent_name}_zero_hits")

    for agent_name in ("vector", "lexical", "structured"):
        _append_agent_failure(agent_name, agent_diagnostics.get(agent_name))
    if author_search_attempted:
        _append_agent_failure("lexical_author", author_diag)
    if domain and len(distinct_domains) > 1:
        retrieval_failures.append("cross_domain_conflict")
    if not reranked:
        retrieval_failures.append("no_results")
    if author_gap:
        retrieval_failures.append("author_gap")
    explain_trace = synthesis.get("explain_trace") if isinstance(synthesis, dict) else None
    if explain_trace == "synthesis_timeout":
        retrieval_failures.append("synthesis_timeout")
    elif explain_trace in ("synthesis_error", "synthesis_unavailable"):
        retrieval_failures.append("synthesis_error")
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
        confidence = float(confidence)
    except Exception:
        confidence = 0.3
    # Confidence floor calibration for strong evidence alignment.
    if reranked and author_terms and not author_gap:
        top = reranked[0]
        top_score = top.get("rerank_score")
        if top_score is None:
            top_score = top.get("score")
        top_author_hit = _author_matches(top, author_terms)
        has_severe_failure = any(
            t in retrieval_failures
            for t in (
                "no_results",
                "vector_timeout",
                "vector_error",
                "lexical_timeout",
                "lexical_error",
                "structured_timeout",
                "structured_error",
                "synthesis_timeout",
                "synthesis_error",
            )
        )
        try:
            strong_score = (top_score is not None) and (float(top_score) >= float(settings.confidence_alignment_top_score_min))
        except Exception:
            strong_score = False
        if top_author_hit and strong_score and not has_severe_failure:
            confidence = max(confidence, float(settings.confidence_alignment_floor))
    try:
        hallucination_risk = 1.0 - float(confidence)
    except Exception:
        hallucination_risk = 0.7
    provenance = synthesis.get("provenance") if isinstance(synthesis, dict) else []
    if not provenance:
        hallucination_risk += 0.2
    if "low_top_score" in retrieval_failures:
        hallucination_risk += 0.1
    if author_gap:
        hallucination_risk += 0.1
    hallucination_risk = max(0.0, min(1.0, hallucination_risk))
    denom = max(1, min(settings.max_evidence_snippets, len(reranked)))
    evidence_coverage = min(1.0, float(len(provenance)) / float(denom))
    record_hallucination_risk(hallucination_risk)
    record_evidence_coverage(evidence_coverage)

    response = {
        "user_id": user_id,
        "query": query,
        "domain": domain,
        "domain_source": domain_source,
        "author_terms": author_terms,
        "author_gap": author_gap,
        "retrieval_stats": {
            "counts_raw": counts_raw,
            "counts_post_policy": counts_post_policy,
            "distinct_domains": distinct_domains,
            "author_search_attempted": author_search_attempted,
            "agent_diagnostics": {
                **agent_diagnostics,
                "lexical_author": author_diag if author_search_attempted else {"attempted": 0, "ok": 0, "zero_hits": 0, "timeout": 0, "error": 0},
            },
        },
        "retrieval_failures": retrieval_failures,
        "hallucination_risk": hallucination_risk,
        "evidence_coverage": evidence_coverage,
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
    if cache_key is not None:
        response["cache"] = "miss"
        await redis_client.cache_set_json(cache_key, response, settings.query_cache_ttl_s)
    return response

def _sse(event: str, data) -> str:
    """Format a Server-Sent Events message."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

@router.post("/query/stream")
async def query_stream_endpoint(request: Request):
    """Streaming variant of /query. Runs the same plan->retrieve->rerank->judge
    pipeline, then streams the synthesis answer as SSE token events followed by
    a final event carrying provenance/confidence/trace. The buffered /query
    endpoint remains the stable, cache-backed contract."""
    body = await request.json()
    if not body or "query" not in body or "user_id" not in body:
        raise HTTPException(status_code=422, detail="body must contain 'user_id' and 'query'")
    user_id = body["user_id"]
    query = body["query"]
    prefs = body.get("preferences") if isinstance(body, dict) else {}
    if not isinstance(prefs, dict):
        prefs = {}
    tenant = None
    if settings.tenant_isolation:
        tenant = _resolve_tenant(request, body) or user_id

    ctx = await _retrieve_and_judge(query, prefs, tenant)
    reranked = ctx["reranked"]
    judge_output = ctx["judge_output"]
    author_terms = ctx["author_terms"]
    author_gap = ctx["author_gap"]
    domain = ctx["domain"]
    domain_source = ctx["domain_source"]
    intent = ctx["plan"].get("intent") if isinstance(ctx["plan"], dict) else None
    confidence = judge_output.get("confidence", 0.3) if isinstance(judge_output, dict) else 0.3

    async def event_stream():
        yield _sse("meta", {
            "domain": domain,
            "intent": intent,
            "author_terms": author_terms,
            "author_gap": author_gap,
        })
        answer_parts = []
        if settings.enable_synthesis:
            async for delta in synthesize_answer_stream(query, reranked, judge_output, author_terms, author_gap):
                answer_parts.append(delta)
                yield _sse("token", {"text": delta})
        answer = "".join(answer_parts).strip()
        provenance = build_stream_provenance(reranked, judge_output, author_terms, author_gap)
        # Audit the streamed answer just like the buffered endpoint.
        log_query_result(
            user_id=user_id,
            query=query,
            intent=intent,
            answer=answer,
            provenance=provenance,
            confidence=confidence,
            domain=domain or "domain_unknown",
            domain_source=domain_source,
        )
        yield _sse("final", {
            "answer": answer,
            "provenance": provenance,
            "confidence": confidence,
            "explain_trace": "synthesis_stream",
        })
        yield _sse("done", {})

    return StreamingResponse(event_stream(), media_type="text/event-stream")

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
