# app/api.py
from fastapi import APIRouter, Request, HTTPException, Response
from ingestion import ingest_folder
from speculative import plan_query
from agents import run_agents, route_domain, apply_policy_filter, apply_freshness_filter, apply_policy_rules
from reranker import rerank
from judge import judge_evidence, build_graph_context, build_graph_reasoning
from synthesis import synthesize_answer
from config import settings
from store import log_query_result, fetch_audit_logs, export_audit_jsonl, log_feedback, fetch_feedback
from metrics import render_prometheus
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

    reranked = await rerank(query, deduped)

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
        synthesis = await synthesize_answer(query, reranked, judge_output)
        if not synthesis.get("answer"):
            synthesis["answer"] = "I could not synthesize a confident answer from the available evidence."

    response = {
        "user_id": user_id,
        "query": query,
        "domain": domain,
        "domain_source": domain_source,
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
