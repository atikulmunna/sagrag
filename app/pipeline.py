# app/pipeline.py
"""The query pipeline: plan -> retrieve -> rerank -> graph -> judge -> synthesize.

Each stage is a small async function over a shared ``QueryContext``. The
buffered /query endpoint and the SSE /query/stream endpoint run the same
stages (the stream reports progress between them) and finish through the same
``finalize`` step, so both get identical failure tags, confidence calibration,
metrics, and response shape.
"""
import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field

import agents
import authors
import namespaces
import verification
from config import settings
from judge import build_graph_context, judge_evidence
from metrics import (
    record_author_gap,
    record_author_query,
    record_claims,
    record_evidence_coverage,
    record_hallucination_risk,
    record_retrieval_failure,
)
from reranker import rerank
from speculative import plan_query
from store import log_query_result
from synthesis import build_stream_provenance, normalize_dashes, synthesize_answer

CACHE_PREFIX = "sagrag:query:v2:"
_NO_ANSWER = "I could not synthesize a confident answer from the available evidence."
ABSTAIN_ANSWER = (
    "The cited passages do not support a confident answer to this question. "
    "The closest passages are listed so you can read them directly."
)
_SEVERE_FAILURES = frozenset({
    "no_results",
    "vector_timeout", "vector_error",
    "lexical_timeout", "lexical_error",
    "structured_timeout", "structured_error",
    "synthesis_timeout", "synthesis_error",
})
_POLICY_FIELDS = (
    "blocklist", "allowlist", "source_types_allow", "source_types_block", "domains_allow", "domains_block",
)


class UnknownDomain(ValueError):
    """A requested domain that is invalid or not available to the caller."""


def _parse_list(value):
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip().lower() for v in value.split(",") if v.strip()]
    return []


# --- cache -------------------------------------------------------------------


def policy_signature() -> str:
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


def query_cache_key(query, prefs, tenant) -> str:
    """Cache key over the inputs that determine a query's answer: normalized
    query text, per-request preferences, tenant, and the policy signature."""
    payload = {
        "q": " ".join((query or "").lower().split()),
        "prefs": prefs or {},
        "tenant": tenant,
        "policy": policy_signature(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return CACHE_PREFIX + digest


# --- context -----------------------------------------------------------------


@dataclass
class QueryContext:
    query: str
    prefs: dict
    tenant: str | None = None
    plan: dict = field(default_factory=dict)
    freshness_days: int | None = None
    domain: str | None = None
    domain_source: str = "unknown"
    raw_results: list = field(default_factory=list)
    normalized: list = field(default_factory=list)
    deduped: list = field(default_factory=list)
    counts_raw: dict = field(default_factory=dict)
    counts_post_policy: dict = field(default_factory=dict)
    distinct_domains: list = field(default_factory=list)
    agent_diagnostics: dict = field(default_factory=dict)
    author_terms: list = field(default_factory=list)
    author_search_attempted: bool = False
    author_diag: dict = field(default_factory=lambda: {"attempted": 0, "ok": 0, "zero_hits": 0, "timeout": 0, "error": 0})
    reranked: list = field(default_factory=list)
    author_gap: bool = False
    graph_signals: list = field(default_factory=list)
    graph_subgraph: dict | None = None
    graph_reasoning: dict | None = None
    judge_output: dict = field(default_factory=dict)
    verification: dict | None = None
    timings_ms: dict = field(default_factory=dict)


def validated_preferences(prefs: dict, tenant: str | None) -> dict:
    """Normalize preferences and check a requested domain is one the caller
    may search. Raises UnknownDomain otherwise."""
    prefs = {k: v for k, v in (prefs or {}).items() if v not in (None, [], "")}
    if prefs.get("domain"):
        try:
            domain = namespaces.validate_domain(prefs["domain"])
        except namespaces.InvalidName as e:
            raise UnknownDomain(str(e)) from e
        if domain not in agents.known_domains(tenant):
            raise UnknownDomain(f"unknown domain: {domain!r}")
        prefs["domain"] = domain
    return prefs


def _normalize_hit(r: dict, author_bias: float | None = None) -> dict | None:
    payload = r.get("payload") or {}
    source = r.get("source") or {}
    text = payload.get("text") or source.get("text") or ""
    if not text:
        return None

    def pick(key):
        v = payload.get(key)
        return v if v is not None else source.get(key)

    item = {
        "id": r.get("id"),
        "text": text,
        "source": pick("source"),
        "timestamp": pick("timestamp"),
        "offset_start": pick("offset_start"),
        "offset_end": pick("offset_end"),
        "source_type": (pick("source_type") or "").lower(),
        "domain": (pick("domain") or "").lower(),
        "score": r.get("score"),
        "agent": r.get("agent"),
        "elapsed_ms": r.get("elapsed_ms"),
    }
    if author_bias is not None:
        item["author_bias"] = author_bias
    return item


def _dedupe_by_text(items):
    seen = set()
    out = []
    for r in items:
        fp = " ".join(r["text"].lower().split())
        if fp in seen:
            continue
        seen.add(fp)
        out.append(r)
    return out


def _count_by_agent(items):
    counts: dict = {}
    for r in items:
        agent = r.get("agent") or "unknown"
        counts[agent] = counts.get(agent, 0) + 1
    return counts


def _score(r):
    s = r.get("rerank_score")
    return r.get("score") if s is None else s


# --- stages ------------------------------------------------------------------


async def plan_stage(ctx: QueryContext):
    ctx.plan = await plan_query(ctx.query)
    constraints = ctx.plan.get("constraints") or {}
    freshness = ctx.prefs.get("freshness_days")
    if freshness is None:
        freshness = constraints.get("freshness_days")
    if freshness is None:
        freshness = settings.default_freshness_days
    ctx.freshness_days = freshness


async def retrieve_stage(ctx: QueryContext):
    prefs = ctx.prefs
    ctx.domain = agents.route_domain(ctx.query, preferences=prefs, tenant=ctx.tenant)
    if prefs.get("domain"):
        ctx.domain_source = "preference"
    elif ctx.domain:
        ctx.domain_source = "keyword"
    fallback_domains = []
    for d in settings.domain_fallbacks or []:
        slug = namespaces.slugify(d)
        if slug and slug not in fallback_domains:
            fallback_domains.append(slug)
    queries = ctx.plan.get("queries") or [ctx.query]
    ctx.raw_results, ctx.agent_diagnostics = await agents.run_agents(
        queries, domain=ctx.domain, fallback_domains=fallback_domains, tenant=ctx.tenant
    )
    ctx.counts_raw = _count_by_agent(ctx.raw_results)

    policy = {name: _parse_list(getattr(settings, f"policy_{name}")) + _parse_list(prefs.get(name)) for name in _POLICY_FIELDS}
    normalized = [n for n in (_normalize_hit(r) for r in ctx.raw_results) if n]
    normalized = agents.apply_policy_filter(normalized, blocklist=policy["blocklist"])
    normalized = agents.apply_policy_rules(
        normalized,
        allowlist=policy["allowlist"],
        source_types_allow=policy["source_types_allow"],
        source_types_block=policy["source_types_block"],
        domains_allow=policy["domains_allow"],
        domains_block=policy["domains_block"],
        rules=settings.policy_rules,
    )
    ctx.normalized = agents.apply_freshness_filter(normalized, freshness_days=ctx.freshness_days)
    ctx.counts_post_policy = _count_by_agent(ctx.normalized)
    ctx.distinct_domains = sorted({r.get("domain") for r in ctx.normalized if r.get("domain")})

    deduped, ctx.author_terms = authors.apply_author_bias(_dedupe_by_text(ctx.normalized), ctx.query)
    if ctx.author_terms:
        ctx.author_search_attempted = True
        query_terms = authors.extract_query_terms(ctx.query, ctx.author_terms)
        author_queries = [ctx.query]
        if query_terms:
            author_queries += [f"{t} {' '.join(query_terms)}" for t in ctx.author_terms]
        author_raw_count = 0
        for q in author_queries:
            hits, status = await agents.author_lexical_search(
                q, ctx.author_terms, required_terms=query_terms, k=24,
                domain=ctx.domain, tenant=ctx.tenant, return_status=True,
            )
            ctx.author_diag["attempted"] += 1
            ctx.author_diag[status] = ctx.author_diag.get(status, 0) + 1
            author_raw_count += len(hits)
            deduped += [n for n in (_normalize_hit(h, author_bias=0.8) for h in hits) if n]
        if author_raw_count:
            ctx.counts_raw["lexical_author"] = ctx.counts_raw.get("lexical_author", 0) + author_raw_count
        # re-dedupe after author-guided expansion
        deduped = _dedupe_by_text(deduped)
    ctx.deduped = deduped


def fuse_candidates(items: list[dict], limit: int, k: int = 60) -> list[dict]:
    """Pick at most ``limit`` candidates by reciprocal rank fusion.

    Raw scores are not comparable across retrievers (BM25 vs cosine), but ranks
    are: each item scores 1 / (k + its rank within its own retriever), so every
    retriever's best hits make the cut before anyone's long tail.
    """
    if limit <= 0 or len(items) <= limit:
        return items
    by_agent: dict = {}
    for i, item in enumerate(items):
        by_agent.setdefault(item.get("agent") or "unknown", []).append(i)
    fused = {}
    for indices in by_agent.values():
        ranked = sorted(indices, key=lambda i: float(items[i].get("score") or 0.0), reverse=True)
        for rank, i in enumerate(ranked, start=1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank)
    keep = sorted(sorted(fused, key=lambda i: fused[i], reverse=True)[:limit])
    return [items[i] for i in keep]


async def rerank_stage(ctx: QueryContext):
    reranked = await rerank(ctx.query, fuse_candidates(ctx.deduped, settings.rerank_max_candidates))
    if ctx.author_terms:
        record_author_query(ctx.author_terms)
        for r in reranked:
            r["rerank_score"] = float(_score(r) or 0.0) + float(r.get("author_bias") or 0.0)
        reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        query_terms = authors.extract_query_terms(ctx.query, ctx.author_terms)
        author_hits = [r for r in reranked if authors.author_matches(r, ctx.author_terms)]
        if author_hits:
            # Soft-hard: prefer the author-only set when an author is explicit.
            if not query_terms:
                reranked = author_hits
            else:
                filtered = [r for r in author_hits if any(t in (r.get("text") or "").lower() for t in query_terms)]
                if filtered:
                    reranked = filtered
                else:
                    # No author passages mention the query terms; allow fallback
                    # and remove the author bias so non-author relevance surfaces.
                    ctx.author_gap = True
                    record_author_gap(ctx.author_terms)
                    for r in reranked:
                        r["rerank_score"] = float(r.get("rerank_score") or 0.0) - float(r.get("author_bias") or 0.0)
                    reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    ctx.reranked = reranked


async def graph_stage(ctx: QueryContext):
    if not settings.graph_enabled:
        return
    try:
        # spaCy + Neo4j are blocking; keep them off the event loop.
        graph = await asyncio.to_thread(build_graph_context, ctx.reranked, ctx.tenant)
    except Exception:
        return
    ctx.graph_signals = graph.get("graph_signals") or []
    ctx.graph_subgraph = graph.get("graph_subgraph")
    ctx.graph_reasoning = graph.get("graph_reasoning")


async def judge_stage(ctx: QueryContext):
    judge_output = {}
    if settings.enable_judge:
        judge_output = await judge_evidence(
            ctx.query, ctx.reranked, ctx.graph_signals, ctx.graph_subgraph, ctx.graph_reasoning
        )
        if not isinstance(judge_output, dict):
            judge_output = {}
    judge_output["graph_reasoning"] = ctx.graph_reasoning
    ctx.judge_output = judge_output


STAGES = (
    ("plan", plan_stage),
    ("retrieve", retrieve_stage),
    ("rerank", rerank_stage),
    ("graph", graph_stage),
    ("judge", judge_stage),
)


async def iter_stages(ctx: QueryContext):
    """Run every pre-synthesis stage, yielding progress events between them."""
    for name, stage in STAGES:
        if name == "graph" and not settings.graph_enabled:
            continue
        yield {"stage": name, "state": "running"}
        started = time.monotonic()
        await stage(ctx)
        ctx.timings_ms[name] = int((time.monotonic() - started) * 1000)
        yield {"stage": name, "state": "done", "ms": ctx.timings_ms[name]}


async def run_stages(ctx: QueryContext) -> QueryContext:
    async for _ in iter_stages(ctx):
        pass
    return ctx


def judge_confidence(ctx: QueryContext) -> float:
    try:
        return float(ctx.judge_output.get("confidence", 0.3))
    except (TypeError, ValueError):
        return 0.3


async def synthesize(ctx: QueryContext) -> dict:
    """Buffered synthesis (JSON contract) with its timing recorded."""
    if not settings.enable_synthesis:
        return {"answer": "", "provenance": [], "confidence": judge_confidence(ctx), "explain_trace": "synthesis_disabled"}
    started = time.monotonic()
    synthesis = await synthesize_answer(ctx.query, ctx.reranked, ctx.judge_output, ctx.author_terms, ctx.author_gap)
    ctx.timings_ms["synthesize"] = int((time.monotonic() - started) * 1000)
    if not synthesis.get("answer") and synthesis.get("explain_trace") in ("synthesis_unavailable", "synthesis_timeout", "synthesis_error"):
        synthesis["answer"] = _NO_ANSWER
    return synthesis


async def verify_stage(ctx: QueryContext, synthesis: dict):
    """Check each answer sentence against the passages it cites (opt-in)."""
    if not settings.verify_enabled:
        return
    started = time.monotonic()
    ctx.verification = await verification.verify(
        synthesis.get("answer") or "", ctx.reranked, synthesis.get("provenance") or []
    )
    ctx.timings_ms["verify"] = int((time.monotonic() - started) * 1000)


def stream_synthesis_result(ctx: QueryContext, answer: str) -> dict:
    """The synthesis dict for a streamed answer (provenance is server-side)."""
    return {
        "answer": normalize_dashes(answer),
        "provenance": build_stream_provenance(ctx.reranked, ctx.judge_output, ctx.author_terms, ctx.author_gap),
        "confidence": judge_confidence(ctx),
        "explain_trace": "synthesis_stream",
    }


# --- finalize ----------------------------------------------------------------


def _agent_failure(agent_name: str, diag: dict | None) -> str | None:
    d = diag or {}
    if int(d.get("attempted") or 0) == 0 or int(d.get("ok") or 0) > 0:
        return None
    if int(d.get("timeout") or 0) > 0:
        return f"{agent_name}_timeout"
    if int(d.get("error") or 0) > 0:
        return f"{agent_name}_error"
    return f"{agent_name}_zero_hits"


def retrieval_failures(ctx: QueryContext, results: list, explain_trace) -> list[str]:
    failures = []
    if not ctx.domain:
        failures.append("no_domain")
    if ctx.raw_results and not ctx.normalized:
        failures.append("policy_blocked")
    for agent_name in ("vector", "lexical", "structured"):
        tag = _agent_failure(agent_name, ctx.agent_diagnostics.get(agent_name))
        if tag:
            failures.append(tag)
    if ctx.author_search_attempted:
        tag = _agent_failure("lexical_author", ctx.author_diag)
        if tag:
            failures.append(tag)
    if ctx.domain and len(ctx.distinct_domains) > 1:
        failures.append("cross_domain_conflict")
    if not results:
        failures.append("no_results")
    if ctx.author_gap:
        failures.append("author_gap")
    if explain_trace == "synthesis_timeout":
        failures.append("synthesis_timeout")
    elif explain_trace in ("synthesis_error", "synthesis_unavailable"):
        failures.append("synthesis_error")
    claims = (ctx.verification or {}).get("claims") or []
    if any(c["status"] == "unsupported" for c in claims):
        failures.append("unsupported_claims")
    if any(c["status"] == "contradicted" for c in claims):
        failures.append("contradicted_claims")
    if len(results) < settings.min_results_count:
        failures.append("low_result_count")
    if results:
        top_score = _score(results[0])
        try:
            if top_score is not None and float(top_score) < settings.min_top_rerank_score:
                failures.append("low_top_score")
        except (TypeError, ValueError):
            pass
    return failures


def finalize(ctx: QueryContext, synthesis: dict, user_id: str, query_id: str) -> dict:
    """Score the answer, record metrics, and build the response body."""
    results = ctx.reranked
    if ctx.author_gap and ctx.author_terms:
        non_author = [r for r in results if not authors.author_matches(r, ctx.author_terms)]
        if non_author:
            results = non_author
    explain_trace = synthesis.get("explain_trace")
    failures = retrieval_failures(ctx, results, explain_trace)
    for tag in failures:
        record_retrieval_failure(tag)

    try:
        confidence = float(synthesis.get("confidence", 0.3))
    except (TypeError, ValueError):
        confidence = 0.3
    # Confidence floor calibration for strong evidence alignment.
    if results and ctx.author_terms and not ctx.author_gap:
        top = results[0]
        try:
            strong_score = _score(top) is not None and float(_score(top)) >= float(settings.confidence_alignment_top_score_min)
        except (TypeError, ValueError):
            strong_score = False
        if authors.author_matches(top, ctx.author_terms) and strong_score and not (_SEVERE_FAILURES & set(failures)):
            confidence = max(confidence, float(settings.confidence_alignment_floor))
    confidence = max(0.0, min(1.0, confidence))

    provenance = synthesis.get("provenance") or []
    answer = synthesis.get("answer") or ""
    verified = ctx.verification or None
    abstained = False
    if verified:
        # Measured: the share of sentences no cited passage entails.
        risk_method = "nli"
        hallucination_risk = 1.0 - float(verified["supported_ratio"])
        record_claims(verified["claims"])
        if settings.abstain_below_support > 0 and verified["supported_ratio"] < settings.abstain_below_support:
            abstained = True
            failures.append("abstained")
            record_retrieval_failure("abstained")  # the loop above ran before this tag existed
    else:
        # Heuristic fallback: self-reported confidence plus fixed penalties.
        risk_method = "heuristic"
        hallucination_risk = 1.0 - confidence
        if not provenance:
            hallucination_risk += 0.2
        if "low_top_score" in failures:
            hallucination_risk += 0.1
        if ctx.author_gap:
            hallucination_risk += 0.1
    hallucination_risk = max(0.0, min(1.0, hallucination_risk))
    denom = max(1, min(settings.max_evidence_snippets, len(results)))
    evidence_coverage = min(1.0, float(len(provenance)) / float(denom))
    record_hallucination_risk(hallucination_risk)
    record_evidence_coverage(evidence_coverage)

    empty_diag = {"attempted": 0, "ok": 0, "zero_hits": 0, "timeout": 0, "error": 0}
    return {
        "query_id": query_id,
        "user_id": user_id,
        "query": ctx.query,
        "domain": ctx.domain,
        "domain_source": ctx.domain_source,
        "author_terms": ctx.author_terms,
        "author_gap": ctx.author_gap,
        "retrieval_stats": {
            "counts_raw": ctx.counts_raw,
            "counts_post_policy": ctx.counts_post_policy,
            "distinct_domains": ctx.distinct_domains,
            "author_search_attempted": ctx.author_search_attempted,
            "agent_diagnostics": {
                **ctx.agent_diagnostics,
                "lexical_author": ctx.author_diag if ctx.author_search_attempted else empty_diag,
            },
        },
        "retrieval_failures": failures,
        "hallucination_risk": hallucination_risk,
        "evidence_coverage": evidence_coverage,
        "intent": ctx.plan.get("intent"),
        "plan": ctx.plan,
        "results": results,
        "graph_signals": ctx.graph_signals,
        "graph_subgraph": ctx.graph_subgraph,
        "graph_reasoning": ctx.graph_reasoning,
        "judge": ctx.judge_output,
        "timings_ms": dict(ctx.timings_ms),
        "answer": ABSTAIN_ANSWER if abstained else answer,
        "provenance": provenance,
        "confidence": confidence,
        "explain_trace": explain_trace,
        "hallucination_risk_method": risk_method,
        "claims": verified["claims"] if verified else None,
        "supported_ratio": verified["supported_ratio"] if verified else None,
        "abstained": abstained,
        "unverified_answer": answer if abstained else None,
    }


async def audit(response: dict, tenant: str | None):
    """Best-effort audit row for an answered query (runs off the event loop)."""
    await asyncio.to_thread(
        log_query_result,
        user_id=response.get("user_id"),
        query=response.get("query"),
        intent=response.get("intent"),
        answer=response.get("answer", ""),
        provenance=response.get("provenance", []),
        confidence=response.get("confidence"),
        domain=response.get("domain") or "domain_unknown",
        domain_source=response.get("domain_source"),
        tenant=tenant,
        query_id=response.get("query_id"),
    )
