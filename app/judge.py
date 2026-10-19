import asyncio
import logging

import refs
from config import settings
from jsonutil import as_unit_float, safe_json_extract
from llm_client import llm

_LOG = logging.getLogger(__name__)


def _finalize(judge_output: dict, graph_reasoning) -> dict:
    """Apply graph-derived contradiction/conflict penalties and relation boosts."""
    contradicted = extract_contradictions(graph_reasoning)
    if contradicted:
        judge_output["contradictions"] = contradicted
        judge_output = apply_contradiction_penalty(judge_output, len(contradicted))
    judge_output = apply_relation_conflict_penalty(judge_output, graph_reasoning)
    judge_output = apply_relation_boost(judge_output, graph_reasoning, len(contradicted))
    return judge_output


def _top_ids(evidence, n=3):
    return [e.get("id") for e in evidence[:n] if e.get("id")]


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "confidence": {"type": "number"},
        "trusted": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["confidence", "trusted", "notes"],
}


async def judge_evidence(query, evidence, graph_signals, graph_subgraph=None, graph_reasoning=None):
    labelled = refs.label(evidence, settings.max_evidence_snippets)
    prompt = f"""
You are an evidence judge for a RAG system. Decide which evidence passages
actually answer the user query and how confident an answer built on them can be.
Return JSON: {{"confidence": 0.0-1.0, "trusted": ["E1", ...], "notes": "one short sentence"}}
Refer to passages only by their labels (E1, E2, ...).

User query:
{query}

Graph signals:
{graph_signals}

Graph subgraph:
{graph_subgraph}

Graph reasoning:
{graph_reasoning}

Instruction: if graph_reasoning shows claims with contradict_count > 0, lower confidence and include a note.
Instruction: if graph_reasoning includes strong relations, use them to support or refute evidence.
Instruction: increase confidence when multiple chunks support the same relation; decrease when relations conflict.
Instruction: prefer evidence with higher evidence_scores and path counts when selecting trusted passages.

Evidence:
{refs.render(labelled, 400)}
"""
    try:
        out = await asyncio.wait_for(
            llm.completion(prompt, max_tokens=settings.judge_max_tokens, format=JUDGE_SCHEMA, temperature=0),
            timeout=settings.judge_timeout_s,
        )
        parsed = safe_json_extract(out)
    except Exception as e:
        # Fall back to trusting top evidence when judge fails (timeout/provider errors).
        _LOG.warning("judge_failed: %r", e)
        fallback = {"confidence": 0.3, "trusted_ids": _top_ids(evidence), "notes": "judge_error_fallback"}
        return _finalize(fallback, graph_reasoning)
    if not parsed:
        # Non-JSON fallback: trust top evidence ids so synthesis can proceed.
        _LOG.warning("judge_non_json")
        fallback = {"confidence": 0.4, "trusted_ids": _top_ids(evidence), "notes": "judge_non_json"}
        return _finalize(fallback, graph_reasoning)
    # Map E-labels (or ids) back to real chunk ids; never trust invented ones.
    trusted = parsed.get("trusted")
    if trusted is None:
        trusted = parsed.get("trusted_ids")
    result = {
        "confidence": as_unit_float(parsed.get("confidence"), default=0.4),
        "trusted_ids": refs.resolve(trusted, labelled, evidence),
        "notes": str(parsed.get("notes") or "")[:300],
    }
    return _finalize(result, graph_reasoning)


def build_graph_reasoning(chunk_ids, tenant=None):
    if not chunk_ids:
        return None
    try:
        from graph import graph_reasoner
        return graph_reasoner(chunk_ids, tenant=tenant)
    except Exception as e:
        _LOG.warning("graph_reasoner_failed: %s", e)
        return None


def build_graph_subgraph(chunk_ids, tenant=None):
    if not chunk_ids:
        return None
    try:
        from graph import subgraph_for_chunks
        return subgraph_for_chunks(chunk_ids, tenant=tenant)
    except Exception as e:
        _LOG.warning("graph_subgraph_failed: %s", e)
        return None


def build_graph_signals(entities, tenant=None):
    if not entities:
        return []
    try:
        from graph import support_density_for_entities
        return support_density_for_entities(list(entities), tenant=tenant)
    except Exception as e:
        _LOG.warning("graph_signals_failed: %s", e)
        return []


def extract_entities_from_results(results):
    try:
        from utils import extract_entities
    except Exception:
        return set()
    entities = set()
    for r in results:
        text = r.get("text") if isinstance(r, dict) else None
        if not text:
            continue
        for ent in extract_entities(text):
            entities.add(ent)
    return entities


def build_graph_context(results, tenant=None):
    """Synchronous (spaCy + Neo4j); callers run it in a worker thread."""
    entities = extract_entities_from_results(results)
    graph_signals = build_graph_signals(entities, tenant=tenant) if entities else []
    chunk_ids = [r.get("id") for r in results if isinstance(r, dict) and r.get("id")]
    return {
        "graph_signals": graph_signals,
        "graph_subgraph": build_graph_subgraph(chunk_ids, tenant=tenant),
        "graph_reasoning": build_graph_reasoning(chunk_ids, tenant=tenant),
    }


def extract_contradictions(graph_reasoning):
    if not isinstance(graph_reasoning, dict):
        return []
    claims = graph_reasoning.get("claims", [])
    contradicted = [c for c in claims if c.get("contradict_count", 0) > 0]
    return [{"id": c.get("id"), "contradict_count": c.get("contradict_count")} for c in contradicted]


def _add_note(judge_output: dict, note: str):
    if judge_output.get("notes"):
        judge_output["notes"] = f"{judge_output['notes']}; {note}"
    else:
        judge_output["notes"] = note


def apply_contradiction_penalty(judge_output, contradiction_count: int):
    if not isinstance(judge_output, dict) or judge_output.get("confidence") is None:
        return judge_output
    try:
        cap = settings.judge_contradiction_confidence_cap
        base = float(judge_output["confidence"])
        penalty = min(settings.judge_contradiction_penalty_max, settings.judge_contradiction_penalty_per_claim * contradiction_count)
        judge_output["confidence"] = min(base * (1.0 - penalty), cap)
        _add_note(judge_output, f"confidence adjusted for {contradiction_count} contradictions (penalty={penalty:.2f}, cap={cap:.2f})")
    except (TypeError, ValueError):
        pass
    return judge_output


def apply_relation_boost(judge_output, graph_reasoning, contradiction_count: int = 0):
    if not isinstance(judge_output, dict) or judge_output.get("confidence") is None:
        return judge_output
    if contradiction_count > 0:
        return judge_output
    strength = 0
    if isinstance(graph_reasoning, dict):
        strength = len(graph_reasoning.get("relation_strength") or [])
    if strength <= 0:
        return judge_output
    try:
        base = float(judge_output["confidence"])
        boost = min(settings.judge_relation_boost_max, settings.judge_relation_boost_per_relation * strength)
        judge_output["confidence"] = min(1.0, base + boost)
        _add_note(judge_output, f"confidence boosted by relation strength (boost={boost:.2f})")
    except (TypeError, ValueError):
        pass
    return judge_output


def apply_relation_conflict_penalty(judge_output, graph_reasoning):
    if not isinstance(judge_output, dict) or judge_output.get("confidence") is None:
        return judge_output
    conflicts = 0
    if isinstance(graph_reasoning, dict):
        conflicts = len(graph_reasoning.get("relation_conflicts") or [])
    if conflicts <= 0:
        return judge_output
    try:
        base = float(judge_output["confidence"])
        penalty = min(0.5, settings.judge_relation_conflict_penalty * conflicts)
        judge_output["confidence"] = max(0.0, base * (1.0 - penalty))
        _add_note(judge_output, f"confidence reduced for relation conflicts (penalty={penalty:.2f})")
    except (TypeError, ValueError):
        pass
    return judge_output
