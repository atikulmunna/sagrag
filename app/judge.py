import asyncio
import json
import re

from llm_client import llm
from config import settings

def _safe_json_extract(text: str):
    try:
        match = re.search(r"{.*}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return None

async def judge_evidence(query, evidence, graph_signals, graph_subgraph=None, graph_reasoning=None):
    evidence_snippets = []
    for e in evidence[: settings.max_evidence_snippets]:
        snippet = e.get("text", "")[:400]
        evidence_snippets.append({"id": e.get("id"), "text": snippet, "source": e.get("source")})
    prompt = f"""
You are an evidence judge for a RAG system.
Return JSON only with keys:
{{"confidence": 0.0-1.0, "trusted_ids": [...], "notes": "..."}}

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
Instruction: prefer evidence with higher evidence_scores and path counts when selecting trusted_ids.

Evidence snippets:
{evidence_snippets}
"""
    try:
        out = await asyncio.wait_for(
            llm.completion(prompt, max_tokens=settings.judge_max_tokens),
            timeout=settings.judge_timeout_s,
        )
        parsed = _safe_json_extract(out)
        if parsed:
            # Apply structured contradiction signals when available.
            contradicted = extract_contradictions(graph_reasoning)
            if contradicted:
                parsed["contradictions"] = contradicted
                parsed = apply_contradiction_penalty(parsed, len(contradicted))
            parsed = apply_relation_conflict_penalty(parsed, graph_reasoning)
            parsed = apply_relation_boost(parsed, graph_reasoning, len(contradicted))
            return parsed
        # Non-JSON fallback: trust top evidence ids so synthesis can proceed.
        trusted_ids = [e.get("id") for e in evidence[:3] if e.get("id")]
        fallback = {"confidence": 0.4, "trusted_ids": trusted_ids, "notes": "judge_non_json"}
        contradicted = extract_contradictions(graph_reasoning)
        if contradicted:
            fallback["contradictions"] = contradicted
            fallback = apply_contradiction_penalty(fallback, len(contradicted))
        fallback = apply_relation_conflict_penalty(fallback, graph_reasoning)
        fallback = apply_relation_boost(fallback, graph_reasoning, len(contradicted))
        return fallback
    except Exception:
        # Fall back to trusting top evidence when judge fails (timeout/provider errors).
        trusted_ids = [e.get("id") for e in evidence[:3] if e.get("id")]
        fallback = {"confidence": 0.3, "trusted_ids": trusted_ids, "notes": "judge_error_fallback"}
        contradicted = extract_contradictions(graph_reasoning)
        if contradicted:
            fallback["contradictions"] = contradicted
            fallback = apply_contradiction_penalty(fallback, len(contradicted))
        fallback = apply_relation_conflict_penalty(fallback, graph_reasoning)
        fallback = apply_relation_boost(fallback, graph_reasoning, len(contradicted))
        return fallback
    fallback = {"confidence": 0.3, "trusted_ids": [], "notes": "judge_unavailable"}
    contradicted = extract_contradictions(graph_reasoning)
    if contradicted:
        fallback["contradictions"] = contradicted
        fallback = apply_contradiction_penalty(fallback, len(contradicted))
    fallback = apply_relation_conflict_penalty(fallback, graph_reasoning)
    fallback = apply_relation_boost(fallback, graph_reasoning, len(contradicted))
    return fallback

def build_graph_reasoning(chunk_ids):
    if not chunk_ids:
        return None
    try:
        from graph import graph_reasoner
        return graph_reasoner(chunk_ids)
    except Exception:
        return None

def build_graph_subgraph(chunk_ids):
    if not chunk_ids:
        return None
    try:
        from graph import subgraph_for_chunks
        return subgraph_for_chunks(chunk_ids)
    except Exception:
        return None

def build_graph_signals(entities):
    if not entities:
        return []
    try:
        from graph import support_density_for_entities
        return support_density_for_entities(list(entities))
    except Exception:
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

def build_graph_context(results):
    entities = extract_entities_from_results(results)
    graph_signals = build_graph_signals(entities) if entities else []
    chunk_ids = [r.get("id") for r in results if isinstance(r, dict) and r.get("id")]
    graph_subgraph = build_graph_subgraph(chunk_ids)
    graph_reasoning = build_graph_reasoning(chunk_ids)
    return {
        "graph_signals": graph_signals,
        "graph_subgraph": graph_subgraph,
        "graph_reasoning": graph_reasoning,
    }

def extract_contradictions(graph_reasoning):
    if not isinstance(graph_reasoning, dict):
        return []
    claims = graph_reasoning.get("claims", [])
    contradicted = [c for c in claims if c.get("contradict_count", 0) > 0]
    return [{"id": c.get("id"), "contradict_count": c.get("contradict_count")} for c in contradicted]

def apply_contradiction_penalty(judge_output, contradiction_count: int):
    if not isinstance(judge_output, dict):
        return judge_output
    if "confidence" not in judge_output or judge_output["confidence"] is None:
        return judge_output
    try:
        cap = settings.judge_contradiction_confidence_cap
        base = float(judge_output["confidence"])
        penalty = min(settings.judge_contradiction_penalty_max, settings.judge_contradiction_penalty_per_claim * contradiction_count)
        adjusted = base * (1.0 - penalty)
        judge_output["confidence"] = min(adjusted, cap)
        note = f"confidence adjusted for {contradiction_count} contradictions (penalty={penalty:.2f}, cap={cap:.2f})"
        if "notes" in judge_output and judge_output["notes"]:
            judge_output["notes"] = f"{judge_output['notes']}; {note}"
        else:
            judge_output["notes"] = note
    except Exception:
        pass
    return judge_output

def apply_relation_boost(judge_output, graph_reasoning, contradiction_count: int = 0):
    if not isinstance(judge_output, dict):
        return judge_output
    if "confidence" not in judge_output or judge_output["confidence"] is None:
        return judge_output
    try:
        if contradiction_count > 0:
            return judge_output
        strength = 0
        if isinstance(graph_reasoning, dict):
            strength = len(graph_reasoning.get("relation_strength") or [])
        if strength <= 0:
            return judge_output
        base = float(judge_output["confidence"])
        boost = min(settings.judge_relation_boost_max, settings.judge_relation_boost_per_relation * strength)
        judge_output["confidence"] = min(1.0, base + boost)
        note = f"confidence boosted by relation strength (boost={boost:.2f})"
        if "notes" in judge_output and judge_output["notes"]:
            judge_output["notes"] = f"{judge_output['notes']}; {note}"
        else:
            judge_output["notes"] = note
    except Exception:
        pass
    return judge_output

def apply_relation_conflict_penalty(judge_output, graph_reasoning):
    if not isinstance(judge_output, dict):
        return judge_output
    if "confidence" not in judge_output or judge_output["confidence"] is None:
        return judge_output
    try:
        conflicts = 0
        if isinstance(graph_reasoning, dict):
            conflicts = len(graph_reasoning.get("relation_conflicts") or [])
        if conflicts <= 0:
            return judge_output
        base = float(judge_output["confidence"])
        penalty = min(0.5, settings.judge_relation_conflict_penalty * conflicts)
        judge_output["confidence"] = max(0.0, base * (1.0 - penalty))
        note = f"confidence reduced for relation conflicts (penalty={penalty:.2f})"
        if "notes" in judge_output and judge_output["notes"]:
            judge_output["notes"] = f"{judge_output['notes']}; {note}"
        else:
            judge_output["notes"] = note
    except Exception:
        pass
    return judge_output
