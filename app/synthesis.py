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

async def synthesize_answer(query, evidence, judge_output):
    evidence_snippets = []
    for e in evidence[: settings.max_evidence_snippets]:
        snippet = e.get("text", "")[:500]
        evidence_snippets.append({
            "id": e.get("id"),
            "text": snippet,
            "source": e.get("source"),
            "offset_start": e.get("offset_start"),
            "offset_end": e.get("offset_end"),
        })
    graph_relations = ""
    evidence_scores = ""
    if isinstance(judge_output, dict):
        rel = judge_output.get("graph_reasoning")
        if isinstance(rel, dict):
            if rel.get("relation_strength"):
                graph_relations = rel.get("relation_strength")
            elif rel.get("relations"):
                graph_relations = rel.get("relations")
            if rel.get("evidence_scores"):
                evidence_scores = rel.get("evidence_scores")
    prompt = f"""
You are a synthesis model. Return JSON only:
{{"answer": "...", "provenance": [...], "confidence": 0.0-1.0, "explain_trace": "..."}}
Each provenance item must include id, source, offset_start, and offset_end.

Grounding rules:
- Prefer claims supported by graph relations and evidence snippets.
- If relations contradict, mention the conflict and lower confidence.
- Do not invent relations; cite only those provided.

User query:
{query}

Judge output:
{judge_output}

Graph relations (if any):
{graph_relations}

Evidence scores (if any):
{evidence_scores}

Evidence snippets:
{evidence_snippets}
"""
    try:
        out = await asyncio.wait_for(llm.completion(prompt, max_tokens=500), timeout=settings.synthesis_timeout_s)
        parsed = _safe_json_extract(out)
        if parsed and parsed.get("answer"):
            prov = parsed.get("provenance") if isinstance(parsed, dict) else None
            if isinstance(prov, list):
                by_id = {e.get("id"): e for e in evidence}
                cleaned = []
                for p in prov:
                    if not isinstance(p, dict):
                        continue
                    eid = p.get("id") or p.get("chunk_id")
                    if eid in by_id:
                        e = by_id[eid]
                        p.setdefault("offset_start", e.get("offset_start"))
                        p.setdefault("offset_end", e.get("offset_end"))
                        p.setdefault("source", e.get("source"))
                    if p.get("offset_start") is None or p.get("offset_end") is None or not p.get("source"):
                        continue
                    cleaned.append(p)
                parsed["provenance"] = cleaned
            # Fallback provenance if model returned none after cleaning.
            if not parsed.get("provenance"):
                trusted = []
                if isinstance(judge_output, dict):
                    trusted = judge_output.get("trusted_ids") or []
                by_id = {e.get("id"): e for e in evidence}
                picks = [by_id[t] for t in trusted if t in by_id]
                if not picks:
                    picks = evidence[:3]
                parsed["provenance"] = [
                    {
                        "id": p.get("id"),
                        "source": p.get("source"),
                        "offset_start": p.get("offset_start"),
                        "offset_end": p.get("offset_end"),
                    }
                    for p in picks
                    if p.get("offset_start") is not None and p.get("offset_end") is not None and p.get("source")
                ]
            return parsed
    except Exception:
        pass
    fallback = {
        "answer": "",
        "provenance": [],
        "confidence": judge_output.get("confidence", 0.3) if isinstance(judge_output, dict) else 0.3,
        "explain_trace": "synthesis_unavailable",
    }
    trusted = []
    if isinstance(judge_output, dict):
        trusted = judge_output.get("trusted_ids") or []
    by_id = {e.get("id"): e for e in evidence}
    picks = [by_id[t] for t in trusted if t in by_id]
    if not picks:
        picks = evidence[:3]
    fallback["provenance"] = [
        {
            "id": p.get("id"),
            "source": p.get("source"),
            "offset_start": p.get("offset_start"),
            "offset_end": p.get("offset_end"),
        }
        for p in picks
        if p.get("offset_start") is not None and p.get("offset_end") is not None and p.get("source")
    ]
    return fallback
