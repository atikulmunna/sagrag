import asyncio
import json
import logging
import re

from llm_client import llm
from config import settings

_LOG = logging.getLogger(__name__)

def _safe_json_extract(text: str):
    try:
        match = re.search(r"{.*}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return None

def _extract_sentences(text: str, max_sentences: int = 2):
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = []
    for s in parts:
        s = s.strip()
        if len(s) < 20:
            continue
        if not re.match(r"^[A-Z]", s):
            continue
        out.append(s)
        if len(out) >= max_sentences:
            break
    return out

def _format_fallback_answer(picks, author_terms, author_gap):
    sentences = []
    for p in picks:
        text = (p.get("text") or "").strip()
        if not text:
            continue
        sentences.extend(_extract_sentences(text, max_sentences=2))
        if len(sentences) >= 2:
            break
    if not sentences:
        return ""
    if author_terms and author_gap:
        prefix = f"No direct passages from {', '.join(author_terms)} mention the query keywords in the current dataset. "
        return prefix + " ".join(sentences[:2])
    return " ".join(sentences[:2])

async def synthesize_answer(query, evidence, judge_output, author_terms=None, author_gap=False):
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
    author_hint = ""
    if author_terms:
        author_hint = f"\nAuthor focus: {author_terms}\n"
        if author_gap:
            author_hint += "Note: No author passages explicitly mention the query keywords; use other sources and say so briefly. Do not quote unrelated author passages.\n"
    prompt = f"""
You are a synthesis model. Return JSON only:
{{"answer": "...", "provenance": [...], "confidence": 0.0-1.0, "explain_trace": "..."}}
Each provenance item must include id, source, offset_start, and offset_end.

Grounding rules:
- Answer directly in 2-4 sentences.
- If the query names an author, prioritize evidence from that author. If none exists, say so briefly.
- Prefer claims supported by graph relations and evidence snippets.
- If relations contradict, mention the conflict and lower confidence.
- Do not invent relations; cite only those provided.

User query:
{query}
{author_hint}

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
        out = await asyncio.wait_for(
            llm.completion(prompt, max_tokens=settings.synthesis_max_tokens),
            timeout=settings.synthesis_timeout_s,
        )
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
        _LOG.warning("synthesis_parse_failed")
        # Non-JSON fallback: use raw text as answer with best-effort provenance.
        trusted = []
        if isinstance(judge_output, dict):
            trusted = judge_output.get("trusted_ids") or []
        by_id = {e.get("id"): e for e in evidence}
        picks = [by_id[t] for t in trusted if t in by_id]
        if not picks:
            picks = evidence[:3]
        if author_terms and author_gap:
            def _is_author(e):
                s = (e.get("source") or "").lower()
                t = (e.get("text") or "").lower()
                return any(a in s or a in t for a in author_terms)
            non_author = [e for e in picks if not _is_author(e)]
            if non_author:
                picks = non_author
        provenance = [
            {
                "id": p.get("id"),
                "source": p.get("source"),
                "offset_start": p.get("offset_start"),
                "offset_end": p.get("offset_end"),
            }
            for p in picks
            if p.get("offset_start") is not None and p.get("offset_end") is not None and p.get("source")
        ]
        if out and out.strip():
            _LOG.warning("synthesis_non_json")
            return {
                "answer": out.strip(),
                "provenance": provenance,
                "confidence": judge_output.get("confidence", 0.3) if isinstance(judge_output, dict) else 0.3,
                "explain_trace": "synthesis_non_json",
            }
    except Exception:
        _LOG.exception("synthesis_failed")
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
    if author_terms and author_gap:
        def _is_author(e):
            s = (e.get("source") or "").lower()
            t = (e.get("text") or "").lower()
            return any(a in s or a in t for a in author_terms)
        non_author = [e for e in picks if not _is_author(e)]
        if non_author:
            picks = non_author
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
    if not fallback["answer"]:
        formatted = _format_fallback_answer(picks, author_terms, author_gap)
        if formatted:
            fallback["answer"] = formatted
            fallback["explain_trace"] = "synthesis_fallback_formatted"
    return fallback
