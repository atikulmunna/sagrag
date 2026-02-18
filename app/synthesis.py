import asyncio
import json
import logging
import re
import time

from llm_client import llm
from config import settings
from metrics import record_synthesis

_LOG = logging.getLogger(__name__)

_TECHNICAL_PATTERNS = (
    "{",
    "}",
    "[{",
    "graph_reasoning",
    "provenance",
    "offset_start",
    "offset_end",
    "\"source\"",
    "'source'",
    "\"id\"",
    "'id'",
)

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

def _normalize_sentence_key(sentence: str) -> str:
    s = re.sub(r"[^a-z0-9\s]", "", sentence.lower())
    s = re.sub(r"\s+", " ", s).strip()
    # Use prefix to collapse near-duplicates.
    return " ".join(s.split()[:10])

def _clamp_natural_answer(text: str, min_sentences: int = 2, max_sentences: int = 4) -> str:
    sents = _extract_sentences(text, max_sentences=8)
    if not sents:
        return _clean_answer_text(text)
    deduped = []
    seen = set()
    for s in sents:
        key = _normalize_sentence_key(s)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
        if len(deduped) >= max_sentences:
            break
    if len(deduped) < min_sentences and sents:
        deduped = sents[:max_sentences]
    return _clean_answer_text(" ".join(deduped))

def _looks_technical(text: str) -> bool:
    if not text:
        return True
    t = text.lower()
    return any(p in t for p in _TECHNICAL_PATTERNS)

def _clean_answer_text(text: str) -> str:
    if not text:
        return ""
    # Remove obvious structured/log lines.
    lines = []
    for line in text.splitlines():
        l = line.strip()
        if not l:
            continue
        low = l.lower()
        if low.startswith("{") or low.startswith("[") or "provenance" in low or "graph_reasoning" in low:
            continue
        lines.append(l)
    text = " ".join(lines).strip()
    text = re.sub(r"\s+", " ", text)
    return text

def _is_natural_answer(text: str) -> bool:
    t = _clean_answer_text(text)
    if len(t) < 40:
        return False
    if _looks_technical(t):
        return False
    return True

async def _naturalize_answer(query: str, picks: list[dict], author_terms=None, author_gap=False) -> str:
    snippets = []
    for p in picks[:3]:
        snippets.append({
            "text": (p.get("text") or "")[:350],
            "source": p.get("source"),
        })
    author_hint = ""
    if author_terms:
        author_hint = f"Author focus: {author_terms}. "
        if author_gap:
            author_hint += "No direct keyword match from author; summarize nearby evidence naturally. "
    prompt = f"""
You are a writing assistant. Produce a natural, user-friendly answer in plain English.

Rules:
- 2-4 sentences.
- Do not output JSON, lists, dicts, code, or metadata.
- Do not copy long passages verbatim.
- Avoid repetition and filler.
- Explain the idea clearly and practically.

Query: {query}
{author_hint}
Evidence: {snippets}
"""
    try:
        timeout_s = max(3.0, min(float(settings.synthesis_timeout_s), 10.0))
        out = await asyncio.wait_for(
            llm.completion(prompt, max_tokens=min(220, settings.synthesis_max_tokens)),
            timeout=timeout_s,
        )
        cleaned = _clamp_natural_answer(out, min_sentences=2, max_sentences=4)
        if _is_natural_answer(cleaned):
            return cleaned
    except Exception:
        pass
    return ""

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
        return "I could not find enough clean evidence to generate a natural answer."
    if author_terms and author_gap:
        prefix = f"No direct passages from {', '.join(author_terms)} mention the query keywords in the current dataset. "
        return _clean_answer_text(prefix + " ".join(sentences[:2]))
    return _clean_answer_text(" ".join(sentences[:2]))

async def synthesize_answer(query, evidence, judge_output, author_terms=None, author_gap=False):
    started = time.monotonic()
    def _finish(payload: dict, outcome: str):
        latency_ms = int((time.monotonic() - started) * 1000)
        record_synthesis(outcome, latency_ms)
        return payload

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
    fail_trace = "synthesis_unavailable"
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
            parsed["answer"] = _clamp_natural_answer(parsed.get("answer", ""), min_sentences=2, max_sentences=4)
            if not _is_natural_answer(parsed["answer"]):
                trusted = []
                if isinstance(judge_output, dict):
                    trusted = judge_output.get("trusted_ids") or []
                by_id = {e.get("id"): e for e in evidence}
                picks = [by_id[t] for t in trusted if t in by_id]
                if not picks:
                    picks = evidence[:3]
                natural = await _naturalize_answer(query, picks, author_terms, author_gap)
                if natural:
                    parsed["answer"] = natural
                    parsed["explain_trace"] = "synthesis_naturalized"
                else:
                    parsed["answer"] = _clamp_natural_answer(
                        _format_fallback_answer(picks, author_terms, author_gap),
                        min_sentences=1,
                        max_sentences=4,
                    )
                    parsed["explain_trace"] = "synthesis_fallback_formatted"
            return _finish(parsed, "success")
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
            # Keep answer clean: do not surface raw model blob when JSON contract fails.
            clean_answer = await _naturalize_answer(query, picks, author_terms, author_gap)
            if not clean_answer:
                clean_answer = _format_fallback_answer(picks, author_terms, author_gap)
            if not clean_answer:
                clean_answer = "I could not synthesize a clean answer from the available evidence."
            return _finish({
                "answer": clean_answer,
                "provenance": provenance,
                "confidence": judge_output.get("confidence", 0.3) if isinstance(judge_output, dict) else 0.3,
                "explain_trace": "synthesis_non_json",
            }, "non_json")
    except asyncio.TimeoutError:
        fail_trace = "synthesis_timeout"
        _LOG.exception("synthesis_timeout")
    except Exception:
        fail_trace = "synthesis_error"
        _LOG.exception("synthesis_failed")
    fallback = {
        "answer": "",
        "provenance": [],
        "confidence": judge_output.get("confidence", 0.3) if isinstance(judge_output, dict) else 0.3,
        "explain_trace": fail_trace,
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
            # Preserve timeout/error traces for telemetry; only mark
            # formatted fallback when there was no explicit synthesis failure.
            if fail_trace == "synthesis_unavailable":
                fallback["explain_trace"] = "synthesis_fallback_formatted"
    outcome = "fallback_formatted" if fallback.get("answer") else fail_trace
    if fail_trace == "synthesis_timeout":
        outcome = "timeout"
    elif fail_trace in ("synthesis_error", "synthesis_unavailable"):
        outcome = "error"
    return _finish(fallback, outcome)
