import asyncio
import json
import logging
import re
import time

import refs
from config import settings
from jsonutil import as_unit_float, safe_json_extract
from llm_client import llm
from metrics import record_synthesis

_LOG = logging.getLogger(__name__)

# Backwards-compatible alias (tests and older callers import it from here).
_safe_json_extract = safe_json_extract

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


SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "cited": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["answer", "cited", "confidence"],
}


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


_RANGE_DASH_RE = re.compile(r"(\d)\s*[\u2013\u2014]\s*(\d)")
_DASH_RE = re.compile(r"\s*[\u2013\u2014]\s*")
_DOUBLED_PUNCT_RE = re.compile(r",\s*([,.;:!?])")


def normalize_dashes(text: str) -> str:
    """House style for generated answers: no em or en dashes.

    A dash between two numbers becomes "to" ("1917 to 1925"); any other em
    or en dash becomes a comma. Source passages are never rewritten, only
    model-written answers.
    """
    if not text:
        return text
    text = _RANGE_DASH_RE.sub(r"\1 to \2", text)
    text = _DASH_RE.sub(", ", text)
    return _DOUBLED_PUNCT_RE.sub(r"\1", text)


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
    return normalize_dashes(text)


def _is_natural_answer(text: str) -> bool:
    t = _clean_answer_text(text)
    if len(t) < 40:
        return False
    if _looks_technical(t):
        return False
    return True


def _snippets(picks, limit=3, chars=350):
    return [{"text": (p.get("text") or "")[:chars], "source": p.get("source")} for p in picks[:limit]]


def _author_hint(author_terms, author_gap, gap_note):
    if not author_terms:
        return ""
    hint = f"Author focus: {author_terms}. "
    if author_gap:
        hint += gap_note
    return hint


async def _naturalize_answer(query: str, picks: list[dict], author_terms=None, author_gap=False) -> str:
    author_hint = _author_hint(
        author_terms, author_gap, "No direct keyword match from author; summarize nearby evidence naturally. "
    )
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
Evidence: {_snippets(picks)}
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
    except Exception as e:
        _LOG.info("naturalize_failed: %r", e)
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


def _is_author_passage(e, author_terms):
    s = (e.get("source") or "").lower()
    t = (e.get("text") or "").lower()
    return any(a in s or a in t for a in author_terms)


def _pick_evidence(evidence, judge_output, author_terms=None, author_gap=False):
    """Select the evidence items to ground an answer on: judge-trusted ids
    when available, else the top few. Under an author gap, prefer non-author
    passages."""
    trusted = []
    if isinstance(judge_output, dict):
        trusted = judge_output.get("trusted_ids") or []
    by_id = {e.get("id"): e for e in evidence}
    picks = [by_id[t] for t in trusted if t in by_id]
    if not picks:
        picks = evidence[:3]
    if author_terms and author_gap:
        non_author = [e for e in picks if not _is_author_passage(e, author_terms)]
        if non_author:
            picks = non_author
    return picks


def _provenance_item(e):
    if e.get("offset_start") is None or e.get("offset_end") is None or not e.get("source"):
        return None
    return {
        "id": e.get("id"),
        "source": e.get("source"),
        "offset_start": e.get("offset_start"),
        "offset_end": e.get("offset_end"),
    }


def _provenance_for(picks):
    return [item for item in (_provenance_item(p) for p in picks) if item]


def validate_provenance(model_provenance, evidence, labelled=None):
    """Keep only citations that point at retrieved evidence.

    Accepts E-labels ("E2"), chunk ids, or {"id": ...} objects. Source and
    offsets always come from the evidence, never from the model, so an invented
    id, file, or span can never reach the response.
    """
    if not isinstance(model_provenance, list):
        return []
    by_id = {e.get("id"): e for e in evidence}
    ids = refs.resolve(model_provenance, labelled or [], evidence)
    return [item for item in (_provenance_item(by_id[eid]) for eid in ids) if item]


_ANSWER_PREFIX_RE = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)')


def salvage_answer(raw: str) -> str:
    """Complete sentences of an answer whose JSON was cut off by max_tokens."""
    m = _ANSWER_PREFIX_RE.search(raw or "")
    if not m:
        return ""
    fragment = m.group(1)
    if fragment.endswith("\\"):
        fragment = fragment[:-1]  # cut inside an escape sequence
    try:
        text = json.loads(f'"{fragment}"')
    except ValueError:
        text = fragment
    end = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    return text[: end + 1].strip() if end > 0 else ""


def build_stream_provenance(evidence, judge_output, author_terms=None, author_gap=False):
    """Compute provenance server-side for the streaming path (the model streams
    prose, not JSON, so provenance can't come from the model output)."""
    return _provenance_for(_pick_evidence(evidence, judge_output, author_terms, author_gap))


def _judge_confidence(judge_output, default=0.3):
    if isinstance(judge_output, dict):
        return as_unit_float(judge_output.get("confidence"), default=default)
    return default


async def synthesize_answer_stream(query, evidence, judge_output, author_terms=None, author_gap=False):
    """Stream a plain-prose answer as text deltas (no JSON contract).

    Yields incremental token strings. If the model produces nothing (or errors),
    yields a single formatted fallback built from the evidence so the client
    always receives an answer. Provenance/confidence are computed separately by
    the caller via build_stream_provenance / the judge confidence.
    """
    started = time.monotonic()
    picks = _pick_evidence(evidence, judge_output, author_terms, author_gap)
    author_hint = _author_hint(
        author_terms, author_gap, "No direct keyword match from that author; summarize nearby evidence and say so briefly. "
    )
    prompt = f"""You are a writing assistant. Answer the query in natural, plain English.

Rules:
- 2-4 sentences.
- Do not output JSON, lists, dicts, code, or metadata.
- Ground the answer in the evidence; do not invent facts.
- Avoid repetition and filler.

Query: {query}
{author_hint}
Evidence: {_snippets(picks)}
"""
    got_any = False
    try:
        async for delta in llm.completion_stream(
            prompt, max_tokens=settings.synthesis_max_tokens, temperature=0.2
        ):
            got_any = True
            yield delta
    except Exception:
        _LOG.exception("synthesis_stream_failed")
    if not got_any:
        fallback = _format_fallback_answer(picks, author_terms, author_gap)
        if fallback:
            yield fallback
    latency_ms = int((time.monotonic() - started) * 1000)
    record_synthesis("stream" if got_any else "stream_fallback", latency_ms)


async def synthesize_answer(query, evidence, judge_output, author_terms=None, author_gap=False):
    """Grounded answer with provenance. Always returns exactly the keys
    ``answer``, ``provenance``, ``confidence`` and ``explain_trace``."""
    started = time.monotonic()

    def _finish(answer, provenance, confidence, trace, outcome):
        record_synthesis(outcome, int((time.monotonic() - started) * 1000))
        return {"answer": answer, "provenance": provenance, "confidence": confidence, "explain_trace": trace}

    labelled = refs.label(evidence, settings.max_evidence_snippets)
    label_of = {e.get("id"): ref for ref, e in labelled}
    judge_summary = "none"
    if isinstance(judge_output, dict):
        trusted = [label_of[t] for t in judge_output.get("trusted_ids") or [] if t in label_of]
        judge_summary = (
            f"confidence {judge_output.get('confidence')}; trusted passages: {', '.join(trusted) or 'none'}; "
            f"notes: {judge_output.get('notes') or 'none'}"
        )
    graph_relations = ""
    evidence_scores = ""
    if isinstance(judge_output, dict):
        rel = judge_output.get("graph_reasoning")
        if isinstance(rel, dict):
            graph_relations = rel.get("relation_strength") or rel.get("relations") or ""
            evidence_scores = rel.get("evidence_scores") or ""
    author_hint = ""
    if author_terms:
        author_hint = f"\nAuthor focus: {author_terms}\n"
        if author_gap:
            author_hint += "Note: No author passages explicitly mention the query keywords; use other sources and say so briefly. Do not quote unrelated author passages.\n"
    prompt = f"""
You are a synthesis model. Return JSON only:
{{"answer": "2-4 sentences", "cited": ["E1", ...], "confidence": 0.0-1.0}}
List in "cited" the labels of the evidence passages your answer relies on.

Grounding rules:
- Answer directly in 2-4 sentences.
- If the query names an author, prioritize evidence from that author. If none exists, say so briefly.
- Prefer claims supported by graph relations and evidence snippets.
- If relations contradict, mention the conflict and lower confidence.
- Do not invent relations; cite only those provided.

User query:
{query}
{author_hint}

Judge assessment:
{judge_summary}

Graph relations (if any):
{graph_relations}

Evidence scores (if any):
{evidence_scores}

Evidence:
{refs.render(labelled, 500)}
"""
    judge_conf = _judge_confidence(judge_output)
    picks = _pick_evidence(evidence, judge_output, author_terms, author_gap)
    fail_trace = "synthesis_unavailable"
    try:
        out = await asyncio.wait_for(
            llm.completion(prompt, max_tokens=settings.synthesis_max_tokens, format=SYNTHESIS_SCHEMA, temperature=0),
            timeout=settings.synthesis_timeout_s,
        )
        parsed = safe_json_extract(out)
        if parsed and isinstance(parsed.get("answer"), str) and parsed["answer"].strip():
            cited = parsed.get("cited")
            if cited is None:
                cited = parsed.get("provenance")
            provenance = validate_provenance(cited, evidence, labelled) or _provenance_for(picks)
            confidence = as_unit_float(parsed.get("confidence"), default=judge_conf)
            trace = "synthesis_json"
            answer = _clamp_natural_answer(parsed["answer"], min_sentences=2, max_sentences=4)
            if not _is_natural_answer(answer):
                natural = await _naturalize_answer(query, picks, author_terms, author_gap)
                if natural:
                    answer, trace = natural, "synthesis_naturalized"
                else:
                    answer = _clamp_natural_answer(
                        _format_fallback_answer(picks, author_terms, author_gap),
                        min_sentences=1,
                        max_sentences=4,
                    )
                    trace = "synthesis_fallback_formatted"
            return _finish(answer, provenance, confidence, trace, "success")
        _LOG.warning("synthesis_parse_failed")
        salvaged = _clamp_natural_answer(salvage_answer(out), min_sentences=1, max_sentences=4)
        if _is_natural_answer(salvaged):
            # The reply ran out of tokens mid-JSON; keep its complete sentences
            # instead of paying for another LLM call.
            return _finish(salvaged, _provenance_for(picks), judge_conf, "synthesis_json_truncated", "truncated")
        if out and out.strip():
            _LOG.warning("synthesis_non_json")
            # Keep answer clean: do not surface raw model blob when JSON contract fails.
            clean_answer = await _naturalize_answer(query, picks, author_terms, author_gap)
            if not clean_answer:
                clean_answer = _format_fallback_answer(picks, author_terms, author_gap)
            return _finish(clean_answer, _provenance_for(picks), judge_conf, "synthesis_non_json", "non_json")
    except asyncio.TimeoutError:
        fail_trace = "synthesis_timeout"
        _LOG.warning("synthesis_timeout")
    except Exception:
        fail_trace = "synthesis_error"
        _LOG.exception("synthesis_failed")
    answer = _format_fallback_answer(picks, author_terms, author_gap)
    trace = fail_trace
    # Preserve timeout/error traces for telemetry; only mark formatted
    # fallback when there was no explicit synthesis failure.
    if answer and fail_trace == "synthesis_unavailable":
        trace = "synthesis_fallback_formatted"
    outcome = "timeout" if fail_trace == "synthesis_timeout" else "error"
    return _finish(answer, _provenance_for(picks), judge_conf, trace, outcome)
