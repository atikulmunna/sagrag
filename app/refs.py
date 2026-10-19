# app/refs.py
"""Short evidence references for LLM prompts.

Chunk ids are 36-character UUIDs. Asking a model to echo them costs about 20
tokens each, and under a tight ``max_tokens`` the JSON gets cut off mid-id, so
the whole reply is lost. Prompts therefore label evidence ``E1``..``En`` and
the model's references are mapped back to real ids here.
"""
import re

_REF_RE = re.compile(r"^\[?\s*E(\d+)\s*\]?$", re.IGNORECASE)


def label(evidence: list[dict], limit: int) -> list[tuple[str, dict]]:
    """[("E1", item), ("E2", item), ...] for the first ``limit`` items."""
    return [(f"E{i}", e) for i, e in enumerate(evidence[:limit], start=1)]


def render(labelled: list[tuple[str, dict]], chars: int) -> str:
    """Compact evidence block: one "[E1] (source) text" line per item."""
    lines = []
    for ref, e in labelled:
        text = " ".join(str(e.get("text") or "").split())[:chars]
        lines.append(f"[{ref}] ({e.get('source') or 'unknown'}) {text}")
    return "\n".join(lines)


def resolve(values, labelled: list[tuple[str, dict]], evidence: list[dict]) -> list:
    """Map model references ("E2", "[e2]", 2, or a real chunk id) to evidence ids.

    Unknown references are dropped, duplicates removed, order kept.
    """
    if not isinstance(values, list):
        return []
    by_ref = {ref.upper(): e.get("id") for ref, e in labelled}
    known_ids = {e.get("id") for e in evidence}
    out = []
    for v in values:
        if isinstance(v, dict):
            v = v.get("id") or v.get("ref") or v.get("chunk_id")
        eid = None
        if isinstance(v, int) and not isinstance(v, bool):
            eid = by_ref.get(f"E{v}")
        elif isinstance(v, str):
            m = _REF_RE.match(v.strip())
            eid = by_ref.get(f"E{m.group(1)}") if m else (v if v in known_ids else None)
        if eid is not None and eid not in out:
            out.append(eid)
    return out
