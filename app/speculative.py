# app/speculative.py
"""Query planner: intent, hypotheses, and search sub-queries.

The LLM's plan is treated as untrusted input (the user's query is inside its
prompt, so the query text can steer it). ``sanitize_plan`` therefore keeps the
original query as the first search, caps and dedupes sub-queries, and allows
only a freshness hint as a constraint: the model can never pick a domain or
change policy.
"""
import domain_packs
from config import settings
from jsonutil import safe_json_extract
from llm_client import llm

_MAX_QUERY_CHARS = 300
_MAX_HYPOTHESES = 5


def _clean_str(value, limit):
    if isinstance(value, (str, int, float)) and str(value).strip():
        return " ".join(str(value).split())[:limit]
    return None


def _freshness_days(constraints):
    if not isinstance(constraints, dict):
        return None
    raw = constraints.get("freshness_days", constraints.get("freshness"))
    try:
        days = int(raw)
    except (TypeError, ValueError):
        return None
    return days if 1 <= days <= 36500 else None


def sanitize_plan(plan, user_query: str) -> dict:
    p = plan if isinstance(plan, dict) else {}
    intent = _clean_str(p.get("intent"), 64) or "general"
    raw_h = p.get("hypotheses") if isinstance(p.get("hypotheses"), list) else []
    hypotheses = [h for h in (_clean_str(x, _MAX_QUERY_CHARS) for x in raw_h) if h][:_MAX_HYPOTHESES]
    raw_q = p.get("queries")
    if not isinstance(raw_q, list):
        raw_q = [raw_q] if raw_q is not None else []
    queries = []
    seen = set()
    for q in [user_query, *raw_q]:
        q = _clean_str(q, _MAX_QUERY_CHARS)
        if not q or q.casefold() in seen:
            continue
        seen.add(q.casefold())
        queries.append(q)
        if len(queries) >= max(1, settings.planner_max_queries):
            break
    constraints = {}
    days = _freshness_days(p.get("constraints"))
    if days is not None:
        constraints["freshness_days"] = days
    return {"intent": intent, "hypotheses": hypotheses, "queries": queries, "constraints": constraints}


def _fallback_plan(user_query: str) -> dict:
    """Rule-based plan from domain packs (no domain logic hard-coded here)."""
    intent = "general"
    hypotheses = []
    queries = []
    q = user_query.lower()
    for rule in domain_packs.planner_rules():
        triggers = [str(t).strip().lower() for t in (rule.get("triggers") or [])]
        if any(t and t in q for t in triggers):
            if rule.get("intent"):
                intent = rule["intent"]
            hypotheses.extend(rule.get("hypotheses") or [])
            queries.extend(rule.get("queries") or [])
    return {"intent": intent, "hypotheses": hypotheses, "queries": queries, "constraints": {}}


async def plan_query(user_query: str):
    max_sub = max(0, settings.planner_max_queries - 1)
    prompt = f"""
You are a query planner for a retrieval system.

User query:
\"\"\"{user_query}\"\"\"

Return JSON with keys:
{{"intent": "short label", "hypotheses": ["possible answers to look for"], "queries": ["up to {max_sub} alternative search queries"], "constraints": {{"freshness_days": null}}}}
"""
    try:
        out = await llm.completion(prompt, max_tokens=300, format="json", temperature=0)
        parsed = safe_json_extract(out)
        if parsed:
            return sanitize_plan(parsed, user_query)
    except Exception:
        pass
    return sanitize_plan(_fallback_plan(user_query), user_query)
