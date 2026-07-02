# app/speculative.py
import json
import re
from llm_client import llm
import domain_packs

def _safe_json_extract(text: str):
    try:
        match = re.search(r"{.*}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return None

async def plan_query(user_query: str):
    prompt = f"""
You are a Speculative Query Planner for a RAG system.

User query:
\"\"\"{user_query}\"\"\"

Return JSON with keys:
{{ "intent": "...", "hypotheses": [...], "queries": [...], "constraints": {{}} }}
"""
    try:
        out = await llm.completion(prompt, max_tokens=300, format="json", temperature=0)
        parsed = _safe_json_extract(out)
        if parsed:
            return parsed
    except Exception:
        pass

    # Fallback rules from domain packs (no domain logic hard-coded here).
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
    if not queries:
        queries.append(user_query)
    return {"intent": intent, "hypotheses": hypotheses, "queries": queries, "constraints": {}}
