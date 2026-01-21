# app/speculative.py
import json
import re
from llm_client import llm

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
        out = await llm.completion(prompt, max_tokens=300)
        parsed = _safe_json_extract(out)
        if parsed:
            return parsed
    except Exception:
        pass

    # fallback rules
    intent = "general"
    hypotheses = []
    queries = []
    q = user_query.lower()
    if any(w in q for w in ["fear", "anxiety", "worry"]):
        intent = "stoic-guidance"
        hypotheses.append("User seeks Stoic emotional guidance")
        queries.append("Stoic advice on fear")
        queries.append("Seneca fear wisdom")
    if not queries:
        queries.append(user_query)
    return {"intent": intent, "hypotheses": hypotheses, "queries": queries, "constraints": {}}
