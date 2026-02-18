import json
import os
from typing import Any

import gradio as gr
import requests


API_BASE = os.getenv("SAG_RAG_API_BASE", "http://backend:8000").rstrip("/")
QUERY_URL = f"{API_BASE}/v1/query"
TIMEOUT_S = float(os.getenv("SAG_RAG_UI_TIMEOUT_S", "90"))


def _pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, indent=2)


def run_query(user_id: str, query: str):
    user_id = (user_id or "").strip() or "u1"
    query = (query or "").strip()
    if not query:
        return "Please enter a query.", "", "", "[]", "[]", "{}"

    try:
        resp = requests.post(
            QUERY_URL,
            json={"user_id": user_id, "query": query},
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return (
            f"Request failed: {e}",
            "",
            "",
            "[]",
            "[]",
            "{}",
        )

    answer = data.get("answer", "")
    explain_trace = str(data.get("explain_trace", ""))
    confidence = str(data.get("confidence", ""))
    retrieval_failures = _pretty(data.get("retrieval_failures", []))
    top_results = _pretty((data.get("results") or [])[:3])
    retrieval_stats = _pretty(data.get("retrieval_stats", {}))
    return answer, explain_trace, confidence, retrieval_failures, top_results, retrieval_stats


with gr.Blocks(title="SAG-RAG Gradio UI") as demo:
    gr.Markdown("# SAG-RAG Demo")
    gr.Markdown("Queries the FastAPI backend and shows answer, trace, failures, and top evidence.")

    with gr.Row():
        user_id = gr.Textbox(label="User ID", value="u1")
        query = gr.Textbox(
            label="Query",
            value="What does Seneca say about fear and how to handle it?",
            scale=4,
        )

    run_btn = gr.Button("Run Query", variant="primary")

    answer = gr.Textbox(label="Answer", lines=5)
    with gr.Row():
        explain_trace = gr.Textbox(label="Explain Trace")
        confidence = gr.Textbox(label="Confidence")

    with gr.Row():
        retrieval_failures = gr.Code(label="Retrieval Failures", language="json")
        retrieval_stats = gr.Code(label="Retrieval Stats", language="json")
    top_results = gr.Code(label="Top Results (first 3)", language="json")

    run_btn.click(
        run_query,
        inputs=[user_id, query],
        outputs=[answer, explain_trace, confidence, retrieval_failures, top_results, retrieval_stats],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
