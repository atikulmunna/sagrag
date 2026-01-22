from fastapi import APIRouter
from store import fetch_audit_logs, fetch_feedback

router = APIRouter()

@router.get("/ui")
def ui_page():
    html = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>SAG-RAG Admin</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; }
      h2 { margin-top: 24px; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ddd; padding: 6px; vertical-align: top; }
      th { background: #f2f2f2; }
      pre { white-space: pre-wrap; margin: 0; }
    </style>
  </head>
  <body>
    <h1>SAG-RAG Admin</h1>
    <h2>Recent Queries</h2>
    <div id="audit"></div>
    <h2>Recent Feedback</h2>
    <div id="feedback"></div>
    <script>
      async function loadAudit() {
        const res = await fetch('/v1/audit?limit=20');
        const data = await res.json();
        const rows = data.results || [];
        let html = '<table><tr><th>User</th><th>Query</th><th>Intent</th><th>Confidence</th><th>Answer</th></tr>';
        for (const r of rows) {
          html += `<tr><td>${r.user_id||''}</td><td>${r.query||''}</td><td>${r.intent||''}</td><td>${r.confidence||''}</td><td><pre>${r.answer||''}</pre></td></tr>`;
        }
        html += '</table>';
        document.getElementById('audit').innerHTML = html;
      }
      async function loadFeedback() {
        const res = await fetch('/v1/feedback/list?limit=20');
        const data = await res.json();
        const rows = data.results || [];
        let html = '<table><tr><th>User</th><th>Query</th><th>Rating</th><th>Comment</th></tr>';
        for (const r of rows) {
          html += `<tr><td>${r.user_id||''}</td><td>${r.query||''}</td><td>${r.rating||''}</td><td>${r.comment||''}</td></tr>`;
        }
        html += '</table>';
        document.getElementById('feedback').innerHTML = html;
      }
      loadAudit();
      loadFeedback();
    </script>
  </body>
</html>
"""
    return html
