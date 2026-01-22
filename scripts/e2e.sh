#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Starting services..."
docker compose -f infra/docker-compose.yml up -d

echo "Waiting for backend..."
for i in {1..30}; do
  if curl -sSf http://localhost:8000/health >/dev/null; then
    break
  fi
  sleep 2
done

curl -sSf http://localhost:8000/health >/dev/null

echo "Ingesting docs..."
curl -sSf -X POST http://localhost:8000/v1/ingest >/dev/null

echo "Querying..."
RESP="$(curl -sSf -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u1","query":"What does Seneca say about fear?"}')"

RESP_JSON="$RESP" python - <<'PY'
import json, os
resp = json.loads(os.environ.get("RESP_JSON", "{}"))
assert "answer" in resp, "missing answer"
assert "provenance" in resp, "missing provenance"
assert isinstance(resp.get("provenance"), list), "provenance not list"
print("E2E ok")
PY

echo "Metrics..."
curl -sSf http://localhost:8000/metrics >/dev/null

echo "UI..."
curl -sSf http://localhost:8000/ui >/dev/null

echo "Done."
