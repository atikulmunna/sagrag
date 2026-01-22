# SAG-RAG Backend

Speculative -> Agentic -> Graph RAG backend with FastAPI, Qdrant, Elasticsearch, and Neo4j.
This repo implements the full SAG-RAG pipeline plus production scaffolding (observability, k8s/Helm, CI).

What we built
- End-to-end SAG-RAG pipeline: speculative planning, multi-agent retrieval, re-ranking, graph reasoning, judge, and synthesis with provenance.
- Production scaffolding: Docker, Helm, k8s manifests, CI workflows, and observability stack configs.
- Continuous learning pipeline scaffold: feedback capture, exports, curation, split, and training/eval stubs.

What is special / novel
- Speculative query planning to generate targeted sub-queries before retrieval.
- Hybrid retrieval (vector + lexical + structured) with domain routing and policy controls.
- Knowledge graph enrichment with claims, contradictions, relations, and path-based evidence scoring.
- Explicit judge + synthesis stages to improve factual grounding and provenance.

Key features
- Speculative planner with fallback rules.
- Parallel vector, lexical, and structured retrieval agents.
- Cross-encoder re-ranker and deduplication.
- Knowledge graph building (entities, claims, relations, contradictions).
- Graph reasoning output with evidence scores and path signals.
- Judge step with contradiction penalties and relation boosts.
- Synthesis with provenance (offsets), confidence, and explain trace.
- Feedback endpoint and continuous learning export pipeline.
- Metrics, alerts, and tracing scaffolding.

Tools and techniques used
- FastAPI for API layer.
- Qdrant for vector search.
- Elasticsearch for BM25/lexical search.
- Neo4j for knowledge graph storage and traversal.
- SentenceTransformers for embeddings and re-ranking.
- spaCy for NER and relation extraction.
- OpenTelemetry stubs for tracing; Prometheus/Grafana for metrics.
- Docker, Helm, and Kubernetes manifests for deployment.

Architecture (high level)
1) Client sends query to FastAPI.
2) Speculative planner emits intent and sub-queries.
3) Agent controller fans out to vector/lexical/structured retrievers.
4) Results deduped and re-ranked.
5) Graph builder enriches evidence (entities, claims, relations).
6) Graph reasoning produces evidence scores and contradiction signals.
7) Judge validates evidence and sets confidence.
8) Synthesis generates final answer with provenance.
9) Feedback and audit logs stored for learning.

Total workflow (query path)
Request -> plan -> retrieve -> dedupe -> rerank -> graph context -> judge -> synthesis -> response.

Quickstart (local)
1) Start infra services (Qdrant/Elasticsearch/Neo4j)
2) Run the backend container or `uvicorn app.main:app --host 0.0.0.0 --port 8000`
3) Ingest docs: `POST /v1/ingest`
4) Query: `POST /v1/query`

Local env
- Copy `.env.example` to `.env` and fill in values.

Config (env vars)
- `GEMINI_API_KEY` (required for planner/judge/synthesis)
- `QDRANT_URL`, `ELASTIC_URL`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- `GRAPH_ENABLED` (true/false)
- `DOMAIN_KEYWORDS` (JSON dict of domain -> keyword list)
- `DOMAIN_MIN_KEYWORD_HITS` (default: 2)
- `DOMAIN_ALIASES` (JSON dict of domain -> alias list)
- `POLICY_BLOCKLIST`, `POLICY_ALLOWLIST`
- `POLICY_SOURCE_TYPES_ALLOW`, `POLICY_SOURCE_TYPES_BLOCK`
- `POLICY_DOMAINS_ALLOW`, `POLICY_DOMAINS_BLOCK`
- `POLICY_RULES` (JSON list of rule objects with `action`, `domains`, `source_types`, `contains`, `not_contains`)
- `TENANT_ISOLATION` (true/false)
- `LLM_MAX_CONCURRENT` (limits concurrent LLM calls)

Domain routing example
```
DOMAIN_KEYWORDS='{"stoicism":["stoic","seneca","epictetus","marcus"],"finance":["earnings","revenue","sec"]}'
DOMAIN_MIN_KEYWORD_HITS=2
```

Observability
- Metrics: `GET /metrics` (Prometheus format)
- Grafana dashboard stub: `infra/observability/grafana-dashboard.json`
- Tracing: set `OTEL_ENABLED=true` and `OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317`
- Local stack: `docker compose -f infra/observability/docker-compose.yml up -d`

Deployment
- Helm chart: `infra/helm/sag-rag`
- k8s manifest: `infra/k8s/backend.yaml`
- Learning CronJob: `infra/learning/cron.yaml`

Continuous learning
- Export: `bash scripts/export_training_data.sh /data/learning/train.jsonl`
- Pipeline: `bash infra/learning/pipeline.sh` (curate, split, train stub, eval stub)

Evaluation harness
- `python scripts/eval_harness.py --base-url http://localhost:8000 --cases data/eval_cases.jsonl`

Audit/feedback CLI
- `python scripts/view_audit.py --limit 20`
- `python scripts/view_feedback.py --limit 20`

Admin UI
- `GET /ui` (simple dashboard for recent queries and feedback)

Notes on completeness
This repo is a production-ready MVP with full pipeline wiring and scaffolding.
Advanced training (LoRA), full GNN reasoning, and large-scale evaluation are left as research extensions.

Comparison (SAG-RAG vs other RAG approaches)

| Technique | Strengths | Limitations | How SAG-RAG improves |
|---|---|---|---|
| Naive RAG (single retriever, no planning) | Simple, fast to build | Irrelevant context, weaker grounding | Speculative planner + multi-agent retrieval reduce noise |
| Vector-only RAG | Captures semantic similarity | Misses exact matches, brittle for numbers | Adds lexical + structured retrievers and re-ranking |
| BM25-only RAG | Good exact match recall | Poor semantic coverage | Adds embeddings + planner to target needs |
| Hybrid RAG (vector + BM25) | Better recall than single method | Still noisy; limited reasoning | Adds graph reasoning, judge, and provenance |
| Rerank-only pipelines | Higher relevance | Still lacks reasoning/verification | Adds explicit judge and contradiction handling |
| Graph-RAG (basic entity graph) | Improved explainability | Shallow reasoning | Adds claims, contradictions, relation paths, and evidence scoring |
| Toolformer-style agentic RAG | Flexible tool use | Hard to audit; noisy | Explicit planning + retrieval envelopes + audit logs |
