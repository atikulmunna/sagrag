# SAG-RAG Architecture

This document expands the README overview into a component + data-flow reference,
a config reference, the endpoint surface, and the failure-tag taxonomy. For a
quick start see the [README](../README.md); for the live, always-current API
schema see the OpenAPI docs at `/docs` (Swagger UI) and `/redoc` when the backend
is running.

SAG-RAG is a **Speculative → Agentic → Graph** RAG backend: it plans sub-queries
before retrieving, fans out to several retrievers in parallel, enriches evidence
with a knowledge graph, runs an explicit judge, and synthesizes a grounded answer
with provenance. Every stage is designed to **degrade gracefully** — a missing
service, a non-JSON LLM reply, or an unreachable Redis reduces quality but never
takes the request path down.

## Components

| Component | Module(s) | Backing service | Responsibility |
|---|---|---|---|
| API + middleware | `main.py`, `api.py` | — | Routing, rate limiting, auth, metrics |
| Speculative planner | `speculative.py` | Ollama | Intent + normalized sub-queries (JSON mode, temp 0) |
| Domain router | `agents.py` (`route_domain`) | — | Maps query → domain via keywords/aliases (domain packs) |
| Retrieval agents | `agents.py` | Qdrant, Elasticsearch | Parallel vector / lexical / structured / author search |
| Re-ranker | `reranker.py` | SentenceTransformers | Cross-encoder relevance re-scoring |
| Policy + freshness | `agents.py` | — | Allow/block lists, source-type/domain rules, recency |
| Graph reasoning | `graph.py`, `judge.py` | Neo4j | Entities, claims, relations, contradictions, path signals |
| Judge | `judge.py` | Ollama | Evidence validation, contradiction penalties, confidence |
| Synthesis | `synthesis.py` | Ollama | Grounded answer + provenance (buffered JSON and SSE stream) |
| Domain packs | `domain_packs.py` | `data/domain_packs/*.json` | Externalized authors/stopwords/synonyms/planner hints |
| Ingestion | `ingestion.py`, `ingest_jobs.py`, `graph.py` | all stores | Chunk → embed → batched upsert/index/graph; delete-by-source |
| Caching / limiter / queue | `redis_client.py` | Redis (best-effort) | Rate limit, response + embedding cache, ingest queue |
| Auth | `security.py` | — | `x-api-key` → tenant, enforced server-side |
| Telemetry | `metrics.py`, `otel.py` | Prometheus, OTel | Prometheus text metrics + optional OTLP traces |
| Persistence | `store.py` | SQLite | Audit log, feedback |

## Query data flow

```
Request
  → auth (x-api-key → tenant)            [security.py, opt-in]
  → rate limit (Redis or in-process)     [main.py]
  → response cache lookup                [redis_client, best-effort]
  → plan_query (intent + sub-queries)    [speculative.py]
  → route_domain                         [agents.py]
  → retrieval fanout (vector/lexical/structured/author, parallel)
  → aggregate → policy + freshness filter → dedupe → author bias → rerank
  → (optional) graph context + reasoning [graph.py, judge.py]
  → judge_evidence (confidence, penalties/boosts)
  → synthesize_answer (JSON) | synthesize_answer_stream (SSE)
  → response (answer + provenance + confidence + explain_trace)
  → cache store + audit log + metrics
```

The buffered `POST /v1/query` is the **stable contract** and the regression anchor
for the eval harness. `POST /v1/query/stream` runs the identical
plan→retrieve→rerank→judge pipeline, then streams the synthesis answer as SSE
token events followed by a final event carrying provenance/confidence/trace.

## Ingestion data flow

```
POST /v1/ingest  → creates a background job (asyncio), returns job_id immediately
  worker: for each doc → chunk_text (offsets) → embed
    → batched Qdrant upsert
    → Elasticsearch bulk index
    → Neo4j UNWIND (entities/claims/relations)
  reingest_replaces_source: delete existing source rows first (no duplicates)
GET  /v1/ingest/{job_id}          → job status (queued/running/succeeded/failed)
DELETE /v1/ingest/source/{source} → remove a source from all three stores
```

Ingestion runs off the event loop (`asyncio.to_thread`), so `/health` and queries
stay responsive during a large ingest.

## Endpoint reference

All endpoints are mounted at both `/v1/*` (canonical) and `/*` (alias). When
`AUTH_ENABLED=true`, every endpoint requires `x-api-key` except `/health`,
`/metrics`, and the OpenAPI docs.

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/query` | Buffered grounded answer + provenance (cached) |
| POST | `/v1/query/stream` | SSE token stream + final provenance event |
| POST | `/v1/ingest` | Start a background ingest job → `{job_id}` |
| GET | `/v1/ingest/{job_id}` | Ingest job status |
| DELETE | `/v1/ingest/source/{source}` | Delete a source from all stores |
| GET | `/v1/audit` | Recent query audit log (paged) |
| POST | `/v1/audit/export` | Export audit log to JSONL |
| POST | `/v1/feedback` | Record user feedback (rating/comment) |
| GET | `/v1/feedback/list` | Recent feedback |
| POST | `/v1/learning/export` | Export high-rated interactions for training |
| POST | `/v1/graph/summary` | Graph summary for a query/domain |
| GET | `/metrics` | Prometheus text metrics (auth-exempt) |
| GET | `/health` | Liveness (auth-exempt) |
| GET | `/ui` | Minimal admin dashboard |

## Config reference

Full env-var list and shapes live in [`.env.example`](../.env.example); defaults
in `app/config.py`. Grouped highlights:

- **LLM / models:** `OLLAMA_URL`, `OLLAMA_MODEL`, `EMBEDDING_MODEL_NAME`,
  `RERANKER_MODEL_NAME`, `LLM_MAX_CONCURRENT`, `*_TIMEOUT_S`, `*_MAX_TOKENS`.
- **Stores:** `QDRANT_URL`, `ELASTIC_URL`, `NEO4J_URI/USER/PASSWORD`, `GRAPH_ENABLED`.
- **Routing / domain packs:** `DOMAIN_KEYWORDS`, `DOMAIN_ALIASES`,
  `DOMAIN_MIN_KEYWORD_HITS`, `QUERY_TERM_SYNONYMS`, `DOMAIN_PACKS_PATH`,
  `AUTHOR_INDEX_PATH`.
- **Policy:** `POLICY_BLOCKLIST/ALLOWLIST`, `POLICY_SOURCE_TYPES_*`,
  `POLICY_DOMAINS_*`, `POLICY_RULES`, `DEFAULT_FRESHNESS_DAYS`.
- **Redis (best-effort):** `REDIS_URL`, `REDIS_RATE_LIMIT_ENABLED`,
  `REDIS_CACHE_ENABLED`, `QUERY_CACHE_TTL_S`, `EMBED_CACHE_TTL_S`,
  `INGEST_QUEUE_NAME`.
- **Ingestion:** `INGEST_BATCH_SIZE`, `REINGEST_REPLACES_SOURCE`.
- **Auth / tenancy:** `AUTH_ENABLED`, `API_KEYS`, `API_KEY_MAP`, `TENANT_ISOLATION`.
- **Observability:** `OTEL_ENABLED`, `OTEL_SERVICE_NAME`,
  `OTEL_EXPORTER_OTLP_ENDPOINT`, `RATE_LIMIT_PER_MINUTE`.

## Failure-tag taxonomy

Retrieval and synthesis failures are counted in `sag_rag_retrieval_failures_total{tag=...}`
so degraded paths are observable rather than silent:

- **Routing:** `no_domain`, `cross_domain_conflict`.
- **Policy:** `policy_blocked`.
- **Per-retriever** (`vector`/`lexical`/`structured`/`lexical_author`):
  `*_timeout`, `*_zero_hits`, `*_error`.
- **Result quality:** `no_results`, `low_result_count`, `low_top_score`,
  `author_gap`.
- **Synthesis:** `synthesis_timeout`, `synthesis_error` (plus
  `sag_rag_synthesis_total{outcome=non_json|timeout|error|ok|stream|stream_fallback}`).

Answer-quality gauges: `sag_rag_hallucination_risk_bucket` and
`sag_rag_evidence_coverage_bucket`.

## Observability & security

See the README [Observability](../README.md#observability) and
[Security](../README.md#security-api-key-auth--tenants) sections. In short:
Prometheus/Grafana/OTel ship behind the `observability` Compose profile; auth is
opt-in via `x-api-key` with server-side tenant binding.

## Evaluation

`tools/ablation_eval.py` fires a labeled query set (`data/eval/queries.jsonl`) at
one or more configs; `tools/eval_metrics.py` scores retrieval (recall@k / hit-rate)
and answer quality (LLM-judge faithfulness + lexical-overlap baseline) into a
Markdown report at `docs/eval_report.md`. The report is generated against a live
stack (not committed) — see [`data/eval/README.md`](../data/eval/README.md).

## Known limitations / research extensions

- **Graph reasoning depth.** The graph layer captures entities, claims, relations,
  contradictions, and simple path signals, but does not do multi-hop GNN reasoning
  or learned graph traversal — deeper reasoning is a research extension.
- **Evaluation scope.** The bundled eval set is small and single-domain (stoicism);
  recall@k and faithfulness numbers are indicative, not benchmark-grade. Expanding
  ground truth across domains is future work.
- **Tenant isolation scope.** Auth binds the tenant server-side, but storage
  shards by tenant *and* domain with **domain taking precedence**: no-domain data
  is isolated per tenant (`docs_{tenant}`), while domain-classified data is shared
  across tenants (`docs_{domain}`). Strict cross-domain isolation would require
  composing both into `docs_{tenant}_{domain}` (`_target_names` in `ingestion.py`
  + collection selection in `agents.py`). This is a deliberate, tested default.
- **Single-node infra.** Qdrant / Elasticsearch / Neo4j / Redis run single-node via
  Compose. Redis is best-effort (rate limit + cache + queue degrade to in-process
  when it's down); the ingest "queue" is enqueued for observability but the worker
  is in-process (`asyncio`), not a separate distributed consumer yet.
- **Training.** LoRA / fine-tuning is out of scope; `learning/export` only emits
  high-rated interactions as training data.
- **Route duplication.** Endpoints are intentionally mounted at both `/v1/*` and
  `/*` for compatibility; the root alias is a known redundancy kept for stability.
