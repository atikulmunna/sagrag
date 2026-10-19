# SAG-RAG Architecture

This document expands the README overview into a component and data-flow
reference, the tenancy model, the endpoint surface, and the failure-tag
taxonomy. For a quick start see the [README](../README.md); for the live API
schema see the OpenAPI docs at `/docs` (Swagger UI) and `/redoc` when the
backend is running.

SAG-RAG is a **Speculative → Agentic → Graph** RAG backend: it plans sub-queries
before retrieving, fans out to several retrievers in parallel, optionally
enriches evidence with a knowledge graph, runs an explicit judge, and
synthesizes an answer with validated provenance. Every stage is designed to
**degrade gracefully**: a missing service, a non-JSON LLM reply, or an
unreachable Redis reduces quality but never takes the request path down, and
each degraded path is tagged and counted.

## Components

| Component | Module(s) | Backing service | Responsibility |
|---|---|---|---|
| App + middleware | `main.py` | none | Lifespan, auth gate, rate limiting, request metrics, `/health`, `/ready`, optional CORS |
| Routes | `api.py`, `schemas.py` | none | Pydantic-validated endpoints, tenant resolution, SSE streaming |
| Query pipeline | `pipeline.py` | none | Stage functions over a `QueryContext`; failure tags, confidence calibration, response shape |
| Speculative planner | `speculative.py` | Ollama | Intent, hypotheses, sub-queries (JSON mode, temp 0), sanitized before use |
| Domain router | `agents.py` (`route_domain`, `known_domains`) | Elasticsearch (index listing) | Query → domain via keywords/aliases; validates requested domains |
| Retrieval agents | `agents.py` | Qdrant, Elasticsearch | Parallel vector and lexical search; structured lines derived from lexical hits; author search |
| Embeddings | `embeddings.py` | SentenceTransformers, Redis | Single model load, batched + cached query embeddings |
| Author logic | `authors.py` | none | Author detection, term expansion, bias and author-gap handling |
| Re-ranker | `reranker.py` | SentenceTransformers | Cross-encoder relevance re-scoring |
| Policy + freshness | `agents.py` | none | Allow/block lists, source-type/domain rules (first match wins), recency |
| Graph reasoning | `graph.py`, `judge.py` | Neo4j | Entities, claims, relations, heuristic contradictions, path signals (tenant-scoped) |
| Judge | `judge.py` | Ollama | Trusted passages, confidence, contradiction penalties, relation boosts |
| Synthesis | `synthesis.py` | Ollama | Answer + validated provenance (buffered JSON and SSE stream) |
| Evidence references | `refs.py` | none | `E1`..`En` labels in judge/synthesis prompts, mapped back to chunk ids |
| Verification | `verification.py` | SentenceTransformers (NLI cross-encoder) | Per-sentence support check against cited passages; abstention; graph contradiction confirmation |
| Storage naming | `namespaces.py` | none | Collection/index names per tenant and domain; slug validation |
| Domain packs | `domain_packs.py` | `data/domain_packs/*.json` | Externalized authors, stopwords, synonyms, planner hints |
| Ingestion | `ingestion.py`, `ingest_jobs.py`, `graph.py` | all stores | Chunk → embed → batched upsert/index/graph; scoped delete-by-source |
| Caching / limiter / queue | `redis_client.py` | Redis (best effort) | Rate limit, response + embedding cache, ingest queue |
| Auth | `security.py` | none | `x-api-key` → tenant (constant-time compare, hash-derived default tenants) |
| Telemetry | `metrics.py`, `otel.py` | Prometheus, OTel | Prometheus text metrics + optional OTLP traces |
| Persistence | `store.py`, `continuous_learning.py` | SQLite (WAL) | Audit log and feedback (tenant + `query_id`), training export |
| Web UI | `web/`, `infra/nginx/web.conf` | nginx | Static app; proxies `/v1`, `/ready`, `/health` to the backend |

## Query data flow

```
Request (validated by schemas.QueryRequest)
  → auth (x-api-key → tenant)              [main.py + security.py, opt-in]
  → rate limit (tenant, else client IP)    [main.py, Redis or in-process]
  → domain preference check                [pipeline.validated_preferences]
  → response cache lookup                  [redis_client, best effort]
  → plan        intent + sub-queries, sanitized          [speculative.py]
  → retrieve    route domain; embed sub-queries once;
                vector + lexical per (query, domain); structured lines from
                lexical hits; policy + freshness; dedupe; author bias and
                author search                            [agents.py, authors.py]
  → rerank      reciprocal rank fusion picks at most RERANK_MAX_CANDIDATES,
                then the cross-encoder; author filtering / gap fallback
  → graph       (optional) entities, claims, relations   [graph.py via judge.py]
  → judge       trusted ids (validated), confidence, graph penalties/boosts
  → synthesize  JSON answer | SSE token stream           [synthesis.py]
  → verify      (opt-in) each sentence vs cited passages (NLI)  [verification.py]
  → finalize    failure tags, confidence floor, hallucination risk
                (measured share of unsupported sentences when verified), coverage
  → audit (tenant, query_id) + cache store + metrics
```

`POST /v1/query` and `POST /v1/query/stream` run the same stages and the same
`finalize` step. The stream emits a `stage` event before and after each stage
(with its duration), `meta` before the answer, `token` deltas, and a `final`
event whose body equals the buffered response.

## Ingestion data flow

```
POST /v1/ingest  → background job (asyncio task, one at a time per process)
  worker: for each file under /data/docs[/<domain>]
    → read (.txt raw; .csv/.json flattened to "key: value" lines)
    → chunk_text: 200-word windows, 50-word overlap, text[start:end] == chunk
    → embed (batched)
    → delete the source's old rows in this namespace (reingest_replaces_source)
    → batched Qdrant upsert, Elasticsearch bulk index, Neo4j UNWIND writes
    → errors collected per stage; job ends succeeded | completed_with_errors | failed
GET    /v1/ingest/{job_id}          → job status (visible to the owning tenant only)
DELETE /v1/ingest/source/{source}   → remove a source from the caller's namespaces
```

Ingestion runs off the event loop (`asyncio.to_thread`), so `/health` and
queries stay responsive during a large ingest. Chunk ids are
`uuid5(collection:source:index)`, unique per namespace. Payloads record
`offset_basis` (`raw` or `flattened`) so offsets for structured files are read
against the flattened text.

## Tenancy and storage names

| Mode | No domain | Domain `d` |
|---|---|---|
| Shared (isolation off) | `docs`, `docs_index` | `docs_d`, `docs_index_d` (falls back to `docs`) |
| Isolated tenant `t` | `docs_t_t`, `docs_index_t_t` | `docs_t_t__d_d`, `docs_index_t_t__d_d` (no fallback) |

Tenant and domain names are slugs without consecutive separators and domains
may not start with `t_`, so a domain can never name a tenant namespace.
Tenants that are not slug-shaped are hashed. With auth on, the tenant comes
from the key (`API_KEY_MAP`, or a hash of the key); body and query-string
tenants are ignored. Audit and feedback rows, exports, job status, deletes,
and every Neo4j read and write are filtered by tenant. `DOMAIN_INDEX_MAP`
overrides apply to the shared layout only.

## Endpoint reference

All endpoints are mounted at both `/v1/*` (canonical) and `/*` (alias). When
`AUTH_ENABLED=true`, every endpoint requires `x-api-key` except `/health`,
`/ready`, `/metrics`, and the OpenAPI docs.

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/query` | Buffered answer + provenance + diagnostics (cached), returns `query_id` |
| POST | `/v1/query/stream` | SSE `stage` / `meta` / `token` / `final` / `error` / `done` |
| GET | `/v1/domains` | Domains valid for `preferences.domain` |
| POST | `/v1/ingest` | Start a background ingest job → 202 `{job_id}` |
| GET | `/v1/ingest/{job_id}` | Ingest job status |
| DELETE | `/v1/ingest/source/{source}` | Delete a source from the caller's namespaces |
| GET | `/v1/audit` | Recent query audit log (cursor paging) |
| POST | `/v1/audit/export` | NDJSON download, or `save` to `AUDIT_EXPORT_PATH` |
| POST | `/v1/feedback` | Record a rating (1 to 5), comment, and optional `query_id` |
| GET | `/v1/feedback/list` | Recent feedback |
| POST | `/v1/learning/export` | Interactions rated `>= min_rating` (NDJSON or `save`) |
| POST | `/v1/graph/summary` | Graph reasoning for chunk ids (tenant-scoped) |
| GET | `/metrics` | Prometheus text metrics (auth-exempt) |
| GET | `/health` | Liveness (auth-exempt) |
| GET | `/ready` | Dependency checks: Qdrant, Elasticsearch, Ollama (+ Neo4j when enabled; Redis informational); 503 when a required one fails |

## Failure-tag taxonomy

Retrieval and synthesis failures are returned in `retrieval_failures` and
counted in `sag_rag_retrieval_failures_total{tag=...}`:

- **Routing:** `no_domain`, `cross_domain_conflict`.
- **Policy:** `policy_blocked`.
- **Per retriever** (`vector`, `lexical`, `structured`, `lexical_author`):
  `*_timeout`, `*_error`, `*_zero_hits`. The structured agent inherits the
  lexical search's timeout/error status.
- **Result quality:** `no_results`, `low_result_count`, `low_top_score`,
  `author_gap`.
- **Synthesis:** `synthesis_timeout`, `synthesis_error` (plus
  `sag_rag_synthesis_total{outcome=success|non_json|timeout|error|stream|stream_fallback}`).
- **Verification:** `unsupported_claims`, `contradicted_claims`, `abstained`
  (plus `sag_rag_claims_total{status=supported|unsupported|contradicted}`).

Answer-quality histograms: `sag_rag_hallucination_risk` and
`sag_rag_evidence_coverage` (`_bucket`, `_sum`, `_count`).

## Observability and security

See the README [Observability](../README.md#observability) and
[Security](../README.md#security) sections. In short: Prometheus, Grafana and
OTel ship behind the `observability` Compose profile; request metrics are
labelled by route template; auth is opt-in via `x-api-key` with server-side
tenant binding; the web container sends a strict CSP.

## Evaluation

`tools/ablation_eval.py` fires a labeled query set (`data/eval/queries.jsonl`)
at one or more configs; `tools/eval_metrics.py` scores retrieval (recall@k /
hit rate) and answer quality (LLM-judge faithfulness and a lexical-overlap
baseline) into a Markdown report at `docs/eval_report.md`. The report is
generated against a live stack and not committed; see
[`data/eval/README.md`](../data/eval/README.md).

## Known limitations

- **Scores.** Confidence comes from the judge model plus graph penalties and
  an alignment floor and is not calibrated. Hallucination risk is measured per
  sentence only when verification is on; otherwise it is derived from
  confidence.
- **Verifier scope.** A small NLI cross-encoder checks sentences against the
  cited passages only, using sentence-sized premises; it has a twelve-case
  sanity set (`tools/verify_eval.py`), not a benchmark.
- **Graph reasoning depth.** Entities, claims, relations, negation-based
  contradictions, and simple path signals; no multi-hop or learned traversal.
- **Evaluation scope.** The bundled eval set is small and single-domain;
  recall@k and faithfulness numbers are indicative, not benchmark-grade.
- **Single-node infra.** Qdrant, Elasticsearch, Neo4j and Redis run
  single-node via Compose. The ingest queue entry is written for
  observability, but the worker is in-process, not a distributed consumer.
  Metrics and the job registry are per process.
- **Training.** LoRA or fine-tuning is out of scope; `learning/export` only
  emits highly rated interactions.
- **Route duplication.** Endpoints are mounted at both `/v1/*` and `/*` for
  compatibility.
