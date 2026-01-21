# Speculative → Agentic → Graph RAG (SAG-RAG)

A complete, deployable architecture and deployment plan for a hybrid Retrieval-Augmented Generation system that combines speculative query planning, multi-agent retrieval, and graph-based reasoning.

## Executive Summary

SAG-RAG reduces hallucinations and improves explainability by forcing the LLM to speculate what it will need, using a panel of specialized retriever agents to gather evidence, and validating/connecting evidence using a knowledge graph before final synthesis. The result is a modular, auditable, and cost-effective RAG system suitable for on-prem or cloud deployment.

## High-Level Architecture

1. **Client** (web, slack, voice) → sends user query
2. **API Gateway / Frontend** (FastAPI / Node) → pre-processes, rate-limits, auth
3. **Speculative Planner** (small LLM) → emits: intent label(s), hypotheses, prioritized information needs, retrieval queries, and retrieval constraints (time range, domain)
4. **Agent Controller** → orchestrates retrieval agents in parallel and aggregates results
5. **Retrieval Agents** (parallel):
   - Semantic Vector Retriever (Qdrant/Weaviate)
   - Lexical Retriever (BM25 / Elasticsearch)
   - Table/JSON Extractor (structured doc retriever)
   - Domain Classifier Agent (routes to domain-specific index)
   - Freshness/Time Filter Agent
   - Legal/Policy Filter Agent (optional)
6. **Evidence Re-Ranker & Deduplicator** (Redis / Lambda re-ranker)
7. **Graph Builder / Knowledge Graph** (Neo4j / AWS Neptune / Dgraph)
   - Nodes: documents, claims, entities, citations, facts
   - Edges: supports, contradicts, cites, derived_from, updated_by
8. **Graph Reasoner** (GNN or symbolic traverser) → collects connected evidence subgraph
9. **Judge LLM** (verifier) → checks alignment/conflicts and outputs confidence scores and provenance tokens
10. **Synthesis LLM** → final answer with provenance, suggested citations, and confidence
11. **Feedback collector** → user ratings, corrections, production telemetry
12. **Continuous Learning pipeline** → LoRA fine-tuning / RLHF / dataset curation

## Data Model & Indexing

### Document Chunk Model

```
doc_id, chunk_id, text, embedding, entities[], entities_norm[], 
timestamp, version, source_type, metadata (domain, language, sensitivity)
```

### Graph Schema (Simplified)

**Node types:**
- DocumentChunk
- Entity
- Claim
- Source
- UserReport

**Edge types:**
- CITES
- SUPPORTS
- CONTRADICTS
- MENTIONS
- UPDATED_BY

**Node attributes:**
- confidence_score
- last_verified
- extracted_by

### Indexes

- **Vector index:** chunk embeddings (dimension e.g. 1536)
- **Inverted index:** chunk text (for BM25)
- **Entity index:** canonical entity lookup table
- **Temporal index:** timestamp and valid_from/valid_to

## Speculative Planner (Detailed)

**Purpose:** Generate what-ifs — short, crisp queries the retrieval system should run.

**Inputs:** user query + session context + user profile

**Outputs (example):**
```
- intent: "product_inquiry"
- hypotheses: ["user asks for price history", "user expects supported OS list"]
- queries: ["product X price history 2022..2024", "product X supported OS"]
- constraints: {domain: "product", freshness: 365 days}
```

**Implementation options:**
- Small instruct-tuned model (100M–1B) locally via LoRA
- Rules + templates for low-cost deterministic fallback

**Why important:** Encourages targeted retrieval, reduces noisy context, and helps the judge LLM debug mistakes.

## Agent Controller & Retrieval Agents

### Controller Responsibilities

- Fan-out the speculative queries to agents
- Apply timeouts and concurrency limits
- Collect and attach retrieval metadata (score, method, latency)

### Agent Types (Detailed)

- **Vector Retriever:** nearest neighbor search on embeddings (Qdrant/Weaviate)
- **Lexical Retriever:** BM25 style search; exact-match boosts
- **Structured Extractor:** uses regex/parsers/JSONPath to pull tables and fields
- **Domain Router:** routes queries to domain-specific indices (namespace isolation)
- **Freshness Agent:** filters by timestamp and version
- **Policy Agent:** removes disallowed or sensitive docs before exposure

**Return envelope for each agent:**
```json
{
  "agent_id": "...",
  "results": [{"chunk_id": "...", "text": "...", "score": 0.9, "metadata": {}}],
  "elapsed_ms": 150
}
```

## Evidence Re-Ranking & Deduplication

- Use a lightweight re-ranker model (cross-encoder like MiniLM tuned for relevance) to re-score aggregated retrievals
- Deduplicate by normalized content fingerprint (SimHash or cosine threshold)
- Keep top-K (configurable, default K=10)

## Knowledge Graph Construction & Usage

### Ingestion Pipeline

1. Extract entities & relations from chunks using an IE model (NER + RE + coref)
2. Normalize entities (linking to canonical IDs)
3. Create/merge nodes and edges with provenance

### Graph Operations Used at Query-Time

- Subgraph extraction around retrieved nodes
- Shortest-path checks between claims and sources
- Find corroborating or contradicting paths
- Surface chain-of-evidence for the judge LLM

**Graph store choices:** Neo4j (ACID, good tooling) or Dgraph/Neptune for cloud scale.

## Graph Reasoner & Judge LLM

### Graph Reasoner Tasks

- Walk the subgraph to find corroboration density
- Score each claim by number and quality of supporting nodes
- Provide a small evidence_summary that the Judge LLM ingests

### Judge LLM Responsibilities

- Accept user query, evidence_summary, and hypotheses
- Detect contradictions and hallucination risk
- Produce a confidence_score and mark which retrieved chunks are trustworthy

**Model options:**
- Small-medium model (few hundred million parameters) for the judge to reduce cost
- Optionally use an ensemble of judges for critical domains

## Synthesis LLM

**Inputs:** user query, validated evidence, provenance snippets, confidence metadata

**Outputs:**
- final answer
- provenance[] (top N chunks with citations)
- confidence
- explain_trace (short chain-of-evidence)

**Safety & style:** system prompts should enforce clarity, avoid overclaiming, and require the model to list uncertain facts with confidence brackets.

## API Design (Example)

### Request

```http
POST /v1/query
```

```json
{
  "user_id": "u_123",
  "query": "Is feature X supported in product Y?",
  "session_state": {},
  "preferences": {
    "freshness_days": 365
  }
}
```

### Response

```json
{
  "answer": "Yes — product Y supports X on these OSes...",
  "provenance": [
    {
      "source": "doc.pdf",
      "chunk_id": "c_234",
      "cursor": 0
    }
  ],
  "confidence": 0.86,
  "explain_trace": "Speculative planner requested price history; vector+lexical found matching docs; judge verified no contradiction"
}
```

## Deployment Plan

### Containerization

Each service as a Docker container:

- api-gateway
- speculative-planner
- agent-controller
- retrieval-agents (stateless)
- re-ranker
- graph-service
- judge-llm
- synthesis-llm
- feedback/ingestion

### Orchestration

- Kubernetes with Helm charts
- Use Horizontal Pod Autoscaler for LLM services
- Node pools: GPU nodes for model services (judge + synthesis), CPU nodes for agents and graph

### Storage

- **Vector DB:** Qdrant (persistent SSD volumes)
- **Graph DB:** Neo4j on a dedicated statefulset or managed Neptune
- **Metadata:** PostgreSQL
- **Cache:** Redis (re-rank cache & session store)
- **Object storage:** S3-compatible for raw documents

### CI/CD

- Code builds via GitHub Actions or GitLab CI
- Container images to registry (ECR/GCR)
- Canary deployments and automated smoke tests

### Observability

- **Traces:** OpenTelemetry
- **Logs:** ELK / Loki + Grafana
- **Metrics:** Prometheus + Grafana
- **Alerting:** PagerDuty / Opsgenie

## Security & Compliance

- Role-based access control to indices and graph
- PII detection and scrubbing in ingestion
- Query-level audit logs: store query, returned chunks, timestamps
- Data residency options (on-premise indexes)
- Model access tokens with short TTL

## Cost & Scaling Considerations

- Keep speculative planner & judge LLM small to limit calls
- Use smaller re-rankers and only call big synthesis LLM when confidence below threshold
- Cache common queries and chain-of-thought traces
- Use mixed-precision inference on GPU and cheap CPU offloading for non-ML tasks

## Evaluation & Metrics

- **Accuracy:** human-annotated test set (EM, F1)
- **Hallucination rate:** percent of unsupported claims
- **Response latency:** p50/p95/p99
- **Cost per 1k queries**
- **Provenance coverage:** fraction of answers with >=1 high-quality citation
- **User satisfaction:** 5-star rating + feedback text

## Example: Minimal Viable Implementation (MVP)

**Goal:** Deploy a cloud MVP that demonstrates core flows with minimal infra.

**MVP components:**
- FastAPI gateway (Docker)
- Speculative planner: small instruction model (hosted via a local inference server)
- Vector DB: hosted Qdrant
- Lexical: Elasticsearch managed
- Graph: small Neo4j instance
- Judge LLM & Synthesis: single mid-sized model (serve via Triton or HuggingFace Inference)
- Re-ranker: tiny cross-encoder

**Minimum features:** speculative planning, vector+lexical retrieval, graph subgraph extraction, judge verification, final synthesis with provenance.

## Troubleshooting & Pitfalls

- **Over-speculation:** planner emits too many queries → rate-limit & cap K
- **Graph bloat:** prune stale nodes and run periodic consolidation
- **Latency:** parallelize agents aggressively and use timeouts
- **Privacy leakage:** run policy agent early in the pipeline

## Roadmap & Milestones (12 Weeks)

- **Week 1–2:** Data ingestion + vector & lexical indexing
- **Week 3–4:** Speculative planner + agent controller
- **Week 5–6:** Basic graph construction + Neo4j
- **Week 7–8:** Judge LLM + re-ranker
- **Week 9:** Synthesis LLM + provenance formatting
- **Week 10:** Deploy MVP to k8s + CI/CD
- **Week 11:** Instrumentation + metrics
- **Week 12:** User testing, iterate, LoRA fine-tune

## Deliverables Available

- Dockerfiles and Helm charts
- FastAPI boilerplate (complete)
- Speculative planner prompt library + LoRA starter script
- Agent controller implementation (async Python)
- Graph ingestion pipeline (NER + RE) code
- Judge & synthesis prompt templates
- End-to-end test harness + synthetic dataset

## Ready Next Actions

- Generate the FastAPI codebase and Dockerfiles
- Scaffold Kubernetes manifests and Helm charts
- Produce a small synthetic dataset and unit tests

**Tell me which deliverable you want first and I'll produce it.**