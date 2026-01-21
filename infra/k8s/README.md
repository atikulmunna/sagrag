SAG-RAG k8s manifests

This folder keeps the backend Deployment/Service only. Qdrant, Elasticsearch, and Neo4j are expected to be external services (managed or separate StatefulSets).

Configure endpoints

- Qdrant: set `QDRANT_URL`
- Elasticsearch: set `ELASTIC_URL`
- Neo4j: set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`

Update `infra/k8s/backend.yaml` or use environment overrides in your deployment pipeline.
