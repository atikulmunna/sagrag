SAG-RAG Helm Chart

Install (dev)

  helm install sag-rag infra/helm/sag-rag

Install (prod)

  helm install sag-rag infra/helm/sag-rag -f infra/helm/sag-rag/values-prod.yaml

Upgrade

  helm upgrade sag-rag infra/helm/sag-rag -f infra/helm/sag-rag/values-prod.yaml

Notes

- Set external service endpoints in values.
- Set GEMINI_API_KEY via values or a secret manager (creates a Secret when set).
- Non-secret config is stored in a ConfigMap created by the chart.
- Probes are enabled by default; adjust in values.yaml if needed.
