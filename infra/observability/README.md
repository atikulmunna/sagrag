Observability setup

Prometheus

- Use `infra/observability/prometheus.yml` and set the target to your backend service.

Grafana

- Import `infra/observability/grafana-dashboard.json`.
- Add a Prometheus datasource pointing to your Prometheus server.

Example datasource (Grafana provisioning)

apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    url: http://prometheus:9090
    access: proxy

Docker Compose (local)

  docker compose -f infra/observability/docker-compose.yml up -d

Grafana is available at http://localhost:3000 (admin/admin by default).
Alertmanager is available at http://localhost:9093.
