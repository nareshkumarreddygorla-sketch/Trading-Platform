# Production Deployment

## Prerequisites

- Docker & Docker Compose (or Kubernetes cluster)
- PostgreSQL 15+, Redis 7+, Kafka 3+ (or managed equivalents)
- Angel One API credentials (for live India trading)

## Configuration

1. Copy `config/settings.example.yaml` to `config/settings.yaml`.
2. Set environment or config: broker credentials, DB URLs, Kafka brokers, feature flags.
3. Never commit secrets; use a secrets manager in production.

## Docker Compose (Single Node)

```bash
docker-compose -f deploy/docker/docker-compose.yaml up -d
```

Services: `api`, `engine` (strategy+risk+execution worker), `market-data`, `redis`, `postgres`, `kafka`, `zookeeper` (if needed).

## Kubernetes

1. Create namespace: `kubectl create namespace trading`
2. Apply secrets and configmaps: `kubectl apply -f deploy/k8s/config/`
3. Deploy: `kubectl apply -f deploy/k8s/`
4. Ingress and TLS as per cluster.

## CI/CD (GitHub Actions)

- On PR: lint, unit tests, integration tests (with testcontainers or mocks).
- On merge to main: build images, push to registry, deploy to staging (then manual/auto promote to prod).
- Feature flags control strategy rollout and risk limit changes.

## Blue/Green

- Two deployments (blue/green); switch ingress to new version after health checks.
- Rollback by switching back.

## Monitoring

- Prometheus scrapes `/metrics` from API and engine.
- Grafana dashboards: `deploy/observability/grafana/dashboards/`
- Alerts: `deploy/observability/prometheus/alerts.yaml`

## Compliance

- All order and risk events are logged to audit store (DB or object store).
- Config changes are versioned and auditable.
