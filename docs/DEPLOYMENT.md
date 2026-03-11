# Deployment Guide

## 1. Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.11+ | Backend runtime |
| Node.js | 18+ | Frontend build |
| Docker | 24+ | Containerization |
| Docker Compose | 2.20+ | Local orchestration |
| PostgreSQL | 15+ | Primary database |
| Redis | 7+ | Cache and pub/sub |
| kubectl | 1.28+ | Kubernetes deployment (production) |

### Required Accounts & Credentials

| Credential | Purpose | Storage |
|-----------|---------|---------|
| Angel One API Key | Live trading (India) | Environment variable (encrypted) |
| Angel One Client ID | Broker authentication | Environment variable (encrypted) |
| Angel One Password | Broker authentication | Environment variable (encrypted) |
| Angel One TOTP Secret | Two-factor auth | Environment variable (encrypted) |
| JWT Secret | Token signing | Environment variable |
| PostgreSQL credentials | Database access | Environment variable |
| Redis URL | Cache access | Environment variable |

---

## 2. Environment Configuration

### Required Environment Variables

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/trading

# Redis
REDIS_URL=redis://host:6379/0

# JWT Authentication
JWT_SECRET=<random-256-bit-key>        # Generate with: python -c "import secrets; print(secrets.token_hex(32))"
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Broker (Angel One) - encrypted at rest
ANGEL_ONE_API_KEY=<encrypted>
ANGEL_ONE_CLIENT_ID=<encrypted>
ANGEL_ONE_PASSWORD=<encrypted>
ANGEL_ONE_TOTP_SECRET=<encrypted>

# Trading Mode
PAPER_MODE=true                         # true = paper trading, false = live trading
TRADING_MODE=paper                      # paper | live

# Application
LOG_LEVEL=info                          # debug | info | warning | error
ENVIRONMENT=production                  # development | staging | production

# Market Data
MD_REDIS_URL=redis://host:6379/0
MD_KAFKA_BROKERS=kafka:9092

# Model Training
TRAIN_SYMBOLS=all
TRAIN_PERIOD=60d
TRAIN_INTERVAL=5m
RETRAIN_CRON=0 2 * * 0                  # Weekly retraining at 2 AM Sunday
```

### Configuration Files

| File | Purpose |
|------|---------|
| `config/settings.yaml` | Application settings (copy from `config/settings.example.yaml`) |
| `deploy/deployment_config.yaml` | Deployment-specific configuration |
| `deploy/nse_holidays.json` | NSE market holiday calendar |
| `deploy/nse_lot_sizes.json` | NSE lot size reference data |
| `ruff.toml` | Linting and formatting configuration |
| `pyproject.toml` | Python project and pytest configuration |

---

## 3. Local Development Setup

### Quick Start

```bash
# Clone and enter repository
git clone <repo-url>
cd Trading-Platform

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Run database migrations
alembic upgrade head

# Start development server
uvicorn src.api.app:app --reload --port 8000
```

### Frontend Setup

```bash
cd trading-ui
npm ci
npm run dev                              # Development server on port 3000
```

### Running Tests

```bash
# Full test suite
pytest tests/ -v --tb=short --timeout=60

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Skip slow tests
pytest tests/ -v -m "not slow" -x

# Specific test file
pytest tests/test_risk_engine.py -v

# Lint and format
ruff check src tests --output-format=full
ruff format --check src tests
```

---

## 4. Docker Deployment

### Single-Node (Docker Compose)

```bash
# Start all services
docker-compose -f deploy/docker/docker-compose.yaml up -d

# View logs
docker-compose -f deploy/docker/docker-compose.yaml logs -f api

# Stop all services
docker-compose -f deploy/docker/docker-compose.yaml down
```

### Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `api` | `Dockerfile.api` | 8000 | FastAPI backend |
| `frontend` | `Dockerfile.frontend` | 3000 | Next.js frontend |
| `redis` | `redis:7-alpine` | 6379 (localhost only) | Cache |
| `postgres` | `postgres:15-alpine` | 5432 (localhost only) | Database |

### Docker Image Details

**API Image (`Dockerfile.api`):**
- Multi-stage build (builder + runtime)
- Non-root user for security
- `tini` as PID 1 init process
- Health check via `/health` endpoint
- Start period: 120s (allows time for model training)

**Frontend Image (`Dockerfile.frontend`):**
- Next.js 14 production build
- Environment variables for API/WS URLs

### Volumes

| Volume | Purpose |
|--------|---------|
| `models_data` | Persists trained ML models across deploys |
| `logs_data` | Persists training and application logs |
| `redis_data` | Redis persistence |
| `postgres_data` | PostgreSQL data directory |

---

## 5. Kubernetes Deployment

### Namespace Setup

```bash
# Create namespace
kubectl create namespace trading

# Apply configuration
kubectl apply -f deploy/k8s/config/configmap.yaml -n trading

# Deploy application
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/api-deployment.yaml -n trading
```

### K8s Resource Structure

```
deploy/k8s/
├── namespace.yaml           # trading namespace
├── api-deployment.yaml      # API deployment + service
└── config/
    └── configmap.yaml       # Non-sensitive configuration
```

### Production Hardening (Recommended)

The current K8s manifests are a starting point. For production, add:

| Resource | Purpose | Status |
|----------|---------|--------|
| HorizontalPodAutoscaler | Auto-scale API pods based on CPU/memory | Not yet implemented |
| PodDisruptionBudget | Ensure availability during updates | Not yet implemented |
| NetworkPolicy | Restrict pod-to-pod communication | Not yet implemented |
| Secrets (Vault) | Secure credential management | Using base64 secrets (upgrade to Vault) |
| Ingress + TLS | External access with HTTPS | Requires cluster-specific config |
| Resource limits | CPU/memory requests and limits | Should be added to deployments |

---

## 6. Monitoring and Observability

### Prometheus

```bash
# Deploy Prometheus
kubectl apply -f deploy/observability/prometheus/

# Or with Docker Compose (included in stack)
# Scrapes /metrics from API on port 8000
```

**Key Metrics:**
- `trading_orders_total` — Order count by status
- `trading_pnl_daily` — Daily P&L
- `trading_risk_utilization` — Risk limit utilization
- `trading_latency_seconds` — Signal-to-order latency
- `trading_circuit_breaker_state` — Circuit breaker status

### Grafana

```bash
# Deploy Grafana
kubectl apply -f deploy/observability/grafana/

# Dashboards are pre-configured in:
# deploy/observability/grafana/dashboards/
```

**Dashboard Panels:**
- Live P&L and equity curve
- Order flow and fill rates
- Risk metrics and limit utilization
- System health (latency, errors, throughput)

### AlertManager

```bash
# Deploy AlertManager
kubectl apply -f deploy/observability/alertmanager/

# Alert rules defined in:
# deploy/observability/prometheus/alerts.yaml
```

**Alert Categories:**
- Risk breaches (drawdown, leverage, concentration)
- System failures (connection drops, high latency)
- Data quality (stale feeds, validation failures)
- Kill switch activations

---

## 7. Nginx Reverse Proxy

```
deploy/nginx/
└── nginx.conf               # Reverse proxy configuration
```

- TLS termination at Nginx layer
- Proxy pass to API (port 8000) and frontend (port 3000)
- WebSocket upgrade support for real-time data
- Rate limiting at proxy layer

---

## 8. Blue-Green Deployment

The platform supports blue-green deployment for zero-downtime releases:

```bash
# Execute blue-green deployment
./deploy/blue_green.sh
```

**Process:**
1. Deploy new version alongside existing (green deployment)
2. Run health checks on new deployment
3. Switch Nginx/Ingress to route traffic to new version
4. Monitor for errors during observation period
5. If healthy, decommission old deployment
6. If errors detected, rollback by switching back to old version

---

## 9. Health Checks

### API Health Endpoint

```
GET /health
```

Returns service status including:
- API server status
- Database connectivity
- Redis connectivity
- Broker connection status (when in live mode)

### Docker Health Check

```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 120s
```

---

## 10. Operational Runbook

### Starting the System

1. Ensure PostgreSQL and Redis are running
2. Run database migrations: `alembic upgrade head`
3. Start API server: `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`
4. Start frontend: `cd trading-ui && npm start`
5. Verify health: `curl http://localhost:8000/health`

### Stopping the System

1. Arm kill switch to prevent new orders: `POST /api/kill-switch/arm`
2. Wait for open orders to complete or cancel
3. Stop API server gracefully (SIGTERM)
4. Kill switch state persists to disk for restart recovery

### Emergency Procedures

| Scenario | Action |
|----------|--------|
| Unexpected losses | Kill switch auto-arms on MAX_DRAWDOWN or MAX_DAILY_LOSS |
| Broker disconnection | Kill switch arms on MARKET_FEED_FAILURE |
| Fill mismatch | Kill switch arms on FILL_MISMATCH, requires manual investigation |
| System crash | State recovered from disk on restart (`startup/recovery.py`) |
| Circuit breaker trip | Automatic cooldown period, then HALF_OPEN testing phase |

### Log Locations

| Log | Location | Notes |
|-----|----------|-------|
| API logs | stdout/stderr | Structured JSON format, PII redacted |
| Training logs | `logs/` volume | ML model training output |
| Audit trail | PostgreSQL `audit_events` table | HMAC-SHA256 hash chain |
| Kill switch state | Disk file (persists across restarts) | JSON format |
| Circuit breaker state | Disk file | JSON format |

### Database Maintenance

```bash
# Run pending migrations
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "description"

# Rollback one step
alembic downgrade -1

# View current migration version
alembic current
```
