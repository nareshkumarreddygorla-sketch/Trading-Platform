# Development Roadmap

## Current State

The Trading Platform has a comprehensive feature set with 300+ Python modules, 715+ passing tests, full CI/CD pipeline, and a Next.js frontend. The core risk engine and execution pipeline are well-tested and production-grade. The system supports paper and live trading through Angel One with identical risk logic in both modes.

**What works today:**
- Full order execution pipeline (14-step OrderEntryService)
- Risk engine with 15 checks, circuit breaker, and kill switch
- AI/ML model ensemble (XGBoost, LSTM, Transformer, RL, Sentiment)
- 75+ feature engine
- Strategy framework with classical and ML strategies
- JWT authentication with token blacklist
- CI/CD with automated lint, format, test, and security scanning
- Docker and basic Kubernetes deployment
- Prometheus/Grafana observability

---

## Phase 1: Test Coverage Hardening (Cycles 2-4)

**Goal:** Close critical test gaps in untested modules that handle real money and data.

### Cycle 2: Core Loop & Routing Tests

| Task | Target File | Priority |
|------|-------------|----------|
| Unit tests for autonomous trading loop | `src/execution/autonomous_loop.py` | P0 |
| Unit tests for order routing logic | `src/execution/order_router.py` | P1 |
| Tests for data pipeline validators | `src/data_pipeline/tick_validator.py`, `ohlc_validator.py` | P0 |

**Success Criteria:** Autonomous loop has >60% coverage, order routing has >70% coverage, all validator edge cases tested.

### Cycle 3: Market Data & Compliance Tests

| Task | Target File | Priority |
|------|-------------|----------|
| Feed manager reconnection tests | `src/market_data/feed_manager.py` | P0 |
| WebSocket connector error handling | `src/market_data/angel_one_ws_connector.py` | P0 |
| Compliance surveillance tests | `src/compliance/surveillance.py` | P1 |
| OTR monitor tests | `src/compliance/otr_monitor.py` | P1 |
| Data retention tests | `src/compliance/retention.py` | P1 |
| Redis distributed lock tests | `src/execution/redis_distributed_lock.py` (if exists) | P1 |

**Success Criteria:** Market data layer has >50% coverage, all compliance modules have unit tests.

### Cycle 4: Backtesting & API Tests

| Task | Target File | Priority |
|------|-------------|----------|
| Backtesting fill model tests | `src/backtesting/fill_model.py` | P1 |
| Slippage model tests | `src/backtesting/slippage.py`, `dynamic_slippage.py` | P1 |
| API router tests (23 untested) | `src/api/routers/*.py` | P2 |
| Reporting tests | `src/reporting/daily_report.py`, `performance_attribution.py` | P2 |
| Options pricing tests | `src/options/chain.py`, `greeks.py` | P2 |

**Success Criteria:** All backtesting models have unit tests, top 10 API routers have endpoint tests.

---

## Phase 2: Infrastructure Hardening (Cycles 5-7)

**Goal:** Production-ready infrastructure with proper secrets management, scaling, and resilience.

### Cycle 5: Kubernetes Production Readiness

| Task | Description |
|------|-------------|
| HorizontalPodAutoscaler | Auto-scale API pods based on CPU/memory |
| PodDisruptionBudget | Ensure availability during rolling updates |
| Resource limits | CPU/memory requests and limits on all pods |
| NetworkPolicy | Restrict pod-to-pod communication |
| Liveness/readiness probes | Health check configuration in K8s |

### Cycle 6: Secrets Management

| Task | Description |
|------|-------------|
| HashiCorp Vault integration | Replace base64 K8s secrets with Vault |
| Secret rotation | Automated credential rotation for broker APIs |
| Audit logging for secrets access | Track who accessed what credentials |

### Cycle 7: Deployment Automation

| Task | Description |
|------|-------------|
| Blue-green automation | Fully automated blue-green deployment |
| Rollback automation | One-command rollback with health check verification |
| Canary deployment support | Progressive traffic shifting |
| Database migration safety | Migration dry-run and rollback verification |

---

## Phase 3: System Reliability (Cycles 8-10)

**Goal:** Battle-tested reliability for continuous autonomous operation.

### Cycle 8: Resilience Testing

| Task | Description |
|------|-------------|
| Chaos engineering expansion | Extend `tests/chaos/` with broker failure, DB failure, Redis failure scenarios |
| Network partition tests | Simulate network splits between services |
| Graceful degradation tests | Verify system degrades safely under partial failures |
| Recovery time verification | Measure and optimize MTTR (Mean Time To Recovery) |

### Cycle 9: Performance Optimization

| Task | Description |
|------|-------------|
| Signal-to-order latency profiling | Profile and optimize the hot path |
| Database query optimization | Add indexes, optimize slow queries |
| Memory profiling | Identify and fix memory leaks in long-running processes |
| Connection pool tuning | Optimize PostgreSQL, Redis, and broker connection pools |

### Cycle 10: Multi-Broker Support

| Task | Description |
|------|-------------|
| Zerodha gateway implementation | Complete Kite Connect integration |
| Broker failover logic | Automatic failover between brokers |
| FIX protocol gateway | Standard FIX protocol for institutional brokers |
| Broker-agnostic order routing | Smart routing across multiple brokers |

---

## Phase 4: Feature Completeness (Cycles 11-14)

**Goal:** Complete the feature set for a production-grade trading platform.

### Cycle 11: Advanced Risk

| Task | Description |
|------|-------------|
| Real-time VaR dashboard | Live VaR/CVaR computation and visualization |
| Stress testing automation | Scheduled stress tests with historical scenarios |
| Correlation monitoring | Real-time cross-asset correlation alerts |
| Drawdown recovery analysis | Automated analysis of drawdown events |

### Cycle 12: Frontend Completion

| Task | Description |
|------|-------------|
| Real-time position dashboard | Live P&L, positions, and order flow |
| Strategy management UI | Start/stop/configure strategies from frontend |
| Risk configuration UI | Interactive risk limit management |
| Backtesting UI | Run and visualize backtests from frontend |
| Alert management UI | Configure and manage trading alerts |

### Cycle 13: Reporting & Analytics

| Task | Description |
|------|-------------|
| Automated daily reports | Email/Slack daily P&L and performance summary |
| Performance attribution | Strategy-level and factor-level attribution |
| Tax reporting | Generate tax reports for trading activity |
| Regulatory reporting | Compliance reports for regulatory requirements |

### Cycle 14: Multi-Market Expansion

| Task | Description |
|------|-------------|
| NYSE/NASDAQ data connectors | US market data integration |
| Multi-currency support | Handle INR, USD, GBP positions |
| Cross-market strategies | Strategies spanning multiple markets |
| Timezone-aware scheduling | Market-hours-aware strategy scheduling |

---

## Success Metrics

| Metric | Current | Phase 1 Target | Phase 4 Target |
|--------|---------|----------------|----------------|
| Test count | 715+ | 1000+ | 1500+ |
| Code coverage | ~20% (CI minimum) | 50%+ | 80%+ |
| Critical module coverage | ~40% | 80%+ | 95%+ |
| CI pipeline time | ~4 min | <5 min | <5 min |
| API endpoint test coverage | 3/26 routers | 15/26 routers | 26/26 routers |
| Deployment automation | Manual/semi-auto | Automated blue-green | Full GitOps |
| Mean time to recovery | Unknown | <5 min | <2 min |
| Uptime target | N/A | 99% | 99.9% |

---

## Prioritization Principles

1. **Safety first:** Test coverage for money-handling code (execution, risk, orders) before features
2. **Incremental delivery:** Each cycle produces working, tested, deployable code
3. **Risk reduction:** Address highest-risk gaps first (untested critical paths)
4. **Documentation parity:** Every cycle includes documentation updates
5. **CI discipline:** All changes must pass CI before merge; no manual bypasses in production
