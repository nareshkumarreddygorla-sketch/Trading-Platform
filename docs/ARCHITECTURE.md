# Trading Platform Architecture

## 1. Overview

The Trading Platform is an institutional-grade autonomous AI trading system built on Python/FastAPI with a Next.js frontend. It operates across NSE India (primary), with architecture extensible to NYSE/NASDAQ, LSE, and FX markets. The system makes real-time buy/sell decisions using multi-strategy and ML-driven logic, applies comprehensive risk management, and maintains full observability and compliance.

**Design Principles:** Risk-first, modular, event-driven, async-native, horizontally scalable, zero single point of failure, audit-ready.

---

## 2. High-Level Architecture

```
                        ┌───────────────────────────────┐
                        │       External Systems        │
                        │  NSE  NYSE  LSE  FX  Angel    │
                        └──────────┬────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     Market Data Layer        │
                    │  Connectors → Normalizer     │
                    │  → Redis Cache + Bar Cache   │
                    │  → PostgreSQL (OHLCV repo)   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    Feature Engine (75+)      │
                    │  Price, Technical, Micro,    │
                    │  Regime, Cross-asset         │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Strategy / AI Layer        │
                    │  Classical + ML Ensemble     │
                    │  → Signal Generation         │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │       Risk Engine            │
                    │  4-Level Checks + VaR/CVaR   │
                    │  Circuit Breaker + Kill SW   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    Execution Engine           │
                    │  Order Entry (14-step)        │
                    │  → Router → Gateway → Broker │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
    ┌─────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
    │   Persistence    │ │  Observability  │ │   Compliance    │
    │  PostgreSQL +    │ │  Prometheus +   │ │  Audit Trail +  │
    │  Redis + Kafka   │ │  Grafana        │ │  Surveillance   │
    └──────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## 3. Module Map

The `src/` directory contains 300+ Python modules organized into 20+ packages:

```
src/
├── core/                    # Domain primitives and configuration
│   ├── events.py            # Bar, Tick, Signal, Exchange, OrderBookSnapshot
│   ├── config.py            # Pydantic settings (AppConfig)
│   └── operational_runbook.py
│
├── api/                     # FastAPI application layer (36 files)
│   ├── app.py               # FastAPI app factory
│   ├── auth.py              # JWT authentication (access + refresh tokens)
│   ├── middleware.py         # Security headers, CORS, request logging
│   ├── token_blacklist.py   # Revoked token tracking
│   ├── ws_manager.py        # WebSocket connection manager
│   ├── logging_config.py    # Structured logging with PII redaction
│   ├── lifespan/            # App startup/shutdown lifecycle
│   └── routers/             # 26 REST API routers
│       ├── auth.py          # Registration, login, logout, refresh
│       ├── orders.py        # Order submission and management
│       ├── positions.py     # Position tracking
│       ├── risk.py          # Risk metrics and configuration
│       ├── broker.py        # Broker connection management
│       ├── market_data.py   # Market data endpoints
│       ├── strategies.py    # Strategy management
│       ├── backtest.py      # Backtesting API
│       └── ... (18 more)
│
├── market_data/             # Market data ingestion and caching (11 files)
│   ├── feed_manager.py      # Central feed orchestrator
│   ├── angel_one_ws_connector.py  # Angel One WebSocket client
│   ├── bar_aggregator.py    # Tick → OHLCV bar aggregation
│   ├── bar_cache.py         # Rolling window bar cache
│   ├── adv_cache.py         # Average daily volume cache
│   ├── normalizer.py        # Unified schema normalization
│   ├── yfinance_fallback_feeder.py  # Yahoo Finance fallback
│   └── connectors/          # Exchange-specific connectors
│
├── ai/                      # AI/ML intelligence layer (100+ files)
│   ├── feature_engine.py    # 75+ feature computation
│   ├── alpha_model.py       # Alpha signal generation
│   ├── models/              # ML model implementations
│   │   ├── xgboost_predictor.py
│   │   ├── lstm_predictor.py
│   │   ├── transformer_predictor.py
│   │   ├── rl_agent.py
│   │   └── sentiment_predictor.py
│   ├── alpha_research/      # Alpha research pipeline
│   │   ├── pipeline/        # Walk-forward validation
│   │   ├── hypothesis/      # Signal hypothesis testing
│   │   ├── scoring/         # Alpha scoring
│   │   └── decay/           # Alpha decay analysis
│   ├── regime/              # Market regime detection (HMM)
│   ├── self_learning/       # Continuous model improvement
│   ├── walk_forward/        # Walk-forward optimization
│   ├── calibration/         # Model calibration
│   ├── drift/               # Feature/concept drift detection
│   ├── llm/                 # LLM integration for research
│   ├── meta_allocator/      # Strategy allocation optimization
│   └── portfolio_control/   # Portfolio-level controls
│
├── strategy_engine/         # Trading strategy framework (10 files)
│   ├── base.py              # Strategy base class
│   ├── registry.py          # Strategy discovery and registration
│   ├── runner.py            # Strategy execution on each bar
│   ├── allocator.py         # Capital allocation across strategies
│   ├── classical.py         # EMA, MACD, RSI, Bollinger strategies
│   ├── momentum_breakout.py # Breakout and trend following
│   ├── mean_reversion.py    # Mean reversion strategies
│   ├── ml_strategies.py     # ML-driven signal strategies
│   └── high_winrate.py      # Conservative high-probability strategies
│
├── risk_engine/             # Risk management (12 files)
│   ├── manager.py           # RiskManager (15-step can_place_order)
│   ├── limits.py            # Risk limit definitions
│   ├── circuit_breaker.py   # Circuit breaker state machine
│   ├── var.py               # VaR (parametric, historical, Monte Carlo)
│   ├── stress_testing.py    # Scenario-based stress tests
│   ├── gap_risk.py          # Overnight/weekend gap risk
│   ├── vol_targeting.py     # Volatility targeting
│   ├── tail_risk.py         # Tail risk analysis
│   ├── correlation.py       # Correlation monitoring
│   └── sector_map.py        # Sector classification
│
├── execution/               # Order execution pipeline (30+ files)
│   ├── order_entry/
│   │   └── service.py       # OrderEntryService (14-step pipeline)
│   ├── order_router.py      # Order routing logic
│   ├── autonomous_loop.py   # Main autonomous trading loop
│   ├── angel_one_gateway.py # Angel One broker gateway
│   ├── zerodha_gateway.py   # Zerodha broker gateway
│   ├── lifecycle.py         # Order state machine
│   ├── market_impact.py     # Market impact estimation
│   ├── circuit_limits.py    # Exchange circuit limit checks
│   ├── algorithms/          # Execution algorithms (TWAP, VWAP)
│   ├── broker/              # Broker abstraction layer
│   ├── fill_handler/        # Fill processing
│   ├── gateways/            # Gateway abstractions
│   ├── quality/             # Execution quality metrics
│   └── reconciliation/      # Position reconciliation
│
├── persistence/             # Data access layer (17 files)
│   ├── database.py          # SQLAlchemy async engine
│   ├── models.py            # ORM models (User, Order, Position, etc.)
│   ├── order_repo.py        # Order repository
│   ├── position_repo.py     # Position repository
│   ├── ohlcv_repo.py        # OHLCV data repository
│   ├── trade_store.py       # Trade storage
│   └── ... (11 more repos)
│
├── compliance/              # Regulatory compliance (5 files)
│   ├── audit_trail.py       # HMAC-SHA256 hash chain audit log
│   ├── surveillance.py      # Trade surveillance
│   ├── otr_monitor.py       # Order-to-trade ratio monitoring
│   └── retention.py         # Data retention policies
│
├── backtesting/             # Backtesting engine (6 files)
│   ├── engine.py            # Backtest runner
│   ├── metrics.py           # Performance metrics (Sharpe, drawdown)
│   ├── slippage.py          # Slippage models
│   └── dynamic_slippage.py  # Adaptive slippage estimation
│
├── data_pipeline/           # Data quality and ingestion (7 files)
│   ├── tick_validator.py    # Tick data validation
│   ├── ohlc_validator.py    # OHLCV bar validation
│   ├── data_quality_monitor.py  # Staleness and anomaly detection
│   ├── fii_dii_flow.py      # FII/DII institutional flow tracker
│   └── news_aggregator.py   # News feed aggregation
│
├── risk/                    # Risk utilities
├── agents/                  # AI agent framework (5 files)
│   ├── execution_agent.py   # Trade execution agent
│   ├── research_agent.py    # Research automation agent
│   ├── risk_agent.py        # Risk monitoring agent
│   └── strategy_selector.py # Strategy selection agent
│
├── monitoring/              # Metrics and observability
│   └── metrics.py           # Prometheus metric definitions
│
├── reporting/               # Report generation
│   ├── daily_report.py      # Daily P&L and performance reports
│   └── performance_attribution.py  # Attribution analysis
│
├── options/                 # Options pricing
│   ├── chain.py             # Option chain management
│   └── greeks.py            # Greeks calculation
│
├── scanner/                 # Market scanning
│   ├── market_scanner.py    # Technical scanner
│   ├── dynamic_universe.py  # Dynamic stock universe
│   └── nse_universe.py      # NSE stock universe
│
├── simulation/              # Simulation infrastructure
│   ├── nightly_simulator.py # Nightly simulation runner
│   └── orchestrator.py      # Simulation orchestration
│
├── feature_store/           # Feature store
│   ├── schema.py            # Feature definitions
│   └── store.py             # Feature storage and retrieval
│
├── costs/                   # Cost modeling
│   └── india_costs.py       # India-specific trading costs
│
├── alerts/                  # Alert system
│   └── notifier.py          # Alert notification
│
├── startup/                 # Application startup
│   └── recovery.py          # State recovery on restart
│
├── marketplace/             # Strategy marketplace
│   ├── models.py            # Marketplace data models
│   └── service.py           # Marketplace service
│
└── strategies/              # Strategy implementations
```

---

## 4. Component Details

### 4.1 Market Data Layer

| Component | File | Responsibility |
|-----------|------|----------------|
| Feed Manager | `market_data/feed_manager.py` | Orchestrates all data sources, manages connections |
| Angel One WS | `market_data/angel_one_ws_connector.py` | Real-time WebSocket tick/quote stream |
| Bar Aggregator | `market_data/bar_aggregator.py` | Aggregates ticks into 1m/5m/1h/1d bars |
| Bar Cache | `market_data/bar_cache.py` | In-memory rolling window of recent bars |
| ADV Cache | `market_data/adv_cache.py` | Average daily volume for position sizing |
| Normalizer | `market_data/normalizer.py` | Unified schema, UTC timestamp normalization |
| Yahoo Fallback | `market_data/yfinance_fallback_feeder.py` | Fallback historical data source |

**Contracts:** `Bar`, `Tick`, `OrderBookSnapshot` defined in `core/events.py`.

### 4.2 AI/ML Intelligence Layer

The AI layer produces trading signals through a multi-model ensemble:

```
Market Data
    │
    ▼
Feature Engine (75+ features)
    ├── Price: returns, volatility, momentum
    ├── Technical: RSI, MACD, Bollinger, ATR
    ├── Microstructure: spread, volume profile
    ├── Regime: HMM state, volatility regime
    └── Cross-asset: sector correlation
    │
    ▼
Model Ensemble
    ├── XGBoost     (direction prediction)
    ├── LSTM        (sequence patterns)
    ├── Transformer (attention-based)
    ├── RL Agent    (entry/exit optimization)
    └── Sentiment   (news/social analysis)
    │
    ▼
Weighted Ensemble → Signal
```

**Alpha Research Pipeline:** Hypothesis generation, walk-forward validation, alpha scoring, decay analysis, capacity estimation.

**Self-Learning:** Continuous model retraining, drift detection, performance tracking, calibration.

### 4.3 Strategy Engine

- **Plugin Architecture:** Each strategy implements `generate_signals(market_state) -> List[Signal]`
- **Classical:** EMA crossover, MACD, RSI, Bollinger Bands, ATR, breakout, mean reversion
- **ML-Based:** Model-driven signals from the AI ensemble
- **Execution Flow:** Strategy Registry → Runner → Signal Generation → Allocator → Risk Gate → Order Entry

### 4.4 Risk Engine

Four-level risk check hierarchy:

| Level | Scope | Checks |
|-------|-------|--------|
| L1 | Order | Quantity > 0, price > 0, single trade loss, position size % |
| L2 | Portfolio | Open positions, daily loss, intraday rolling loss, consecutive losses, per-symbol concentration |
| L3 | Systemic | Sector concentration, leverage (200% cap), circuit breaker, kill switch |
| L4 | Advanced | VaR (parametric/historical/Monte Carlo), CVaR, gap risk, vol targeting, stress scenarios |

### 4.5 Execution Engine

The 14-step `OrderEntryService.submit_order()` pipeline is the single mandatory entry point for all orders. See `docs/trading_system_design.md` for the full pipeline specification.

**Broker Gateways:**
- Angel One (India primary) — SmartAPI REST + WebSocket
- Zerodha (India secondary) — Kite Connect
- Paper Simulator — for testing with identical risk logic

### 4.6 Persistence Layer

- **PostgreSQL:** Users, orders, positions, trades, audit events, OHLCV history
- **Redis:** Real-time caches (bars, quotes, session state, rate limiters)
- **Kafka:** Event streaming for order events and market data
- **ORM:** SQLAlchemy async with Alembic migrations

### 4.7 Compliance

- **Audit Trail:** HMAC-SHA256 hash chain for tamper detection
- **Surveillance:** Trade pattern monitoring
- **OTR Monitor:** Order-to-trade ratio compliance
- **Retention:** Regulatory data retention with archival

---

## 5. Infrastructure

### Docker

- `Dockerfile.api` — Multi-stage Python build, non-root user, tini as PID 1
- `Dockerfile.frontend` — Next.js production build
- `docker-compose.yaml` — Full stack: API, frontend, Redis, PostgreSQL, Kafka

### Kubernetes

- Namespace isolation (`trading`)
- ConfigMap for non-sensitive configuration
- Deployment manifests for API service
- Secrets for credentials (base64, should use Vault in production)

### Observability

- **Prometheus:** Metrics scraping from `/metrics` endpoint
- **Grafana:** Dashboards for P&L, orders, risk, system health
- **AlertManager:** Alerts for risk breaches, system failures, SLA violations

### Deployment

- Blue-green deployment strategy (`deploy/blue_green.sh`)
- Nginx reverse proxy with TLS termination
- Health check endpoints for liveness/readiness

---

## 6. Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend Language | Python 3.11+ (async/await) |
| API Framework | FastAPI + Uvicorn |
| Frontend | Next.js 14, React, TypeScript, Tailwind CSS |
| State Management | Zustand + React Query |
| Database | PostgreSQL (via SQLAlchemy async + Alembic) |
| Cache | Redis 7 |
| Event Streaming | Kafka |
| ML/AI | XGBoost, PyTorch (LSTM/Transformer), scikit-learn |
| Auth | JWT (PyJWT) + bcrypt (passlib) + TOTP (pyotp) |
| Containers | Docker + Kubernetes |
| CI/CD | GitHub Actions |
| Linting | Ruff (lint + format) |
| Security Scanning | Bandit + pip-audit |
| Observability | Prometheus + Grafana + AlertManager |

---

## 7. Data Flows

### Real-time Trading Path
```
Exchange → WS Connector → Normalizer → Redis Cache + Bar Cache
    → Feature Engine → AI Ensemble → Signal
    → Risk Gate → Order Entry Pipeline (14 steps)
    → Router → Broker Gateway → Exchange
```

### Historical Data Path
```
REST API / Yahoo Finance → Normalizer → OHLCV Repo (PostgreSQL)
    → Feature Store → Backtesting Engine → Performance Metrics
```

### Control Path
```
Frontend (Next.js) → FastAPI REST API
    → Strategy Management (start/stop/configure)
    → Risk Configuration (limits, circuit breaker)
    → Kill Switch (arm/disarm)
    → Monitoring (metrics, positions, P&L)
```

---

## 8. Testing Infrastructure

- **49 test files** with **715+ tests** passing on Python 3.11 and 3.12
- Unit tests, integration tests, stress/concurrency tests, chaos tests
- Shared fixtures for state isolation (`conftest.py`)
- CI runs full suite on every PR with coverage reporting
- See `docs/development_process.md` for testing standards
