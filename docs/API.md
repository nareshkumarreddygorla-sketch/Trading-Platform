# API Contracts

## Overview

REST API is implemented with **FastAPI**. All timestamps are ISO 8601 UTC. All monetary values in account currency.

## Base URL

- Local: `http://localhost:8000`
- Prefix: `/api/v1`

## Common Types

- **StrategyId**: string (e.g. `ema_cross_1m`, `macd_15m`)
- **Symbol**: string, exchange-qualified (e.g. `NSE:RELIANCE`, `NYSE:AAPL`)
- **OrderSide**: `BUY` | `SELL`
- **OrderStatus**: `PENDING` | `LIVE` | `FILLED` | `PARTIALLY_FILLED` | `CANCELLED` | `REJECTED`

## Endpoints

### Health & Readiness

- `GET /health` — Liveness (200 OK)
- `GET /ready` — Readiness (checks DB, Redis, Kafka connectivity)
- `GET /api/v1/trading/ready` — **Trading readiness**: 200 only when execution path wired, kill switch not armed, circuit closed, equity > 0; 503 otherwise (use for K8s readiness probe)
- `PUT /api/v1/trading/exposure_multiplier` — Set risk exposure multiplier from LLM advisory (body: `{"multiplier": 0.5 | 1.0 | 1.5}`); caps at [0.5, 1.5]

### Market Data

- `GET /api/v1/market/quote/{symbol}` — Latest quote (cached)
- `GET /api/v1/market/bars/{symbol}?interval=1m&from=&to=` — OHLCV bars

### Strategies

- `GET /api/v1/strategies` — List registered strategies
- `POST /api/v1/strategies/{strategy_id}/enable` — Enable strategy
- `POST /api/v1/strategies/{strategy_id}/disable` — Disable strategy
- `GET /api/v1/strategies/signals` — Last N signals (for dashboard)

### Risk

- `GET /api/v1/risk/state` — Current risk state (VaR, exposure, limits)
- `GET /api/v1/risk/limits` — Current limit config
- `PUT /api/v1/risk/limits` — Update limits (audit logged)

### Orders & Positions

- `GET /api/v1/orders?status=&strategy_id=&limit=` — List orders
- `GET /api/v1/positions` — Current positions
- `POST /api/v1/orders` — Place order (body: symbol, side, quantity, order_type, strategy_id)

### Backtesting

- `POST /api/v1/backtest/run` — Submit backtest job (body: strategy_id, symbol, start, end, config)
- `GET /api/v1/backtest/jobs/{job_id}` — Job status and result summary
- `GET /api/v1/backtest/jobs/{job_id}/equity` — Equity curve (CSV/JSON)

### Alpha Research

- `POST /api/v1/alpha_research/run` — Trigger one pipeline run (generate → validate → score → cluster → capacity). Runs in background.
- `GET /api/v1/alpha_research/status` — Last run status: idle | running | completed | failed.
- `GET /api/v1/alpha_research/results` — Last run results (candidates_generated, validated_passed, top_decile_ids, top_decile_scores).
- `POST /api/v1/alpha_research/decay_multipliers` — Body: `{"signal_ids": ["id1", ...]}`. Returns recommended weight multipliers from DecayMonitor (for meta_allocator Phase H).

### Observability

- `GET /metrics` — Prometheus scrape endpoint

---

All mutation endpoints return audit trail (who, when, what). Order placement includes `strategy_id` and `timestamp` for compliance.
