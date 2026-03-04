# Autonomous Trading Platform — Institutional Grade

Next-generation **global autonomous trading platform** with:

- **Multi-market**: NSE India (first), NYSE/NASDAQ, LSE, FX
- **Real-time** buy/sell decisions with multi-strategy + ML
- **Risk-first**: VaR, Kelly, drawdown limits, circuit breaker
- **Production-ready**: Async, event-driven, Docker/K8s, observability, compliance hooks

## Architecture

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for:

- High-level architecture diagram (Mermaid)
- Market data layer (connectors, Redis, Kafka, TimescaleDB/ClickHouse)
- Strategy engine (classical + ML, plugin-based)
- Risk engine (VaR, CVaR, Kelly, limits, circuit breaker)
- Execution (Angel One, order router, lifecycle)
- Backtesting, feature store, API, observability

## Project Structure

```
Trading-Platform/
├── docs/                    # ARCHITECTURE, API, DEPLOYMENT
├── config/                  # settings.example.yaml
├── src/
│   ├── core/                # events, config (domain types)
│   ├── market_data/         # connectors, normalizer, cache, streaming
│   ├── strategy_engine/     # base, registry, classical (EMA/MACD/RSI), ML stubs, runner
│   ├── risk_engine/         # metrics (VaR, Kelly), limits, manager, circuit_breaker
│   ├── execution/           # base gateway, Angel One, order router, lifecycle
│   ├── backtesting/         # engine, metrics, slippage
│   ├── feature_store/       # schema, store (ML features)
│   ├── api/                 # FastAPI app, routers (health, market, strategies, risk, orders, backtest)
│   └── monitoring/          # Prometheus metrics
├── tests/                   # pytest (health, strategy, risk)
├── deploy/
│   ├── docker/              # Dockerfile.api, docker-compose
│   ├── k8s/                 # namespace, configmap, api-deployment
│   └── observability/       # prometheus config & alerts
├── frontend/                 # React dashboard (Vite + TypeScript)
├── requirements.txt
└── .github/workflows/ci.yaml
```

## Quick Start

1. **Clone and install**
   ```bash
   cd Trading-Platform
   python3 -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Config**
   ```bash
   cp config/settings.example.yaml config/settings.yaml
   # Edit settings.yaml and .env for broker (Angel One), Redis, Kafka
   ```

3. **Run API**
   ```bash
   export PYTHONPATH=.
   uvicorn src.api.app:app --host 0.0.0.0 --port 8000
   ```
   - Health: http://localhost:8000/health  
   - Docs: http://localhost:8000/docs  
   - Metrics: http://localhost:8000/metrics  

4. **Web application (API + dashboard)**
   - **Terminal 1 — API:** `PYTHONPATH=. uvicorn src.api.app:app --host 0.0.0.0 --port 8000`
   - **Terminal 2 — Frontend:** `cd frontend && npm install && npm run dev`
   - Open **http://localhost:5173** in your browser for the dashboard (health, risk, strategies, alpha research).
   - See **[How to use the web application](#how-to-use-the-web-application)** below.

5. **Tests**
   ```bash
   pytest tests -v
   ```

6. **Docker**
   ```bash
   docker-compose -f deploy/docker/docker-compose.yaml up -d
   ```

## How to use the web application

1. **Start both servers** (two terminals):
   - **Terminal 1** (API): from project root run  
     `PYTHONPATH=. uvicorn src.api.app:app --host 0.0.0.0 --port 8000`  
     Leave it running; you should see “Uvicorn running on http://0.0.0.0:8000”.
   - **Terminal 2** (frontend): run  
     `cd frontend && npm install && npm run dev`  
     Leave it running; you should see “Local: http://localhost:5173/”.

2. **Open the app**  
   In your browser go to **http://localhost:5173**. You’ll see the Trading Platform dashboard.

3. **What you can do on the dashboard**
   - **Refresh** — Top-right button to reload all data from the API.
   - **API Health** — Confirms the backend is up (status: ok).
   - **Trading Ready** — Shows whether the system is allowed to place orders (wired, kill switch off, circuit closed).
   - **Risk State** — Circuit breaker, daily P&L, open positions, VaR, max drawdown.
   - **Risk Limits** — Max position %, max daily loss %, max open positions.
   - **Strategies** — List of strategies (e.g. EMA, MACD, RSI). Use **Enable** / **Disable** to turn each on or off.
   - **Alpha Research** — Click **Run pipeline** to run the alpha research pipeline (generate → validate → score). After it finishes, you’ll see how many candidates were generated, how many passed, and the top-decile signals.

4. **If something doesn’t load**  
   Make sure the API is running on port 8000. The frontend proxies requests to it; if the API is down, cards will show “—” or errors. Use **Refresh** after starting the API.

5. **API docs (optional)**  
   For raw API testing: **http://localhost:8000/docs** (Swagger UI).

## API Contracts

See **[docs/API.md](docs/API.md)** for:

- Health, market quote/bars, strategies (enable/disable, signals), risk state/limits, orders/positions, backtest run/jobs, metrics.

## Deployment

See **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** for:

- Docker Compose (API, Redis, Postgres, Kafka)
- Kubernetes (namespace, ConfigMap, API deployment)
- CI/CD (GitHub Actions), feature flags, blue/green.

## Strategy & Risk Objectives

- **First**: survival and risk control (limits, circuit breaker).
- **Second**: profitability (multi-strategy, backtest, walk-forward).
- ML where it improves decisions; ensemble strategy system; statistical/dynamic stops.
- Target: positive monthly returns, controlled drawdown, realizable Sharpe > 1.5 (goal).

## AI Autonomous Improvement Layer

The platform includes an **AI layer** that makes the system self-improving while staying **risk-first**:

- **Phase 1 — Feature engineering**: Price, microstructure, regime, and cross-asset features; versioned in the feature store (`src/ai/feature_engineering/`).
- **Phase 2 — ML prediction engine**: Model registry, XGBoost/LSTM/Vol predictors, ensemble with calibrated outputs (`src/ai/models/`).
- **Phase 3 — Regime detection**: HMM + volatility + trend → regime label; strategies activate by regime (`src/ai/regime/`).
- **Phase 4 — Meta-allocator**: Per-strategy Sharpe/win rate/drawdown; decay detection; risk-parity/Kelly weights (`src/ai/meta_allocator/`).
- **Phase 5 — Self-learning loop**: Drift detection, retrain pipeline, backtest, replace-if-better (`src/ai/self_learning/`).
- **Phase 6 — LLM layer**: News sentiment, macro risk, strategy review (OpenAI/Claude). **LLM does not place trades** (`src/ai/llm/`).
- **Phase 7 — Risk gate**: All AI signals and parameter changes pass `RiskManager` and `AIRiskGate` (`src/ai/risk_gate.py`).

See **[docs/AI_ARCHITECTURE.md](docs/AI_ARCHITECTURE.md)** and **[docs/AI_DEPLOYMENT.md](docs/AI_DEPLOYMENT.md)** for design and deployment.

**Alpha Research & Edge Discovery:** Automated hypothesis generation, statistical validation (IC, FDR, OOS, walk-forward), quality scoring, clustering, capacity modeling, decay monitoring. Run: `PYTHONPATH=. python3 scripts/run_alpha_research.py` or `POST /api/v1/alpha_research/run`. See `docs/ALPHA_RESEARCH_ENGINE.md`.

**Autonomous trading wiring:**
- **Trading readiness** — `GET /api/v1/trading/ready` returns 200 only when execution path is wired, kill switch disarmed, circuit closed, equity > 0 (use for K8s readiness).
- **LLM exposure** — LLM advisory `exposure_multiplier` (0.5–1.5) is applied via `RiskManager.set_exposure_multiplier()`; effective equity scales position caps.
- **Meta-allocator** — Optional `current_drawdown_pct`, `regime_multiplier`, `meta_alpha_scale` in `allocate()`; dynamic position sizing (confidence × drawdown × regime) and meta-alpha reduce-size scale applied when available.
- **Walk-forward retrain** — `RetrainPipeline` supports optional `walk_forward_backtest_fn` and `use_walk_forward`; replacement only if stability score and replacement rule pass.
- **Ensemble calibration** — Optional Platt/Isotonic calibrator on ensemble output before passing to allocator.

## Compliance

- All orders tagged with `strategy_id` and timestamp (audit).
- Full order/trade and risk trigger logs.
- Config changes auditable.
- Use SEBI-registered brokers (e.g. Angel One) and comply with local regulations.

## License & Disclaimer

This is a **serious, institutional-grade** design and codebase. Trading involves risk. No guarantee of returns. Not legal/financial advice.
