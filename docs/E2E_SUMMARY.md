# End-to-End Summary — What We Built

This document describes the **full end-to-end flow** of the Autonomous Trading Platform: from market data ingestion through feature engineering, ML prediction, regime detection, capital allocation, risk checks, execution, backtesting, and autonomous improvement. All paths are risk-first; no AI decision bypasses the risk engine.

---

## 1. Domain & Configuration

**Location:** `src/core/`

- **events.py** — Domain types used across the stack:
  - **Market:** `Bar` (OHLCV, symbol, exchange, interval, ts), `Tick`, `OrderBookSnapshot` (bid/ask, levels).
  - **Strategy/Risk:** `Signal` (strategy_id, symbol, exchange, side, score, portfolio_weight, risk_level, price, stop_loss, target, ts), `Position`, `Order` (order_id, status, filled_qty, avg_price, etc.).
  - **Exchange:** enum NSE, BSE, NYSE, NASDAQ, LSE, FX.
- **config.py** — Central config loading (YAML/env).

**Flow:** All components share these types. Config drives limits, broker credentials, and feature/model settings.

---

## 2. Market Data Layer

**Location:** `src/market_data/`

- **Connectors** — Base connector interface; **Angel One** connector (WebSocket/REST) for NSE India. Exchange → raw ticks/bars.
- **Normalizer** — Converts exchange-specific format to unified `Tick` / `Bar`; time-sync to UTC.
- **Cache** — Real-time L1 (latest quote, order book snapshot); Redis-ready.
- **Streaming** — Publishes tick/bar events to strategies; Kafka/NATS-ready.
- **Historical** — Bars and ticks persisted (TimescaleDB/ClickHouse per docs).

**E2E role:** External exchanges (NSE first) → Connectors → Normalizer → Redis cache + Kafka → consumed by strategy engine and feature pipeline. Historical bars feed backtesting and ML training.

---

## 3. Feature Engineering & Feature Store

**Location:** `src/ai/feature_engineering/`, `src/feature_store/`

- **Specs** — Versioned `FeatureSpec` (name, dtype, version, group). Groups: **PRICE**, **MICROSTRUCTURE**, **REGIME**, **CROSS_ASSET**, **INTRADAY_ALPHA**.
- **Price** — Returns (1m/5m/15m/1h), rolling vol, ATR, BB width, momentum, z-score.
- **Microstructure** — Order flow imbalance, bid-ask spread, volume delta, VWAP deviation, liquidity pressure.
- **Regime** — Vol clustering, Hurst, trend strength, market correlation, sector dispersion.
- **Cross-asset** — Index correlation, India VIX, USDINR impact, global spillover.
- **Intraday alpha (blueprint)** — OFI 5/15/30, POC/value-area deviation, minute_of_day sin/cos, VWAP curvature, liquidity vacuum, vol-of-vol, microstructure noise residual, spread widening signal, cross-stock dispersion, sector momentum rank (specs defined; compute can be wired in pipeline).
- **Pipeline** — Composes feature groups from bars/ticks and writes to feature store (versioned).

**E2E role:** Bars/ticks → feature pipeline → feature store. Stored features are the input to ML models, regime classifier, and (optionally) strategy logic.

---

## 4. Label Engineering (Alpha Evolution)

**Location:** `src/ai/labels/`

- **TripleBarrierLabeler** — For each bar t: look forward until first touch of upper barrier (profit target), lower barrier (stop loss), or time limit H. Label +1 / -1 / 0. Volatility-adjusted barriers; cost-aware barriers via `cost_aware_barriers()` (round-trip cost). No lookahead in features.
- **MetaLabeler** — Given primary prediction (e.g. +1/-1), meta-label = 1 if realized trade after cost was profitable, else 0. Used to filter or size primary signals.
- **cost_aware.py** — `cost_aware_barriers(price, b_u, b_d, cost_pct)`, `net_return(gross, cost)`.

**E2E role:** Price series (and optional primary predictions) → triple-barrier / meta-labels. Labels feed ML training (replace simple next-bar direction); meta-labels can drive confidence for sizing.

---

## 5. ML Prediction Engine

**Location:** `src/ai/models/`

- **Base** — Model contract: `fit()`, `predict(features, context) -> PredictionOutput(prob_up, expected_return, confidence, model_id, version, metadata)`.
- **Registry** — Register models by id; track versions and performance; `replace_if_better(model_id, candidate, candidate_metrics)` for promotion.
- **XGBoost** — Direction/return predictor (fit on features, predict prob_up / expected_return).
- **LSTM** — Stub for sequence model.
- **Vol predictor** — Volatility forecast.
- **Ensemble** — Weighted combination of base models; optional **calibrator** (Platt/Isotonic) applied to `prob_up` after aggregation; NaN/Inf sanitization and clipping.

**E2E role:** Feature store (and optional context) → model registry → XGB/LSTM/Vol → ensemble → calibrated `prob_up`, `expected_return`, `confidence`. Output feeds meta-allocator and (via strategy layer) signals.

---

## 6. Regime Detection

**Location:** `src/ai/regime/`

- **Classifier** — Combines vol percentile, trend strength, optional HMM → regime label (e.g. trending_up, sideways, high_vol).
- **HMM** — Hidden Markov Model for regime states.
- **Volatility regime** — Vol-based regime features.

**E2E role:** Features (and optionally raw series) → regime classifier → regime label. Regime drives meta-allocator (regime_multiplier), position sizing (regime-adjusted), and optionally strategy activation.

---

## 7. Meta-Allocator & Position Sizing

**Location:** `src/ai/meta_allocator/`, `src/ai/position_sizing/`

- **MetaAllocator** — Tracks per-strategy performance (Sharpe, win rate, drawdown, confidence); decay detection; allocates capital across strategies via risk-parity, Kelly, or confidence-weighted. **Optional:** `current_drawdown_pct`, `regime_multiplier`, `meta_alpha_scale` in `allocate()`:
  - **Dynamic position sizing** — When `position_sizing.dynamic_position_fraction` is available, each strategy weight is scaled by confidence × drawdown_scale × regime_multiplier (Kelly-style fraction), then renormalized.
  - **Meta-alpha scale** — When `meta_alpha_scale < 1` (e.g. meta-model says “reduce”), all weights scaled by `meta_alpha_scale` and renormalized.
- **Position sizing** — `dynamic_position_fraction(p_win, win_loss_ratio, confidence, current_drawdown_pct, regime_multiplier)` → fraction of capital; `volatility_target_notional(capital, sigma_forecast, sigma_target)`; `kelly_binary(p_win, win_loss_ratio)`.

**E2E role:** Strategy IDs + equity + (optional) drawdown/regime/meta_alpha → MetaAllocator.allocate() → per-strategy weights (and enabled/disabled). Weights and sizing drive how much capital each strategy can use; combined with risk limits to determine order size.

---

## 8. Meta-Alpha Layer

**Location:** `src/ai/meta_alpha/`

- **MetaAlphaPredictor** — Predicts P(primary wrong), P(confidence inflated), P(regime flip). Outputs recommendation: `reduce_size` | `filter_signal` | `hold`. Heuristic mode when no model loaded; `update(primary_correct, confidence, realized_hit)` for online tracking.

**E2E role:** Primary prediction + confidence + regime → meta-alpha → recommendation and probabilities. Controller converts to `meta_alpha_scale` for MetaAllocator; used to down-weight or filter when primary model is likely wrong.

---

## 9. Risk Engine

**Location:** `src/risk_engine/`

- **Limits** — `RiskLimits`: max_position_pct, max_daily_loss_pct, max_open_positions, circuit_breaker_drawdown_pct, etc. `check_position_size(equity, position_value)`, `check_daily_loss(equity, daily_pnl)`, `check_open_positions(count)`. Rejects when equity ≤ 0.
- **Manager** — `RiskManager(equity, limits)`: maintains `positions`, `daily_pnl`; `update_equity()`, `register_pnl()`, `add_position()`, `remove_position()`; `open_circuit()` / `close_circuit()`; `is_circuit_open()`. **Exposure multiplier (LLM):** `_exposure_multiplier` (0.5–1.5), `set_exposure_multiplier(mult)`, `effective_equity() = equity × multiplier`. All position-size checks and `max_quantity_for_signal()` use effective equity.
- **can_place_order(signal, quantity, price)** — Returns LimitCheckResult: checks circuit, equity, daily loss, open positions, position size (vs effective equity). Rejects with reason on breach.
- **Circuit breaker** — Opens on drawdown/limit breach; blocks new orders until closed.
- **Metrics** — VaR, CVaR, Sharpe, max drawdown, Kelly (for reporting).

**E2E role:** Every order path must pass RiskManager. Equity and exposure multiplier set capacity; positions and PnL updated from execution (FillHandler). Circuit breaker and limits enforce capital preservation.

---

## 10. Execution Layer (Single Order Entry)

**Location:** `src/execution/`, `src/execution/order_entry/`, `src/execution/fill_handler/`, `src/execution/reconciliation/`

- **OrderEntryService** — **Single mandatory entry** for all orders (API, AI, manual). Pipeline:
  1. Validate input (quantity, price, symbol, side).
  2. Idempotency check (Redis key → return existing result if duplicate).
  3. Kill-switch check (if armed → allow only reduce-only orders).
  4. Circuit breaker check (`risk_manager.is_circuit_open()`).
  5. `risk_manager.can_place_order(signal, quantity, price)`.
  6. Atomic exposure reservation (`ExposureReservation.reserve` under global lock).
  7. `OrderRouter.place_order(signal, quantity, ...)` → broker gateway.
  8. `OrderLifecycle.register(order)`.
  9. Persist order (optional callback).
  10. Publish to Kafka (optional callback).
  11. Return OrderEntryResult (success, order_id, reject_reason, latency_ms).
- **OrderEntryRequest/Result** — Request: signal, quantity, order_type, limit_price, idempotency_key, source. Result: success, order_id, broker_order_id, reject_reason, reject_detail, latency_ms.
- **IdempotencyStore** — Redis-backed; TTL 48h; duplicate requests return stored order_id without calling broker.
- **KillSwitch** — Armed/disarmed; when armed, only reduce-only orders allowed; `allow_reduce_only_order(state, symbol, side, qty, net_position)`.
- **ExposureReservation** — Reserve before broker call; commit on success; release on failure so pending orders consume capacity.
- **OrderRouter** — Routes to Angel One (or other) gateway; paper/live.
- **OrderLifecycle** — Tracks order state (NEW → FILLED/CANCELLED/REJECTED).
- **FillHandler** — On broker fill: update lifecycle; `risk_manager.add_position()` for new fill; for close, `remove_position()` + `register_pnl(realized_pnl)`.
- **Reconciliation** — Periodic: fetch broker positions vs risk_manager.positions; on mismatch → log, metric, optionally arm kill switch (FILL_MISMATCH).
- **CircuitAndKillController** — Daily loss, drawdown, rejection spike, fill mismatch → arm kill switch / open circuit.
- **Resilience** — BrokerGatewayResilient wrapper (retries, backoff).

**E2E role:** Strategy/AI produces Signal + quantity. Only path to broker is OrderEntryService.submit_order(request). Risk, idempotency, kill switch, and reservation enforce safety; FillHandler and reconciliation keep risk state in sync with broker.

---

## 11. Backtesting

**Location:** `src/backtesting/`

- **Engine** — Historical bars → run strategies → simulate orders; equity curve, PnL. **Uses FillModel:** no same-bar fill; fill at bar i+latency_bars.
- **FillModel** — `FillModelConfig`: latency_bars, slippage_bps, spread_bps, max_volume_participation_pct, commission_pct. `execute_at_bar_index(signal_bar_index, bars, side, requested_qty, price_hint)` → (fill_bar, fill_price, fill_qty, commission). Realistic execution for backtest.
- **Slippage** — SlippageModel (bps-based).
- **Metrics** — Sharpe, max drawdown, win rate, etc.

**E2E role:** Historical data → strategy/ML logic → simulated orders via FillModel → equity and metrics. Used for strategy validation and for retrain pipeline (walk-forward backtest → replace if better).

---

## 12. Self-Learning & Drift

**Location:** `src/ai/self_learning/`, `src/ai/drift/`, `src/ai/walk_forward/`

- **ConceptDriftDetector / DataDistributionMonitor** — Feature stats (mean/std) vs reference; PSI/batch drift; triggers retrain.
- **RetrainPipeline** — Train (train_fn) → backtest (backtest_fn) → replace_if_better. **Optional walk-forward:** `walk_forward_backtest_fn(model)` returns (mean_sharpe, mean_dd, sharpes_per_window, max_dds_per_window). If `use_walk_forward`, compute stability_score and apply replacement_rule (Sharpe better, drawdown not worse, stability ≥ threshold, N consecutive positive windows); replace only if rule passes.
- **SelfLearningOrchestrator** — run_cycle(current_features): monitor distribution, check drift, if drift run retrain pipelines; on_retrain_complete callback.
- **MultiLayerDriftDetector** — Prediction distribution (PSI), calibration (MSE), Sharpe drop, feature importance (cosine). set_reference(); run_all() → list of DriftSignal. Complements concept drift.
- **Walk-forward** — stability_score(sharpes, max_dds, dd_limit, min_frac_positive); replacement_rule(current_sharpe, current_dd, candidate_sharpe, candidate_dd, stability, consecutive_positive, config).

**E2E role:** Live features → drift check → optional retrain → walk-forward backtest → replace model only if stability and replacement rule pass. Keeps models aligned with current market regime and reduces overfitting.

---

## 13. LLM Layer (Advisory Only)

**Location:** `src/ai/llm/`

- **LLMClient / LLMConfig** — OpenAI/Claude completion.
- **NewsSentimentService** — News text → sentiment, score, risk_reduction_suggestion, reason (JSON).
- **MacroRiskService** — Macro/event text → risk view.
- **StrategyReviewService** — Performance summary → LLM review.
- **AdvisoryService** — Multi-source aggregation (list of {source, text, timestamp}) → **event_severity** (low/medium/high), **exposure_multiplier** (0.5–1.5), reason, sources_cited. Guardrails: multiplier clamped [0.5, 1.5]; optional source requirement for extreme multiplier. **LLM does not place trades.**

**E2E role:** News/macro/strategy summary → LLM → sentiment/macro/review + advisory (event severity, exposure_multiplier). Controller applies exposure_multiplier to RiskManager; effective equity scales position caps.

---

## 14. Autonomous Controller & Trading Readiness

**Location:** `src/ai/autonomous/`, `src/api/routers/trading.py`

- **AutonomousTradingController** — `apply_llm_advisory(exposure_multiplier)` → risk_manager.set_exposure_multiplier(mult). `meta_alpha_scale_for_allocator(recommendation, prob_primary_wrong)` → scale for MetaAllocator.allocate(meta_alpha_scale=...): reduce_size → (1 − prob_primary_wrong); filter_signal → 0.5; hold → 1.0.
- **Trading readiness** — `GET /api/v1/trading/ready`: 200 only when OrderEntryService configured, kill switch NOT armed, circuit NOT open, equity > 0; else 503 with reason. Use for K8s readiness.
- **Exposure multiplier API** — `PUT /api/v1/trading/exposure_multiplier` with body `{"multiplier": 0.5 | 1.0 | 1.5}` → sets risk_manager.set_exposure_multiplier (for LLM advisory integration).

**E2E role:** Central place to apply LLM exposure and meta-alpha scale. Readiness ensures traffic only hits when the system is allowed to trade; exposure API allows external (e.g. LLM pipeline) to set multiplier.

---

## 15. Calibration & Objectives

**Location:** `src/ai/calibration/`, `src/ai/objectives/`

- **PlattCalibrator / IsotonicCalibrator** — Fit on validation (p, y); transform raw prob to calibrated prob. **reliability_curve(y_true, y_prob, n_bins)** — binned predicted prob vs realized frequency (monitoring).
- **Ensemble** — Optional calibrator applied to prob_up after aggregation (see §5).
- **risk_adjusted_reward(returns, drawdown_contrib, turnover, config)** — J = E[R] − λ1×DD − λ2×Turnover − λ3×σ. **sharpe_like_score(returns)** — mean/(std+ε) for validation/hyperparameter search.

**E2E role:** Calibration improves probability quality for sizing and meta-allocator. Objectives define risk-adjusted targets for training and evaluation.

---

## 16. API & Observability

**Location:** `src/api/`, `src/monitoring/`, `deploy/observability/`

- **FastAPI app** — Health, market (quote/bars), strategies (enable/disable, signals), risk (state, limits), orders (place via OrderEntryService), backtest (run/jobs), **trading (ready, exposure_multiplier)**. Lifespan builds RiskManager, OrderRouter, Lifecycle, IdempotencyStore, KillSwitch, ExposureReservation, OrderEntryService; attaches to app.state.
- **Orders** — POST /orders → OrderEntryRequest from body → OrderEntryService.submit_order(request) → OrderEntryResult.
- **Metrics** — /metrics Prometheus scrape; counters/gauges for orders, fills, risk, latency; execution alerts (e.g. alerts_execution.yaml).

**E2E role:** Single API surface for dashboards, algo clients, and K8s probes. All order placement goes through the same risk-gated execution path; readiness and metrics support safe autonomous operation.

---

## 17. End-to-End Data Flow (Summary)

1. **Market data** — Exchange → Connectors → Normalizer → Redis/Kafka + historical DB.
2. **Features** — Bars/ticks → feature pipeline → feature store (price, micro, regime, cross-asset, intraday alpha specs).
3. **Labels** — Price (and optional primary) → triple-barrier / meta-labeler → labels for training and meta confidence.
4. **ML** — Features → registry (XGB/LSTM/Vol) → ensemble → optional calibrator → prob_up, expected_return, confidence.
5. **Regime** — Features → regime classifier → regime label.
6. **Allocation** — Strategy IDs + equity + drawdown/regime/meta_alpha_scale → MetaAllocator (dynamic sizing + meta_alpha scale) → per-strategy weights.
7. **Signals** — Strategies (classical + ML) produce Signal with score, weight, risk_level; quantity can be derived from weight × effective equity and risk limits.
8. **Risk** — Every order: RiskManager.can_place_order (effective equity, limits, circuit); max_quantity_for_signal uses effective_equity × max_position_pct.
9. **Execution** — Signal + quantity → OrderEntryRequest → OrderEntryService (validate → idempotency → kill switch → circuit → risk → reserve → router → lifecycle) → broker; fills → FillHandler → risk positions/PnL; reconciliation → broker vs risk.
10. **Autonomous** — LLM advisory → exposure_multiplier → RiskManager; meta_alpha → meta_alpha_scale → MetaAllocator; drift → retrain → walk-forward → replace if stability rule passes; trading readiness → 200 only when safe to trade.

---

## 18. Documentation Index

| Document | Purpose |
|----------|---------|
| README.md | Overview, structure, quick start, API/Deploy, AI layer, autonomous wiring |
| docs/ARCHITECTURE.md | High-level diagram, market data, strategy, risk, execution, backtest, observability |
| docs/API.md | Health, market, strategies, risk, orders, backtest, trading readiness, exposure multiplier |
| docs/DEPLOYMENT.md | Docker, K8s, CI/CD |
| docs/AI_ARCHITECTURE.md | AI layer flow, component map, risk-first |
| docs/AI_DEPLOYMENT.md | AI deployment and ops |
| docs/EXECUTION_REDESIGN.md | Single order entry, idempotency, kill switch, reservation, fill handler, reconciliation, fill model |
| docs/SECURITY_AUDIT.md | 10-phase audit and fixes |
| docs/ALPHA_EVOLUTION_BLUEPRINT.md | Label engineering, modeling, features, sizing, drift, walk-forward, meta-alpha, objective, LLM, roadmap |
| docs/E2E_SUMMARY.md | This document — full E2E flow |

---

*This platform is designed for institutional-grade autonomous trading: risk-first, single order entry, calibrated ML, dynamic sizing, walk-forward model replacement, and LLM/meta-alpha advisory with no direct order placement by AI.*
