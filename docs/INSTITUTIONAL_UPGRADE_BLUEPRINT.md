# Institutional Upgrade Blueprint — 15 Phases

**Scope:** Transform the autonomous intraday AI trading platform into a capital-preserving, cross-sectionally optimized, statistically disciplined alpha engine. All improvements preserve risk-first design; no AI bypasses the risk engine.

---

## Architecture Diagram (Updated)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MARKET DATA LAYER                                       │
│  Exchange → Connectors → Normalizer → Redis/Kafka + Historical DB                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING → FEATURE STORE (price, micro, regime, cross-asset, intraday)│
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          ▼                             ▼                             ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────────────┐
│ REGIME CLASSIFIER   │   │ REGIME SPECIALISTS  │   │ CROSS-SECTIONAL RANKER      │
│ (vol, trend, HMM)   │   │ Trend / MR / Breakout│   │ OpportunityScore per symbol │
│ → regime_id, weight  │   │ / LowLiq Defensive  │   │ Rank → Top N + filters       │
└──────────┬──────────┘   └──────────┬──────────┘   └──────────────┬──────────────┘
           │                          │                             │
           │    ┌─────────────────────┴─────────────────────────────┘
           │    │
           ▼    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ML ENSEMBLE (XGB/LSTM/Vol) + CALIBRATION → prob_up, E[r], confidence            │
│  META-ALPHA → P(wrong), P(regime_flip), P(confidence_inflated) → BEFORE alloc   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  CORRELATION-AWARE PORTFOLIO OPTIMIZER                                            │
│  Rolling corr matrix → MCR, heat, concentration → Risk parity / vol target       │
│  Max gross/net/sector/correlated-cluster exposure; effective_equity              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PORTFOLIO HEAT & DRAWDOWN CONTROL                                                │
│  ExposureScale = max(0.2, 1 - current_dd/dd_limit); heat limit; trade pause      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  RISK ENGINE (limits, circuit, kill switch, exposure_multiplier)                  │
│  can_place_order(signal, qty, price) → LimitCheckResult                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ORDER ENTRY SERVICE (idempotent, reserve, kill switch, circuit, risk)           │
│  LIQUIDITY-AWARE EXECUTION: dynamic slippage, volume participation cap           │
│  EXECUTION QUALITY TRACKER → feedback to sizing / disable / broker failover       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  SHADOW MODEL GOVERNANCE                                                          │
│  Production | Candidate → Shadow (live predictions, no execution) → Promote/Rollback│
│  Registry: model_id, version, training_window, metrics, stability, promotion_ts   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  CONTINUOUS LEARNING: Data → Feature → Label → Train → Walk-forward → Shadow     │
│  → Promote (if replacement_rule) → Monitor → Drift → Retrain trigger             │
│  SAFETY: kill switch, broker disconnect, data feed failure, PnL anomaly, flood   │
│  OBSERVABILITY: drift, Sharpe drop, execution degradation, drawdown alerts       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1 — Cross-Sectional Opportunity Ranking Engine

### OpportunityScore formula

```
OpportunityScore = w1 * calibrated_prob_direction
                 + w2 * expected_return_risk_adj
                 + w3 * meta_confidence_adj
                 + w4 * regime_weight
                 - w5 * volatility_penalty
                 - w6 * spread_slippage_penalty
                 - w7 * liquidity_penalty
                 - w8 * microstructure_noise_penalty
```

- **calibrated_prob_direction** = |prob_up - 0.5| * 2  (0..1 strength of direction)
- **expected_return_risk_adj** = E[r] / (σ + ε)  (e.g. Sharpe-like per symbol)
- **meta_confidence_adj** = meta_confidence * (1 - P(primary_wrong))
- **regime_weight** = regime activation for this symbol (0..1)
- **volatility_penalty** = min(1, σ / σ_ref)  (higher vol → lower score)
- **spread_slippage_penalty** = (spread_bps + slippage_bps) / ref_bps  (capped)
- **liquidity_penalty** = 1 - min(1, ADV_ratio)  (low liquidity → penalty)
- **microstructure_noise_penalty** = noise_proxy (e.g. residual vol)

### Selection rules

- Rank symbols by OpportunityScore descending.
- Keep only if: score ≥ score_threshold, liquidity ≥ min_adv_ratio, spread ≤ max_spread_bps, E[r]/σ ≥ min_risk_adj_return.
- Apply: sector exposure cap, max correlated symbols (cluster cap), max concurrent signals N.

### Module

- **Location:** `src/ai/ranking/`
- **OpportunityRanker** — `rank(symbol_scores, liquidity, spread, sector, correlation_cluster) → List[RankedSymbol]` with filters and caps.

### Integration

- Inputs: per-symbol calibrated prob, E[r], meta_confidence, regime_weight, vol, spread, liquidity, noise (from feature store + ensemble + meta_alpha).
- Output: ordered list of symbols (and weights) for portfolio optimizer; no over-allocation to correlated/sector-heavy names.

---

## Phase 2 — Correlation-Aware Portfolio Optimizer

### Definitions

- **Rolling correlation matrix** C_ij over active symbols (e.g. 20d returns).
- **Marginal contribution to risk (MCR)** = (C @ w)_i * w_i / (w^T C w)^0.5  (per-name contribution to portfolio vol).
- **Portfolio heat** = sum over positions of |position_value| / equity  (gross exposure ratio).
- **Concentration ratio** = sum of squared weights (Herfindahl).

### Position sizing

- **Risk parity:** weights inversely proportional to marginal risk (or equal risk contribution).
- **Volatility targeting:** notional_i = capital * σ_target / σ_i  (scale by vol).
- **Kelly cap:** f_i = min(f_kelly, f_max).
- **Correlation penalty:** reduce weight when symbol is highly correlated with existing portfolio (e.g. weight *= (1 - λ * max_corr_with_portfolio)).

### Limits (integrate with RiskManager)

- **Max gross exposure** = sum |position_value| / effective_equity ≤ limit.
- **Max net exposure** = |sum signed position_value| / effective_equity ≤ limit.
- **Max sector exposure** = sum position_value in sector / effective_equity ≤ sector_cap.
- **Max correlated cluster exposure** = sum position_value in cluster / effective_equity ≤ cluster_cap.

### Module

- **Location:** `src/ai/portfolio/`
- **CorrelationOptimizer** — Rolling correlation matrix; MCR, heat, concentration; risk-parity/vol-target weights; correlation penalty; enforce gross/net/sector/cluster vs effective_equity.

### Integration

- Consumes ranked symbols and risk budget; outputs position sizes; RiskManager checks against same limits (effective_equity from exposure_multiplier).

---

## Phase 3 — Liquidity-Aware Execution Model

### Dynamic slippage

```
Slippage_bps = base_bps + k1 * (order_size / avg_minute_volume)^alpha
             + k2 * (spread_bps / ref_spread) + k3 * vol_regime_mult
```

- **alpha** > 1 (e.g. 1.5) for nonlinear scaling with participation.
- **vol_regime_mult** = 1.0 in normal vol, > 1 in high vol.

### Backtest

- Fill size capped by **volume participation threshold** (e.g. max 10% of bar volume).
- Use **DynamicSlippageModel** in FillModel: slippage = f(order_size/avg_volume, spread, vol_regime).

### Live

- If intended participation > threshold → **auto-reduce size** to meet cap.
- **Execution quality metrics** (realized slippage, partial fill rate) fed into meta-alpha / sizing (Phase 8).

### Module

- **Location:** `src/backtesting/dynamic_slippage.py`, extend `src/backtesting/fill_model.py` to use it.
- **ExecutionQualityTracker** in `src/execution/quality/` — realized vs expected slippage, partial fill rate, rejection rate; feed into position sizing and strategy disable.

---

## Phase 4 — Regime-Specialist Model Architecture

### Specialists

| Specialist           | When active              | Role                    |
|----------------------|--------------------------|-------------------------|
| Trend                | regime = trending_up/down| Momentum / trend follow |
| Mean reversion       | regime = sideways        | Reversion to POC/VWAP   |
| High vol breakout    | regime = high_vol        | Breakout / vol expansion|
| Low liquidity defensive | regime = low_liq      | Defensive, smaller size|

### Logic

- **Regime classifier** outputs regime_id (and optionally weights).
- **Activate** one or more specialists by regime; **blend** outputs with regime weights (e.g. softmax or linear blend).
- **Disable** strategies unsuitable for current regime (e.g. mean reversion in strong trend).
- **Walk-forward** validation: compute stability score **per regime** (or per regime-window); replacement rule can require stability across regimes.

### Module

- **Location:** `src/ai/regime_specialists/`
- **RegimeSpecialistRegistry** — Register specialist (name, regime_ids, predict_fn). **get_active_models(regime_id) → List[model]**. **blend(predictions, regime_weights) → combined output**.

### Integration

- Sits between feature store and ensemble: features → regime → active specialists → blended prob/E[r]/confidence → ensemble or replaces single universal model.

---

## Phase 5 — Shadow Model Governance Engine

### Lifecycle

1. **Production** model runs live (execution allowed).
2. **Candidate** model trained weekly (or on drift trigger).
3. **Candidate** deployed in **shadow mode**: generates live predictions, **no execution**.
4. **Compare** shadow vs production: Sharpe, drawdown, calibration, stability score, drift signals.
5. **Promote** only if **replacement_rule** passes (Sharpe better, drawdown not worse, stability ≥ threshold, N consecutive positive windows).
6. **Archive** previous production model (version, metrics).
7. **Auto rollback** if live Sharpe drops below threshold (e.g. over W days) after promotion.

### Registry metadata

- model_id, version, training_window_start/end, performance_metrics (Sharpe, dd, win_rate), stability_score, promotion_date, status (production | shadow | archived).

### Module

- **Location:** `src/ai/shadow_governance/`
- **ShadowModelGovernance** — deploy_shadow(candidate), compare(shadow_metrics, production_metrics), promote() / rollback(), get_production_model(), registry with metadata.

### Integration

- Retrain pipeline produces candidate → ShadowModelGovernance.deploy_shadow(candidate). Monitoring job compares shadow vs production; on replacement_rule pass → promote(); on live Sharpe drop → rollback().

---

## Phase 6 — Continuous Learning Loop

- **Pipeline:** Data → Feature → Label (triple-barrier, meta) → Train → Walk-forward → Stability test → Shadow deploy → Compare → Promote (if rule passes) → Monitor → Drift detection → Retrain trigger.
- **Min sample requirement:** Do not train if training sample < N_bars or N_trades.
- **Overfitting detection:** Validation loss >> train loss; or walk-forward variance of Sharpe too high → do not promote.
- **Stability across windows:** Require stability score ≥ threshold and frac_positive_Sharpe ≥ min_frac.
- **Human override:** Emergency kill only; no human required for promote/retrain.

### Integration

- SelfLearningOrchestrator + RetrainPipeline + ShadowModelGovernance + drift detectors; scheduler or event-driven retrain trigger.

---

## Phase 7 — Portfolio Heat & Drawdown Control

- **Rolling max drawdown monitor:** Track peak equity and current equity; dd_pct = (peak - current) / peak * 100.
- **Volatility spike detection:** Current vol > k * rolling_median(vol) → raise flag; optionally reduce exposure.
- **Portfolio heat limit:** heat = sum |position_value| / equity ≤ heat_limit (e.g. 1.0 or 1.5).
- **Trade pause threshold:** If dd_pct ≥ pause_threshold (e.g. 3%), pause new entries until dd improves or manual reset.
- **Exposure scaling:**

```
ExposureScale = max(0.2, 1 - current_drawdown_pct / dd_limit_pct)
Final position size = base_size * ExposureScale * regime_mult * meta_alpha_scale
```

### Module

- **Location:** `src/ai/portfolio_control/`
- **PortfolioHeatController** — update(equity, peak_equity, positions); get_drawdown_pct(), get_heat(), get_exposure_scale(); should_pause_new_trades(); vol_spike_detected().

### Integration

- RiskManager or OrderEntryService consults PortfolioHeatController for exposure_scale and pause; position sizing uses ExposureScale.

---

## Phase 8 — Execution Quality Feedback Loop

- **Track:** Realized slippage vs expected, order rejection rate, partial fill rate, latency, spread widening events.
- **Feed into:**
  - **Position sizing reduction:** If realized slippage > expected by margin → reduce size for next N bars or for symbol.
  - **Strategy disablement:** If rejection rate or partial fill rate exceeds threshold → disable strategy or symbol.
  - **Broker failover:** If latency or rejections spike → switch to backup broker or paper.

### Module

- **Location:** `src/execution/quality/`
- **ExecutionQualityTracker** — record_fill(expected_slippage, realized_slippage, partial_fill), record_rejection(), get_slippage_ratio(), get_rejection_rate(), get_partial_fill_rate(); recommend_size_multiplier(), recommend_disable().

### Integration

- FillHandler and OrderEntryService record outcomes; ExecutionQualityTracker aggregates; meta_allocator or position sizer uses recommend_size_multiplier(); strategy runner uses recommend_disable().

---

## Phase 9 — Meta-Alpha Integration (Before Order Entry)

- **If P(primary_wrong) high:** Reduce size (e.g. size *= (1 - P_wrong)) or **block trade** if P_wrong > block_threshold.
- **If P(regime_flip) high:** Reduce leverage (regime_mult < 1), tighten stop-loss (e.g. closer stop).
- **If confidence inflation detected:** Reduce Kelly fraction (e.g. f *= (1 - inflation_score)).
- **Meta-alpha must influence allocation BEFORE order entry:** i.e. in OpportunityRanker (meta_confidence_adj), in MetaAllocator (meta_alpha_scale), and in position sizing (explicit size reduction or block in strategy/allocator layer). OrderEntryService still receives already-reduced size.

### Integration

- MetaAlphaPredictor output → AutonomousTradingController.meta_alpha_scale_for_allocator() and explicit **size_mult = (1 - P_primary_wrong)** and **block if P_primary_wrong > 0.7**. Apply in ranking/allocator/sizing layer; then pass final signal + quantity to OrderEntryService.

---

## Phase 10 — Capital Scaling Strategy

- **Stage 1 (low capital):** Conservative Kelly cap (e.g. f_max = 0.05), strict drawdown threshold (e.g. 3%).
- **Stage 2 (proven stability):** Increase capital gradually; maintain volatility targeting; relax Kelly cap slightly (e.g. f_max = 0.08) if stability score holds.
- **Stage 3 (aggressive scaling):** Add uncorrelated strategies, expand universe; keep risk budget and sector/correlation caps; maintain vol targeting.

### Implementation

- Config-driven stages (stage_id, f_max, dd_limit_pct, max_open_positions); scaling gate: only move stage if Sharpe stable across walk-forward, drawdown within limit, stability score above threshold (Phase 12).

---

## Phase 11 — Stress Testing & Crisis Simulation

- **Scenarios:** Flash crash (e.g. -5% in 5 min), high vol spike (2x vol), liquidity collapse (volume 0.5x, spread 2x), sudden regime flip (trend → mean reversion).
- **Measure:** Max intraday loss, recovery time, circuit breaker triggers (count and timing).
- **Improve:** Tighten circuit/stop-loss, reduce size in high vol, liquidity filter; run stress tests in CI or weekly.

### Module

- **Location:** `src/backtesting/stress/` (optional) — run_historical_stress(scenario_name, bars) → metrics.

---

## Phase 12 — Performance Target Framework

- **Primary:** Annualized Sharpe, Sortino, max drawdown, Calmar ratio.
- **Secondary:** Win rate, profit factor, turnover, slippage ratio.
- **Scaling condition:** Allow capital/stage scaling only if: Sharpe stable across walk-forward, drawdown within limit, stability score ≥ threshold.

### Module

- **Location:** `src/risk_engine/metrics.py` (extend) or `src/ai/performance_targets/` — compute primary/secondary; scaling_gate(metrics, walk_forward_sharpes, stability_score) → bool.

---

## Phase 13 — Safety & Failsafe Layer

- **Global kill switch** — Already present; when armed, only reduce-only orders.
- **Broker disconnect monitor** — If gateway disconnected or heartbeat missed → arm kill switch (reason=broker_disconnect).
- **Data feed failure monitor** — If no tick/bar for T seconds → arm kill switch or pause (reason=data_feed_failure).
- **PnL anomaly detector** — If |realized_pnl| > k * expected_max_loss → alert and optionally arm kill (reason=pnl_anomaly).
- **Order flood protection** — Max N orders per minute; beyond that reject (reason=order_flood).

### Integration

- OrderEntryService already checks kill switch. Add background tasks: broker heartbeat, data feed watchdog, PnL check post-fill, order rate limiter. On breach → KillSwitch.arm(reason).

---

## Phase 14 — Observability & Alerting

- **Real-time dashboard:** Equity curve, open positions, heat, drawdown, Sharpe (rolling), execution quality (slippage ratio, rejection rate).
- **Alerts:** Drift (PSI/calibration/Sharpe drop), Sharpe drop below threshold, execution degradation (slippage/rejections), equity drawdown above threshold.
- **Channels:** Prometheus metrics, Grafana, PagerDuty/email on critical.

### Implementation

- **Location:** `src/monitoring/` — add metrics for heat, dd, slippage_ratio, rejection_rate; `deploy/observability/prometheus/alerts_*.yaml` — drift, sharpe_drop, execution_degradation, drawdown.

---

## Phase 15 — Final Goal

- **Analyze full market** — Cross-sectional ranker evaluates all tradable symbols.
- **Select only highest quality opportunities** — Score threshold, liquidity/spread/min risk-adj return, sector/correlation caps.
- **Allocate capital optimally** — Correlation-aware optimizer, risk parity, vol targeting, Kelly cap, effective_equity.
- **Protect against drawdowns** — ExposureScale, heat limit, trade pause, circuit breaker.
- **Adapt to regime shifts** — Regime specialists, meta-alpha P(regime_flip) → reduce leverage.
- **Replace weak models** — Shadow governance, replacement_rule, rollback on Sharpe drop.
- **Scale responsibly** — Capital scaling stages, performance target gate.

**Risk-first and no AI bypass of risk engine preserved.**

---

## New Module Structure

```
src/ai/
  ranking/           # Phase 1: OpportunityRanker, OpportunityScore
  portfolio/        # Phase 2: CorrelationOptimizer, MCR, heat, limits
  regime_specialists/ # Phase 4: RegimeSpecialistRegistry, blend
  shadow_governance/ # Phase 5: ShadowModelGovernance, registry metadata
  portfolio_control/ # Phase 7: PortfolioHeatController, drawdown, exposure scale
src/backtesting/
  dynamic_slippage.py # Phase 3: DynamicSlippageModel
  fill_model.py      # (extend to use dynamic slippage)
src/execution/
  quality/          # Phase 8: ExecutionQualityTracker
src/monitoring/
  alerts.py         # Phase 14: drift, sharpe_drop, execution, drawdown
  (existing metrics extended for heat, dd, slippage_ratio)
```

---

## Data Flow Updates

1. **Universe** → Feature Store (per symbol) → Regime Classifier → Regime Specialists (active models) → Ensemble (or specialist blend) → Calibrated prob, E[r], confidence.
2. **MetaAlpha** → P(wrong), P(regime_flip), P(inflated) → Meta_confidence_adj, meta_alpha_scale, size reduction/block **before** allocator.
3. **OpportunityRanker** → Ranked symbols (score, filters, sector/correlation caps) → **CorrelationOptimizer** → Position sizes (risk parity, vol target, correlation penalty) → **PortfolioHeatController** (exposure scale, pause) → **RiskManager** (effective_equity, limits) → **OrderEntryService**.
4. **Execution** → FillHandler → ExecutionQualityTracker → feedback to sizing/disable; OrderEntryService + Liquidity-aware execution (dynamic slippage, participation cap).
5. **Retrain** → Candidate → ShadowModelGovernance (shadow) → Compare → Promote/Rollback; Continuous learning loop with min sample, overfitting check, stability.

---

## Key Formulas Summary

| Item | Formula |
|------|--------|
| OpportunityScore | w1*cal_prob + w2*E[r]/σ + w3*meta_conf - w5*vol_pen - w6*spread_pen - w7*liq_pen - w8*noise_pen |
| ExposureScale | max(0.2, 1 - current_dd_pct / dd_limit_pct) |
| Dynamic slippage | base + k1*(qty/avg_vol)^α + k2*spread_ratio + k3*vol_regime |
| MCR | (C@w)_i * w_i / sqrt(w^T C w) |
| Portfolio heat | sum \|position_value\| / equity |
| Correlation penalty | weight *= (1 - λ * max_corr_with_portfolio) |

---

## Risk Safeguards

- All orders through OrderEntryService (no bypass).
- Kill switch: broker disconnect, data feed failure, PnL anomaly, order flood → arm.
- Circuit breaker: drawdown, daily loss → open circuit.
- Exposure multiplier (LLM): effective_equity scales position caps.
- Meta-alpha: reduce or block **before** order entry.
- Sector/correlation caps in ranker and portfolio optimizer.
- Max concurrent signals, liquidity/spread filters.
- Shadow promote only if replacement_rule; rollback if live Sharpe drops.

---

## Suggested Implementation Order

1. **Phase 1** — Opportunity ranker (score, rank, filters, caps).
2. **Phase 2** — Correlation optimizer (corr matrix, MCR, heat, limits, effective_equity).
3. **Phase 7** — Portfolio heat & drawdown control (exposure scale, pause).
4. **Phase 9** — Meta-alpha integration (explicit size reduction/block before order entry).
5. **Phase 3** — Dynamic slippage + execution quality tracker (Phase 8).
6. **Phase 4** — Regime specialists.
7. **Phase 5** — Shadow model governance.
8. **Phase 6** — Continuous learning loop wiring.
9. **Phases 10–14** — Scaling, stress tests, performance gate, safety monitors, observability.

---

## Implemented Modules (Code)

| Phase | Module | Location |
|-------|--------|----------|
| 1 | OpportunityRanker, OpportunityScoreConfig, RankedSymbol | `src/ai/ranking/` |
| 2 | CorrelationOptimizer, PortfolioWeights, OptimizerConfig | `src/ai/portfolio/` |
| 3 | DynamicSlippageModel, DynamicSlippageConfig | `src/backtesting/dynamic_slippage.py` |
| 4 | RegimeSpecialistRegistry, RegimeSpecialist, SpecialistOutput | `src/ai/regime_specialists/` |
| 5 | ShadowModelGovernance, ModelMetadata, ShadowResult | `src/ai/shadow_governance/` |
| 7 | PortfolioHeatController, HeatConfig | `src/ai/portfolio_control/` |
| 8 | ExecutionQualityTracker, QualityMetrics | `src/execution/quality/` |
| 9 | block_trade_from_meta_alpha, size_multiplier_from_meta_alpha, regime_flip_stop_multiplier | `src/ai/autonomous/controller.py` |
| 13 | OrderRateLimiter | `src/execution/order_entry/rate_limiter.py` |
| 14 | portfolio_heat, drawdown_pct, slippage_ratio, rejection_rate, sharpe_rolling, drift_psi | `src/monitoring/metrics.py` |
| 14 | DriftDetected, SharpeDrop, ExecutionDegradation, EquityDrawdown, PortfolioHeatHigh | `deploy/observability/prometheus/alerts_institutional.yaml` |

---

## Potential Failure Points

| Risk | Mitigation |
|------|------------|
| Ranker over-concentrates in one sector | Sector cap; correlation cluster cap. |
| Correlation matrix singular or stale | Regularize (shrink to diagonal); min lookback; cap weight. |
| Dynamic slippage underestimates in stress | Cap participation; stress-test slippage model. |
| Regime misclassification | Use regime weights (soft); disable only on high confidence. |
| Shadow metric mismatch (live vs backtest) | Compare live shadow vs production over same period; require min sample. |
| Rollback too late | Auto rollback on Sharpe drop over short window (e.g. 5d). |
| Execution quality feedback loop too slow | Use short rolling window (e.g. 50 fills); immediate size cut on slippage spike. |
| Kill switch false positive (data delay) | Heartbeat timeout tuned; manual disarm; audit log. |

---

## Monitoring Plan

- **Real-time:** Equity, heat, drawdown, open positions, order rate, rejection rate, latency.
- **Daily:** Sharpe (rolling), calibration error, drift (PSI, calibration MSE), execution quality (slippage ratio, partial fill rate).
- **On event:** Kill switch arm, circuit open, shadow vs production diff, rollback.
- **Alerts:** Drift > threshold, Sharpe drop > X, drawdown > Y, execution degradation (slippage/rejections), PnL anomaly, broker/data feed failure.
