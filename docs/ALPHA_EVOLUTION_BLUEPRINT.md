# Alpha Evolution Blueprint — Next-Generation Alpha Improvement Plan

**Scope:** AI-driven intraday trading system for Indian equities. Capital preservation priority; execution costs matter.

---

## 1. Label Engineering Redesign

### 1.1 Why Triple-Barrier Improves Signal Robustness

**Current issue:** Simple next-bar direction (1 if r_{t+1} > 0 else 0) or raw return label is noisy, ignores holding period, and does not align with execution (fill at t+latency, hold until target/stop/time).

**Triple-barrier method:** For each observation at time t, define:
- **Profit target (upper barrier):** price reaches p_t * (1 + b_u) first → label +1.
- **Stop loss (lower barrier):** price reaches p_t * (1 - b_d) first → label -1.
- **Time limit (horizontal barrier):** horizon H bars reached first → label 0 (or sign of return over H if neutral).

**Why it improves robustness:**
1. **Execution-aligned:** Labels reflect “what would have happened” if we had entered at t and exited at first touch of barrier or horizon. No lookahead beyond H.
2. **Asymmetric payoff:** b_u and b_d can be set from volatility (e.g. 1× ATR) so labels reflect risk-adjusted outcomes.
3. **Class balance:** Time-out label 0 reduces imbalance and avoids labelling every bar as +1/-1 from noise.
4. **Cost-aware variant:** Barriers can be defined **after** estimated round-trip cost (slippage + commission). So upper barrier = p_t * (1 + b_u) - cost_pct; lower = p_t * (1 - b_d) + cost_pct. Labels then reflect “trade would have been profitable after costs.”

### 1.2 Mathematical Definition

**Inputs:** Price series p_0, p_1, …; at bar t we have p_t.  
**Parameters:**  
- b_u, b_d ∈ (0, 1) (barrier fractions, or volatility-adjusted, e.g. b_u = k_u * σ_t, b_d = k_d * σ_t).  
- H = max holding bars (e.g. 30 for 30-min intraday).  
- c = round-trip cost as fraction of price (e.g. 0.001).

**Cost-aware barriers (in price space):**
- Upper: U_t = p_t * (1 + b_u) − c * p_t = p_t * (1 + b_u − c).
- Lower: L_t = p_t * (1 − b_d) + c * p_t = p_t * (1 − b_d + c).

**Label generation (pseudocode):**

```
function triple_barrier_label(t, p, b_u, b_d, H, c):
  p_t = p[t]
  U = p_t * (1 + b_u - c)
  L = p_t * (1 - b_d + c)
  for τ = 1 to H:
    if t + τ >= len(p): break
    if p[t+τ] >= U: return +1   # profit target hit first
    if p[t+τ] <= L: return -1   # stop hit first
  # time limit
  r_H = (p[t+H] - p_t) / p_t
  return sign(r_H) if abs(r_H) > epsilon else 0
```

**Preventing label leakage:**
- Use only information available at t: p_0..p_t and features computed from p_0..p_t. Do **not** use p_{t+1}..p_{t+H} for feature computation.
- Barriers U, L are computed from p_t and parameters (and optionally σ_t from p_{t−W..t}); no future data.
- Train/test split is **time-based**: train on [0, T_train), test on [T_train, T_train + T_test]. No shuffle across time.

**Alignment with intraday execution:**
- Set H to max intended holding period (e.g. 20–60 bars for 1m data).
- Set b_u, b_d from ATR(t) so barriers are realistic (e.g. 1.5× ATR).
- Cost c = (slippage_bps + commission_bps) / 10^4 per side × 2 for round-trip.

### 1.3 Meta-Labeling Layer

**Primary model:** Predicts “is there an edge?” (triple-barrier label ∈ {−1, 0, +1} or binary “touch upper before lower/time”).

**Meta-label:** Given primary prediction (e.g. +1), meta-label = 1 if the **actual** trade (using same barrier rules) would have been profitable **after costs**, else 0. So meta-model learns “when does the primary model’s signal actually make money?”  
- Training: only on samples where primary predicted a trade (e.g. pred ∈ {−1, +1}); label = realized profit after cost > 0.
- Inference: primary outputs direction; meta outputs probability that this trade will be profitable → use for position sizing or filtering.

### 1.4 Regime-Conditioned Labeling

- Compute regime at t (e.g. trending_up, sideways, high_vol) from features available at t.
- Option A: Train one model per regime (separate datasets by regime).
- Option B: Use same triple-barrier logic but **different** (b_u, b_d, H) per regime (e.g. larger barriers in high_vol, shorter H in sideways).
- Labels remain leak-free: regime is derived from past data only.

---

## 2. Modeling Upgrade

### 2.1 Regime-Specific Models

- **Per-regime models:** Split training data by regime; train model_k for regime k. At inference: classify regime → use model_k. Prevents one model from averaging over incompatible regimes.
- **Regime as feature:** Single model with feature “regime_prob” or one-hot regime. Cheaper; may underperform when regimes are very different.
- **Mixture-of-experts:** Gate network outputs weight w_k per expert; y = Σ w_k * model_k(x). Train gate and experts jointly or in stages.

### 2.2 Probability Calibration

- **Platt scaling:** Fit sigmoid(β0 + β1 * logit) on validation set so P_cal = sigmoid(β0 + β1 * logit(p)).
- **Isotonic regression:** Non-parametric mapping from raw score to calibrated probability; good when reliability curve is non-linear.
- **Monitoring:** Track reliability curve (binned predicted prob vs realized frequency) in production; alert on calibration drift.

### 2.3 Online Ensemble Weighting

- **Confidence-weighted:** w_k ∝ recent accuracy or inverse error; update every N periods.
- **Bayesian model averaging:** Posterior weight ∝ likelihood × prior; approximate with rolling likelihood (e.g. Bernoulli likelihood of correct direction).
- **Decay-based:** Reduce weight of model k if its rolling Sharpe or hit rate drops below threshold.

### 2.4 Feature Importance Monitoring

- **SHAP (or permutation importance)** on validation windows; store importance vector per window.
- **Divergence:** Alert when cosine similarity or L2 distance between current and reference importance vector exceeds threshold. Prevents silent feature drift.

### 2.5 Overfitting and Walk-Forward

- **Overfitting:** Regularization (L1/L2, early stop), small trees, few features; use walk-forward so test is always out-of-sample in time.
- **Walk-forward:** Train on [t−T, t), validate on [t, t+V), test on [t+V, t+V+O). Roll t; aggregate metrics across windows. No peeking into future.

---

## 3. Alpha Feature Expansion (Intraday-Specific)

| Feature | Mathematical definition | Predictive hypothesis | Cost |
|--------|--------------------------|----------------------|------|
| **Order flow imbalance (multi-horizon)** | OFI_h = (V_buy − V_sell) / (V_buy + V_sell) over last h bars | Persistent imbalance predicts short-term direction | Low (tick sum) |
| **Volume profile POC** | Price level with max volume over session (or rolling window) | Reversion to POC; breakouts from POC | Medium (histogram) |
| **Value area** | Price range containing X% (e.g. 70%) of volume | Bounds for mean reversion | Medium |
| **Intraday seasonality** | Time-of-day encoding (e.g. sin/cos of minute-of-day) or dummy for 9:15–10, 10–12, … | Liquidity and volatility vary by time | Low |
| **VWAP deviation curvature** | Second derivative of (p − VWAP) w.r.t. time or d²(deviation)/dt² | Acceleration of mean reversion | Low |
| **Liquidity vacuum detector** | Spread > k × rolling_median(spread) or volume < k × rolling_median(volume) | Avoid trading in vacuum; or trade when vacuum ends | Low |
| **Vol-of-vol** | σ(σ_t) over rolling window of realized vol | Regime change; tail risk | Low |
| **Microstructure noise filter** | Residual from rolling MA or kernel regression on mid; or Roll model σ²_noise | Filter noise for cleaner signals | Medium |
| **Spread widening predictor** | Lagged spread, vol, order imbalance → predict Δ spread | Anticipate cost increase | Low |
| **Cross-stock dispersion** | Std of returns across sector/index constituents | High dispersion → rotation or risk-off | Low |
| **Sector momentum rotation** | Rank of sector return over last N bars | Momentum continuation or rotation | Low |

---

## 4. Position Sizing Intelligence

**Target:** Replace static 5% with dynamic size.

**Formula (dynamic position size as fraction of capital):**

```
f* = (μ * p_win − (1−p_win)) / (μ * (1−p_win) + p_win)   # simplified Kelly for binary
f  = clip(f* * confidence, 0, f_max)
f  = f * regime_multiplier   # e.g. 0.5 in high_vol
f  = f * drawdown_scale      # e.g. 1 − 0.5 * (current_dd / max_dd_limit)
```

**Definitions:**
- μ = expected win/loss ratio (avg win / avg loss).
- p_win = P(win) from model or calibrated prob.
- confidence = model confidence or meta-model “this trade will work” prob.
- f_max = cap (e.g. 0.1).
- regime_multiplier ∈ (0.5, 1] from regime (e.g. 0.5 in crisis).
- drawdown_scale = max(0, 1 − α * (current_drawdown_pct / max_dd_pct)).

**Volatility targeting:** Target annual vol σ_target. Then notional = capital * σ_target / σ_forecast, where σ_forecast = model vol or rolling vol.

**Implementation:** PositionSize = capital * f * (1 / price) for shares; or in notional = capital * f.

---

## 5. Drift Detection Upgrade

**Multi-layer monitoring:**

1. **Prediction distribution drift:** KS test or PSI on binned predictions (current window vs reference).
2. **Calibration drift:** Binned predicted prob vs realized frequency; χ² or MSE of calibration error.
3. **Rolling live Sharpe drift:** Rolling Sharpe over last W days; alert if drops below threshold or drops by Δ from peak.
4. **Feature importance drift:** SHAP/permutation importance per window; alert on divergence from baseline.
5. **Regime frequency shift:** Distribution of regime labels over window vs reference; χ² or PSI.
6. **Correlation structure drift:** Correlation matrix of features (or returns) vs reference; Frobenius norm of difference.

**Threshold design:** Use validation history to set percentiles (e.g. alert if PSI > 95th percentile of historical PSI). Avoid single fixed threshold that is too sensitive.

**Avoiding false retrains:** Require drift to persist over K consecutive checks; and/or require shadow model (new model trained on recent data) to beat current model on holdout before replace.

**Shadow model:** Train candidate on [t−T, t); evaluate on [t, t+V). Replace only if Sharpe (and optionally drawdown) better and stability score above threshold.

---

## 6. Walk-Forward Validation Framework

**Rolling window:** Train on [t−60d, t), test on [t, t+10d]. Roll t by 10d. Repeat.

**Expanding window:** Train on [0, t), test on [t, t+10d]. Roll t. Use when regime may have shifted and long history still relevant.

**Multi-period Sharpe consistency:** For each window w, compute Sharpe_w. Require median(Sharpe_w) > 0 and e.g. 5th percentile > −0.5 so that most windows are not bad.

**Max drawdown stability:** For each window, compute max_dd_w. Require median(max_dd_w) < limit and e.g. 95th percentile < 2× limit.

**Stability score (example):**

```
S = (mean_Sharpe / (std_Sharpe + ε)) * (1 − mean_dd / dd_limit) * I(frac_windows_positive_Sharpe >= 0.7)
```

Only replace if:
- mean_Sharpe_new > mean_Sharpe_current
- mean_dd_new ≤ mean_dd_current (or not worse by more than δ)
- S_new ≥ S_min
- At least N (e.g. 3) consecutive windows with positive Sharpe.

---

## 7. Meta-Alpha Layer

**Meta-model:** Predicts “primary model is likely wrong” or “confidence is inflated” or “regime will flip.”

**Inputs:**
- Primary prediction and confidence.
- Features: recent primary errors (0/1 or residual), recent confidence vs realized hit rate, regime state, volatility, time-of-day.

**Targets:**
- **Meta-label 1:** 1 if primary was wrong (direction ≠ realized), 0 else.
- **Meta-label 2:** 1 if |confidence − realized_frequency| > threshold (miscalibration).
- **Meta-label 3:** 1 if regime flipped in next H bars.

**Architecture:** Lightweight classifier (e.g. logistic or small MLP) trained on recent history. Output: P(primary wrong), P(regime flip). Use to down-weight primary or reduce size.

**Training:** Roll historical windows; for each, compute primary prediction and realized outcome; label meta targets; train on non-overlapping windows.

---

## 8. Risk-Adjusted Objective Function

**Objective:**

Maximize  J = E[R] − λ1 * DD − λ2 * Turnover − λ3 * σ

- E[R] = expected return (e.g. mean of period returns).
- DD = max drawdown or average drawdown.
- Turnover = sum of |Δw| (reduces churn).
- σ = volatility of returns.

**Approximation during training:**
- For tree models: replace default loss (e.g. log loss) with custom loss that penalizes predicted prob when realized return was negative and rewards when positive; or use reward = return − λ * drawdown_contribution in gradient.
- For neural nets: loss = − Sharpe (or − mean return / (σ + ε)) on batch; or REINFORCE-style with reward = return − λ1*dd − λ2*turnover.
- **Direct Sharpe-like:** Optimize (mean_ret / (std_ret + ε)) on validation; hyperparameter search over λ in training loss.

---

## 9. LLM Layer Improvement

- **Multi-source aggregation:** Combine news from NSE, RBI, earnings, global; dedupe and merge by time and theme.
- **Embedding clustering:** Embed headlines; cluster into themes (e.g. “rate hike”, “earnings beat”); track theme sentiment over time.
- **Event severity scoring:** Classify event type (earnings, macro, geopolitical) and severity (low/med/high); map to risk multiplier.
- **Risk multiplier output:** LLM outputs “exposure_multiplier” ∈ {0.5, 1.0, 1.5} with reasoning; apply to position size. Guardrail: cap multiplier to [0.5, 1.5] and require citation/source.
- **Guardrails:** (1) No trade instructions; only “reduce/increase/hold exposure” with multiplier. (2) Require source link or snippet for material claims. (3) Sanity check: if multiplier extreme, require human review or cap.

---

## 10. Institutional Benchmarking

**Renaissance Technologies:** Would prioritize (1) massive alternative data and feature discovery; (2) ensemble of many weak signals; (3) rigorous statistical testing and multiple testing correction; (4) execution at minimal market impact (VWAP, dark pools). For this system: add more data sources, automated feature discovery, and strict p-value / false discovery control.

**Two Sigma:** Would emphasize (1) ML infrastructure and A/B testing of models; (2) probabilistic reasoning and uncertainty quantification; (3) risk parity and diversification across strategies. For this system: add shadow model A/B tests, uncertainty estimates (e.g. Bayesian or dropout), and cross-strategy correlation limits.

**Citadel:** Would focus on (1) execution quality and TCA; (2) real-time risk and position limits; (3) fundamental + quant integration. For this system: add TCA (implementation shortfall, arrival price), real-time PnL and risk dashboards, and optional fundamental overlay (earnings, guidance) as features.

---

## 11. Prioritized Roadmap

**Tier 1 – High alpha impact, low risk**  
- Triple-barrier + cost-aware labels; meta-labeling filter.  
- Probability calibration (Platt or isotonic) and reliability monitoring.  
- Dynamic position sizing (confidence × Kelly cap × drawdown_scale).  
- Walk-forward with stability score and replacement rule.

**Tier 2 – Structural model upgrade**  
- Regime-specific models or regime as feature.  
- Online ensemble weighting (confidence/decay-based).  
- Multi-layer drift (prediction, calibration, Sharpe, importance) + shadow model.  
- Meta-alpha model (when primary wrong / regime flip).

**Tier 3 – Research-level**  
- Alpha feature expansion (order flow multi-horizon, volume profile, vol-of-vol, liquidity vacuum, cross-stock dispersion).  
- Risk-adjusted objective (Sharpe-like or J = E[R] − λ*DD − λ*Turnover).  
- LLM risk multiplier and event severity with guardrails.

**Tier 4 – Institutional research lab**  
- Mixture-of-experts; automated feature discovery.  
- Multi-source LLM aggregation and embedding themes.  
- TCA and execution-quality feedback into labels/sizing.

---

*Implemented in code:*
- **Labels:** `src/ai/labels/` — TripleBarrierLabeler, MetaLabeler, cost_aware_barriers.
- **Position sizing:** `src/ai/position_sizing/` — dynamic_position_fraction, volatility_target_notional.
- **Walk-forward:** `src/ai/walk_forward/` — stability_score, replacement_rule.
- **Meta-alpha:** `src/ai/meta_alpha/` — MetaAlphaPredictor (P(primary wrong), P(regime flip)).
- **Calibration:** `src/ai/calibration/` — PlattCalibrator, IsotonicCalibrator, reliability_curve.
- **Drift:** `src/ai/drift/` — MultiLayerDriftDetector (prediction, calibration, Sharpe, importance).
- **Objectives:** `src/ai/objectives/` — risk_adjusted_reward, sharpe_like_score.
- **Intraday alpha specs:** `src/ai/feature_engineering/specs.py` — INTRADAY_ALPHA_SPECS.
- **LLM advisory:** `src/ai/llm/advisory.py` — AdvisoryService (event severity, exposure_multiplier 0.5–1.5).
