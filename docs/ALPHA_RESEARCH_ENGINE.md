# Alpha Research & Edge Discovery Engine

**Scope:** Fully automated, statistically rigorous framework to discover, validate, rank, deploy, and monitor intraday alpha signals. Institution-grade; no overfitting; cost-aware; capacity-aware.

---

## 1. Architecture Diagram (Text)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  DATA (bars, ticks, order book, sector/index)                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE A — HYPOTHESIS GENERATION ENGINE                                           │
│  Price / Microstructure / Cross-sectional / Regime-conditioned / Nonlinear        │
│  Systematic variations; statistical pre-filter → candidate list (no explosion)   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE B — STATISTICAL VALIDATION LAYER                                           │
│  IC, IC stability (time + regime), turnover-adjusted IC, E[r] after cost,        │
│  FDR correction, min sample, OOS, walk-forward, permutation → REJECT weak         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE C — SIGNAL QUALITY SCORING                                                 │
│  AlphaQualityScore = f(IC_mean, IC_stability, Sharpe_OOS, Regime_Robustness,      │
│    Capacity_Score, -Turnover_penalty, -Slippage_sensitivity)                      │
│  Rank → keep top decile; archive rest                                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE D — SIGNAL CLUSTERING & DIVERSIFICATION                                    │
│  Correlation / distance → hierarchical clustering → one strongest per cluster     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE E — CAPITAL CAPACITY MODELING                                              │
│  Participation impact, slippage scaling, depth sensitivity, turnover pressure     │
│  Simulate small → large capital; reject if edge breaks                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE F — LIVE SHADOW DEPLOYMENT (existing ShadowModelGovernance)                 │
│  New alpha runs shadow; live IC, live Sharpe, decay track; promote if rule pass    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE G — DECAY MONITORING                                                       │
│  Rolling IC, rolling Sharpe, half-life, drift → gradual weight reduction           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE H — META-ALPHA FEEDBACK                                                    │
│  Alpha health → meta-alpha: penalize decaying, boost stable, reduce on uncertainty │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE I — EDGE PRESERVATION RULES (hard constraints on all paths)                │
│  No Sharpe-only optimization; cost-aware validation; turnover cap; 3 WF cycles;   │
│  Correlation with existing alpha < threshold                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE J — RESEARCH AUTOMATION PIPELINE                                           │
│  Data → Hypothesis → Statistical Filter → Backtest → OOS → FDR → Cluster →        │
│  Capacity Sim → Shadow → Promote → Monitor → Decay handling                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
┌─────────────────────────────┐         ┌─────────────────────────────┐
│  EXISTING: Ranking,         │         │  EXISTING: Risk engine,      │
│  Regime specialists,         │         │  Kill switch, Execution,     │
│  Ensemble, Meta-alpha,       │         │  Observability               │
│  Portfolio optimizer, Heat   │         │                             │
└─────────────────────────────┘         └─────────────────────────────┘
```

---

## 2. Mathematical Definitions

### Information Coefficient (IC)

**Definition:** Pearson correlation between signal (or signal rank) and forward return (or return rank) over a window.

- **IC_raw(s, r)** = corr(s_t, r_{t+1})  where s = signal, r = forward return.
- **Rank IC:** IC_rank = corr(rank(s_t), rank(r_{t+1}))  — more robust to outliers.
- **Time-varying IC:** IC_t computed over rolling window W (e.g. 20 days).

**IC stability across time:**  
σ_IC = std(IC_t over windows). Stability = 1 / (1 + σ_IC) or IC_mean / (σ_IC + ε). Higher = more stable.

**IC stability across regimes:**  
For each regime k, IC_k = mean(IC in regime k). Regime_IC_stability = min_k IC_k / (max_k IC_k + ε) or 1 - std(IC_k)/ (mean(IC_k)+ε). Signal is regime-robust if IC does not collapse in any regime.

**Turnover-adjusted IC:**  
IC_adj = IC_raw * (1 - λ * turnover). Turnover = sum |Δw| per period. λ = scaling (e.g. 0.1). Penalizes high-churn signals.

**Expected return after cost:**  
E[r]_net = E[r]_gross - (slippage_bps + spread_bps + commission_bps) * turnover_per_unit. Reject if E[r]_net ≤ 0.

**Capacity-adjusted return:**  
Return at capacity C = E[r]_net * f(C) where f(C) = 1 for C ≤ C_0, and f(C) < 1 (decay) for C > C_0 due to market impact. Capacity_Score = C_0 or integral of f up to break-even.

---

### False Discovery Rate (FDR) Correction

- **Benjamini–Hochberg:** Order p-values p_(1) ≤ … ≤ p_(m). Find largest k such that p_(k) ≤ (k/m) * α. Reject hypotheses 1..k. Controls FDR at level α.
- **Application:** Each candidate signal is a hypothesis (H0: IC = 0). Compute p-value from permutation test or t-test on IC. Apply BH; reject (don’t promote) signals that don’t pass FDR.

**Minimum sample size:**  
N_min bars (e.g. 500) and N_min_trades (e.g. 100). Reject if sample below threshold.

**Out-of-sample:**  
Train on [0, T1), validate on [T1, T2), test on [T2, T3). IC and Sharpe on test only. Reject if test IC or Sharpe collapses vs validation.

**Walk-forward:**  
Multiple (train, test) windows; require ≥ 3 windows with positive test IC (or positive Sharpe). Reject if fewer.

**Permutation testing:**  
Shuffle forward returns; compute IC under null. p-value = fraction of permutations with |IC_perm| ≥ |IC_obs|. Reject if p > α.

---

### Capacity Score

- **Definition:** Maximum notional (or capital) such that expected return after cost remains positive, or such that Sharpe degradation is below threshold (e.g. 10%).
- **Model:** E[r](C) = E[r]_0 - k * (C / ADV)^α. Estimate k, α from simulation or parametric model. Capacity_Score = C_max where E[r](C_max) = 0 or Sharpe(C_max) = 0.9 * Sharpe(0).
- **Normalized:** Capacity_Score_norm = C_max / target_capital ∈ [0, 1]. Use in AlphaQualityScore.

---

### AlphaQualityScore

```
AlphaQualityScore = w1 * IC_mean
                 + w2 * IC_stability
                 + w3 * Sharpe_OOS
                 + w4 * Regime_Robustness
                 + w5 * Capacity_Score
                 - w6 * Turnover_penalty
                 - w7 * Slippage_sensitivity
```

- **IC_mean:** Mean IC (or rank IC) over validation/OOS windows.
- **IC_stability:** 1 / (1 + σ_IC) or IC_mean / (σ_IC + ε).
- **Sharpe_OOS:** Out-of-sample Sharpe (annualized or per-bar scaled).
- **Regime_Robustness:** min(IC_k) / (mean(IC_k) + ε) or 1 - cv(IC_per_regime).
- **Capacity_Score:** Normalized C_max (0..1).
- **Turnover_penalty:** min(1, turnover / turnover_ref).
- **Slippage_sensitivity:** Sensitivity of return to 1 bps slippage (e.g. derivative or empirical).

Rank signals by AlphaQualityScore; **keep top decile**; archive others.

---

## 3. Promotion Rule Logic

A candidate alpha is **promoted** from shadow to production only if ALL hold:

1. **Live IC** stable: mean(live_IC) > 0 and std(live_IC) < threshold_ic_vol.
2. **Live Sharpe** positive over minimum window (e.g. 5 days).
3. **No drift flags:** Multi-layer drift (PSI, calibration, Sharpe drop) not triggered.
4. **Replacement rule:** Same as existing (candidate Sharpe > current, drawdown not worse, stability score ≥ threshold, N consecutive positive windows).
5. **Edge preservation:** Correlation with existing alphas < max_correlation; turnover ≤ max_turnover; passed ≥ 3 walk-forward cycles; cost-aware validation passed.

---

## 4. Decay Detection Formula

**Rolling IC:** IC_rolling(t) = corr(signal[t-W:t], return[t+1:t+1+H]) over rolling window W.

**Rolling Sharpe:** Sharpe_rolling(t) over same window.

**Half-life estimate:**  
Exponential decay model: IC(t) = IC_0 * exp(-t/τ). Fit τ from rolling IC series; half_life = τ * ln(2). If half_life < threshold_days → flag decay.

**Decay detected if:**  
- IC_rolling(t) < IC_rolling(t - L) * decay_threshold (e.g. 0.8) for L = lookback.  
- Or: Sharpe_rolling(t) < Sharpe_rolling(t - L) * decay_threshold.  
- Or: half_life < half_life_min.

**Action:**  
- **Gradual:** Reduce weight: w_new = w_old * (1 - decay_step) each period until IC/Sharpe recovers or weight reaches floor (e.g. 0.2).  
- **Abrupt disable** only if catastrophic (e.g. IC < -0.05 or Sharpe < -0.5 over 2 weeks).

---

## 5. New Module Structure

```
src/ai/alpha_research/
  __init__.py
  hypothesis/          # Phase A
    generator.py      # AlphaHypothesisGenerator
    transforms.py     # Price, micro, cross-section, regime, nonlinear
  validation/         # Phase B
    ic.py             # IC, IC stability, turnover-adjusted IC
    fdr.py            # FDR (Benjamini-Hochberg), permutation test
    validator.py      # StatisticalValidator (OOS, walk-forward, min sample)
  scoring/            # Phase C
    quality.py        # AlphaQualityScore, rank, top decile
  clustering/         # Phase D
    cluster.py        # Signal clustering, hierarchical, one per cluster
  capacity/           # Phase E
    model.py          # CapacityModel, participation impact, slippage scaling
  decay/              # Phase G
    monitor.py        # DecayMonitor, rolling IC/Sharpe, half-life, weight decay
  rules/              # Phase I
    preservation.py   # EdgePreservationRules (hard constraints)
  pipeline/           # Phase J
    orchestrator.py   # ResearchPipeline: full automation
```

---

## 6. Integration Points with Existing System

| Component | Integration |
|-----------|-------------|
| **Cross-sectional ranker** | Consumes AlphaQualityScore-ranked signals; OpportunityScore can include alpha_quality. |
| **Regime specialists** | Hypothesis engine generates regime-conditioned features; validation reports IC per regime. |
| **ML ensemble** | New alphas can be added as features or sub-models; ensemble weights from meta-allocator. |
| **Meta-alpha** | Phase H: DecayMonitor and quality metrics → meta-alpha input; penalize decaying, boost stable. |
| **Portfolio optimizer** | Selected signals (after clustering) are inputs; weights from optimizer + heat controller. |
| **Shadow governance** | Phase F: New alpha runs in shadow via ShadowModelGovernance; promotion uses promotion rule above. |
| **Drift detection** | Multi-layer drift triggers re-validation or weight reduction; no promote if drift. |
| **Execution quality** | ExecutionQualityTracker feeds slippage sensitivity and capacity checks. |
| **Risk engine** | All promoted alphas still pass RiskManager; no bypass. |

---

## 7. Risk Safeguards

- **No alpha promoted without cost-aware validation.** E[r]_net must be positive.
- **No optimization on Sharpe alone.** AlphaQualityScore and promotion rule use IC, stability, regime robustness, capacity, turnover.
- **FDR control.** Reduces false discoveries; α = 0.05 or 0.10.
- **Minimum sample and 3 walk-forward cycles.** Reduces overfitting and regime-specific flukes.
- **Correlation cap with existing alpha.** New signal must not duplicate existing edge.
- **Capacity and participation cap.** Reject signals that break above participation threshold.
- **Decay → gradual weight reduction.** Avoids cliff risk; abrupt disable only if catastrophic.
- **Shadow before production.** All new alpha runs shadow first; promotion only if rule pass.
- **Kill switch and risk engine unchanged.** Alpha research does not bypass order entry or risk.

---

## 8. Failure Modes and Mitigation

| Failure | Mitigation |
|---------|------------|
| **Combinatorial explosion in hypothesis gen** | Pruned template set; statistical pre-filter (e.g. univariate IC) before backtest; max candidates per run. |
| **Overfitting to validation** | Strict OOS; walk-forward; permutation test; FDR; min 3 WF cycles. |
| **Regime-dependent alpha promoted** | Regime_Robustness in score; reject if IC collapses in any regime without justification. |
| **Capacity overstated** | Conservative slippage model; simulate scaling; reject if edge breaks at target capital. |
| **Decay too late** | Short rolling window for IC/Sharpe; half-life estimate; gradual weight decay on first sign. |
| **FDR too loose** | Use α = 0.05; minimum sample size; independent test window. |
| **Clustering merges distinct edges** | Use correlation + distance; pick one per cluster; monitor cluster stability. |
| **Meta-alpha ignores decay** | Explicit feed: alpha_id → health_score; meta-alpha penalizes low health. |

---

## 9. Suggested Rollout Order

1. **Phase B + I** — Validation layer and edge preservation rules (so every candidate is validated and constrained).
2. **Phase A** — Hypothesis generator with pre-filter (so we have a steady stream of candidates).
3. **Phase C** — Signal quality scoring and top-decile selection.
4. **Phase D** — Clustering and diversification (reduce redundancy).
5. **Phase E** — Capacity modeling (reject fragile signals).
6. **Phase F** — Wire shadow deployment for new alpha (existing ShadowModelGovernance).
7. **Phase G** — Decay monitoring and gradual weight reduction.
8. **Phase H** — Meta-alpha feedback (alpha health → allocation).
9. **Phase J** — Full pipeline automation (orchestrator).

---

## 10. Performance Targets (Realistic for Intraday)

- **Information Coefficient:** Mean IC ∈ [0.02, 0.08] for robust intraday signals; IC_stability (1/(1+σ_IC)) > 0.5.
- **Sharpe ratio:** Annualized Sharpe 0.8–1.5 for single signal; portfolio (diversified) target 1.0–1.8.
- **Max drawdown:** < 8% for portfolio; single signal < 12%.
- **Turnover:** Daily turnover < 50% unless justified by capacity and cost.
- **Win rate:** 52–58% after cost (not 70%+ to avoid overfitting).
- **Decay half-life:** Prefer signals with estimated half-life > 20 days.
- **Capacity:** Signal remains profitable at target AUM (e.g. 1–5% ADV participation cap).

No guarantee of profit; design for probability, robustness, and survival.

---

## 11. Implemented Modules (Code)

| Phase | Module | Location |
|-------|--------|----------|
| A | AlphaHypothesisGenerator, HypothesisSpec, transform_templates | `src/ai/alpha_research/hypothesis/` |
| B | ic_rank, ic_stability_time, ic_stability_regime, turnover_adjusted_ic | `src/ai/alpha_research/validation/ic.py` |
| B | fdr_benjamini_hochberg, permutation_test_ic | `src/ai/alpha_research/validation/fdr.py` |
| B | StatisticalValidator, ICResult, ValidationResult | `src/ai/alpha_research/validation/validator.py` |
| C | AlphaQualityScorer, AlphaQualityScoreConfig, rank_and_select | `src/ai/alpha_research/scoring/` |
| D | SignalClustering, ClusterConfig, hierarchical cluster + one per cluster | `src/ai/alpha_research/clustering/` |
| E | CapacityModel, CapacityResult, estimate_capacity | `src/ai/alpha_research/capacity/` |
| G | DecayMonitor, DecayConfig, rolling IC/Sharpe, half-life, weight decay | `src/ai/alpha_research/decay/` |
| I | EdgePreservationRules, check_all | `src/ai/alpha_research/rules/` |
| J | ResearchPipeline, PipelineConfig, run_generation → run_validation → run_scoring → run_clustering → run_capacity_check | `src/ai/alpha_research/pipeline/` |

Phase F (shadow) and H (meta-alpha feedback) use existing `ShadowModelGovernance` and meta_alpha / meta_allocator; feed alpha health from DecayMonitor into meta-alpha inputs.
