# AI Autonomous Improvement Layer — Architecture

## Overview

The **AI Autonomous Improvement Layer** transforms the existing institutional platform into a **self-learning, adaptive trading intelligence engine**. All AI outputs are advisory; **no decision bypasses the risk engine** (Phase 7).

---

## High-Level Flow

```mermaid
flowchart TB
    subgraph Data["Market Data"]
        Bars[OHLCV Bars]
        Ticks[Ticks / Order Book]
        News[News / Macro]
    end

    subgraph Phase1["Phase 1: Feature Engineering"]
        Price[Price Features]
        Micro[Microstructure]
        RegimeFeat[Regime Features]
        Cross[Cross-Asset]
        FS[(Feature Store)]
    end

    subgraph Phase2["Phase 2: ML Prediction Engine"]
        XGB[XGBoost]
        LSTM[LSTM]
        Trans[Transformer]
        RL[RL Policy]
        Vol[Vol Model]
        Ensemble[Ensemble]
    end

    subgraph Phase3["Phase 3: Regime Detection"]
        HMM[HMM]
        Cluster[Clustering]
        RegimeOut[Regime Label]
    end

    subgraph Phase4["Phase 4: Meta-Allocator"]
        Perf[Per-Strategy Perf]
        Decay[Decay Detection]
        Alloc[Capital Allocation]
    end

    subgraph Phase5["Phase 5: Self-Learning Loop"]
        Drift[Concept Drift]
        Retrain[Retrain Pipeline]
        Backtest[Backtest]
        Replace[Replace if Better]
    end

    subgraph Phase6["Phase 6: LLM Layer"]
        Sentiment[News Sentiment]
        Macro[Macro Risk]
        Review[Strategy Review]
    end

    subgraph Risk["Phase 7: Risk-First Control"]
        VaR[VaR Check]
        DD[Drawdown Limit]
        Expo[Exposure Limit]
        CB[Circuit Breaker]
    end

    Bars --> Price & Micro & RegimeFeat & Cross
    Price & Micro & RegimeFeat & Cross --> FS
    FS --> XGB & LSTM & Trans & RL & Vol
    XGB & LSTM & Trans & RL & Vol --> Ensemble
    FS --> HMM & Cluster --> RegimeOut
    RegimeOut --> Alloc
    Perf --> Decay --> Alloc
    Ensemble --> Alloc
    Alloc --> VaR --> DD --> Expo --> CB
    Drift --> Retrain --> Backtest --> Replace
    News --> Sentiment & Macro
    Perf --> Review
    Sentiment & Macro & Review --> Risk
```

---

## Component Map

| Phase | Component | Location | Responsibility |
|-------|-----------|----------|----------------|
| 1 | Feature pipeline | `src/ai/feature_engineering/` | Price, microstructure, regime, cross-asset features; versioned write to feature store |
| 2 | Model registry | `src/ai/models/registry.py` | Versioning, performance tracker, auto-replace |
| 2 | Ensemble | `src/ai/models/ensemble.py` | XGBoost, LSTM, Transformer, RL, Vol; calibrated probabilities |
| 3 | Regime classifier | `src/ai/regime/classifier.py` | HMM + clustering + volatility; outputs regime label |
| 4 | Meta-allocator | `src/ai/meta_allocator/allocator.py` | Per-strategy Sharpe/win rate/drawdown; decay detection; risk-parity/Kelly weights |
| 5 | Self-learning | `src/ai/self_learning/` | Drift detection, retrain pipeline, backtest, replace |
| 6 | LLM layer | `src/ai/llm/` | Sentiment, macro risk, strategy review; **no order placement** |
| 7 | Risk gate | `src/risk_engine/` (existing) | All AI recommendations pass VaR, drawdown, exposure, circuit breaker |

---

## Risk-First Guarantee

- **LLM** only suggests risk parameter or weight changes; an operator or automated policy applies them after risk check.
- **Meta-allocator** output weights are capped and passed through `RiskManager`.
- **Ensemble** signals are treated as strategy signals: they go to the same order path and risk checks.
- **Regime** only enables/disables strategies or adjusts allocation; it does not bypass limits.

---

## Self-Learning Workflow

```mermaid
sequenceDiagram
    participant Scheduler
    participant Drift as Drift Detector
    participant Retrain as Retrain Pipeline
    participant Backtest
    participant Registry as Model Registry
    participant Risk

    Scheduler->>Drift: Weekly / on trigger
    Drift->>Drift: Check feature & target distribution
    Drift->>Retrain: Trigger if drift
    Retrain->>Retrain: Walk-forward train
    Retrain->>Backtest: Backtest new model
    Backtest->>Registry: Compare metrics
    Registry->>Registry: Replace if better
    Registry->>Risk: New model in use (same risk gates)
```

---

## Deployment

- **Feature pipeline**: runs on bar close (stream or batch); writes to feature store with version.
- **Models**: trained in batch (weekly or on drift); promoted via registry after backtest.
- **Regime**: updated every N bars; state cached in Redis.
- **Meta-allocator**: runs after strategy P&L update; outputs weights for next cycle.
- **LLM**: invoked on schedule or event (earnings, RBI); results stored and optionally applied via config.
- **Blue/green**: new model version deployed as new service or config; rollback by reverting model version.
