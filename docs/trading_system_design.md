# Trading System Design

## Overview

This document describes the end-to-end trading system design including the order lifecycle, signal pipeline, risk management flow, and market data architecture.

---

## 1. Order Lifecycle

All orders flow through a single mandatory entry point: `OrderEntryService.submit_order()`. No code path may bypass this pipeline.

### Pipeline Steps (in order)

```
1. INPUT VALIDATION
   -> Validate request fields (symbol, quantity, price, side)
   -> Check for missing or invalid values

2. LOT SIZE ADJUSTMENT
   -> NSELotSizeValidator rounds quantity to valid lot size
   -> Rejects if quantity < minimum lot

3. CIRCUIT LIMIT CHECK
   -> CircuitLimitChecker verifies exchange circuit limits
   -> Rejects if symbol is in circuit halt

4. IDEMPOTENCY GUARD
   -> IdempotencyStore checks for duplicate submission key
   -> Returns cached result if duplicate detected

5. KILL SWITCH CHECK
   -> KillSwitch.is_armed() blocks all new orders when active
   -> Only position-reducing orders allowed when armed

6. RATE LIMIT CHECK
   -> OrderRateLimiter enforces orders-per-minute cap
   -> Rejects with rate_limit_exceeded if exhausted

7. CIRCUIT BREAKER CHECK
   -> CircuitBreaker.allow_order() checks market-wide circuit state
   -> OPEN state blocks all orders; HALF_OPEN allows limited orders

8. RISK CHECKS (15-step can_place_order)
   -> Basic validation (qty > 0, price > 0)
   -> Equity check (equity > 0)
   -> Consecutive loss check
   -> Daily loss limit
   -> Intraday rolling loss
   -> Open positions limit
   -> Position size limit (% of equity)
   -> Single trade loss limit
   -> Per-symbol concentration
   -> Sector concentration
   -> Sector cap
   -> Leverage limit (200% max)
   -> Institutional risk checks

9. EXPOSURE RESERVATION
   -> ExposureReservation.reserve() locks notional exposure
   -> Prevents over-allocation from concurrent orders

10. MARKET IMPACT ESTIMATION
    -> Estimates market impact in basis points
    -> Attaches impact estimate to order

11. ORDER ROUTING
    -> OrderRouter selects broker gateway
    -> Gateway submits to exchange (or paper simulator)

12. LIFECYCLE TRANSITION
    -> Order moves from PENDING -> SUBMITTED
    -> State persisted to database

13. PERSISTENCE
    -> Order record saved with full audit fields
    -> Order event published to Kafka

14. RESULT RETURN
    -> OrderEntryResult with success/failure, order_id, timing
```

### Order States

```
PENDING ──> SUBMITTED ──> PARTIALLY_FILLED ──> FILLED
                    │                             │
                    └──> CANCELLED               │
                    │                             │
                    └──> REJECTED                │
                    │                             │
                    └──> EXPIRED                 │
                                                  │
                              CLOSED <────────────┘
```

---

## 2. Signal Pipeline

### Signal Generation Flow

```
Market Data (Bars/Ticks)
    │
    v
Feature Engine (75+ features)
    ├── Price features (returns, volatility, momentum)
    ├── Technical indicators (RSI, MACD, Bollinger, ATR)
    ├── Microstructure features (spread, volume profile)
    ├── Regime features (HMM state, volatility regime)
    └── Cross-asset features (sector correlation)
    │
    v
AI Model Ensemble
    ├── XGBoost (direction prediction)
    ├── LSTM (sequence patterns)
    ├── Transformer (attention-based prediction)
    ├── RL Agent (entry/exit optimization)
    └── Sentiment (news/social analysis)
    │
    v
Ensemble Engine (weighted combination)
    │
    v
Signal (symbol, side, score, confidence, portfolio_weight)
    │
    v
Risk Gate (pre-trade risk filter)
    │
    v
Order Entry Pipeline
```

### Signal Schema

```python
Signal(
    strategy_id: str,      # Source strategy identifier
    symbol: str,           # Trading symbol (e.g., "RELIANCE")
    exchange: Exchange,    # NSE, BSE, NYSE, etc.
    side: SignalSide,      # BUY or SELL
    score: float,          # 0.0 to 1.0 confidence score
    portfolio_weight: float, # 0.0 to 1.0 allocation weight
    risk_level: str,       # LOW, NORMAL, HIGH
    price: float | None,   # Target price (optional)
)
```

---

## 3. Risk Management Architecture

### Risk Check Hierarchy

```
Level 1: ORDER-LEVEL CHECKS
    ├── Quantity > 0, Price > 0
    ├── Single trade loss limit
    └── Position size as % of equity

Level 2: PORTFOLIO-LEVEL CHECKS
    ├── Open positions limit
    ├── Daily loss limit (% of equity)
    ├── Intraday rolling loss window
    ├── Consecutive loss count
    └── Per-symbol concentration limit

Level 3: SYSTEMIC CHECKS
    ├── Sector concentration (% of equity)
    ├── Total leverage (200% cap)
    ├── Circuit breaker state
    └── Kill switch state

Level 4: ADVANCED RISK
    ├── VaR (parametric / historical / Monte Carlo)
    ├── CVaR / Expected Shortfall
    ├── Gap risk (overnight / weekend)
    ├── Volatility targeting
    └── Stress testing scenarios
```

### Circuit Breaker State Machine

```
                 ┌─────────────┐
     ┌───────────│   CLOSED    │<──────────────┐
     │           │ (normal)    │               │
     │           └──────┬──────┘               │
     │                  │ drawdown > threshold  │
     │                  v                       │
     │           ┌─────────────┐               │
     │           │    OPEN     │               │
     │           │ (halted)    │               │ recovery
     │           └──────┬──────┘               │ conditions
     │                  │ cooldown expired      │ met
     │                  v                       │
     │           ┌─────────────┐               │
     │           │  HALF_OPEN  │───────────────┘
     │           │ (testing)   │
     │           └─────────────┘
     │                  │ continued losses
     └──────────────────┘ (re-trip)
```

---

## 4. Market Data Architecture

### Data Sources

| Source | Type | Data | Status |
|--------|------|------|--------|
| Angel One WebSocket | Real-time | Ticks, quotes, order book | Implemented |
| Angel One REST | Historical | OHLCV bars | Implemented |
| Yahoo Finance | Fallback | OHLCV bars | Implemented |
| FII/DII Flow | Supplementary | Institutional flow data | Implemented |
| News Feeds | Supplementary | RSS/API news | Implemented |

### Data Flow

```
Data Sources
    │
    v
Connectors (Angel One WS / REST / Yahoo Finance)
    │
    v
Normalizer (unified schema, UTC timestamps)
    │
    ├──> Redis Cache (latest bars, quotes)
    ├──> Bar Aggregator (tick -> 1m/5m/1h/1d bars)
    ├──> Bar Cache (rolling window of OHLCV)
    ├──> ADV Cache (average daily volume)
    └──> PostgreSQL (historical persistence via OHLCV repo)
```

### Data Validation

- **Tick Validator**: Price bounds, timestamp ordering, gap detection
- **OHLC Validator**: High >= Low, Close within range, volume non-negative
- **Data Quality Monitor**: Staleness detection, anomaly flagging

---

## 5. Strategy Execution

### Strategy Types

| Type | Implementation | Examples |
|------|---------------|----------|
| Classical | `strategy_engine/classical.py` | EMA crossover, MACD, RSI, Bollinger |
| Momentum | `strategy_engine/momentum_breakout.py` | Breakout, trend following |
| Mean Reversion | `strategy_engine/mean_reversion.py` | Statistical arbitrage |
| ML-Based | `strategy_engine/ml_strategies.py` | Model-driven signals |
| High Win Rate | `strategy_engine/high_winrate.py` | Conservative high-probability |

### Execution Flow

```
Strategy Registry (discover & load strategies)
    │
    v
Strategy Runner (execute enabled strategies on each bar)
    │
    v
Signal Generation (each strategy produces Signal[])
    │
    v
Strategy Allocator (capital allocation across strategies)
    │
    v
Risk Gate (filter signals through risk checks)
    │
    v
Order Entry Pipeline (submit approved orders)
```

---

## 6. Paper vs Live Trading

The system supports two execution modes controlled by the `PAPER_MODE` environment variable:

| Aspect | Paper Mode | Live Mode |
|--------|------------|-----------|
| Order Execution | `PaperFillSimulator` | Angel One gateway |
| Fill Generation | Simulated with configurable slippage | Real broker fills |
| Position Tracking | In-memory + database | In-memory + database |
| Risk Checks | Full pipeline (identical) | Full pipeline (identical) |
| Market Data | Real or historical | Real-time only |
| Kill Switch | Active | Active |

**Important**: Risk checks are identical in both modes. Paper mode is for validation, not relaxed testing.
