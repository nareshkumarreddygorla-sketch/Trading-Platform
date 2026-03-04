# Institutional-Grade Bug Audit — Autonomous AI Trading Platform

**Classification:** Adversarial, capital-at-risk. Assume production with live capital.

---

## PHASE 1 — ARCHITECTURAL REVIEW

### 1.1 API order placement bypasses risk engine — CRITICAL

| Item | Detail |
|------|--------|
| **Severity** | Critical |
| **Component** | `src/api/routers/orders.py` |
| **Root cause** | `POST /orders` is a stub: no call to `RiskManager.can_place_order()`, no `OrderRouter`, no `OrderLifecycle.register()`. Any client can post and receive `order_id: stub` without any risk or execution path. |
| **Reproduction** | `curl -X POST /api/v1/orders -d '{"symbol":"RELIANCE","side":"BUY","quantity":1000000}'` → 200 OK. |
| **Financial impact** | When wired to a real gateway, orders could be placed without position/drawdown/circuit checks → unlimited size, circuit breaker bypass. |
| **Fix** | In `place_order` handler: resolve shared `RiskManager` and `OrderRouter`; call `risk_manager.can_place_order(signal, qty, price)`; if not allowed return 403; else call router and lifecycle.register(order). |
| **Improvement** | Single order-entry path (API + any internal caller) through a dedicated `OrderEntryService` that enforces risk → router → lifecycle. |

### 1.2 OrderRouter has no risk check — CRITICAL

| Item | Detail |
|------|--------|
| **Severity** | Critical |
| **Component** | `src/execution/order_router.py` |
| **Root cause** | `OrderRouter.place_order()` calls gateway directly. No invocation of `RiskManager` or `AIRiskGate`. Any code that holds the router can place orders without risk checks. |
| **Reproduction** | Instantiate `OrderRouter(gateway)`, build a `Signal`, call `router.place_order(signal, quantity=999999, ...)` → order sent to broker. |
| **Financial impact** | Direct capital loss; bypass of position, daily loss, and circuit breaker limits. |
| **Fix** | Router must not be callable without prior risk check, or router receives `RiskManager` and calls `can_place_order` before gateway.place_order; on reject, return a structured error and do not call gateway. |
| **Improvement** | Router accepts an optional `RiskGate` (or `RiskManager`); `place_order` runs check first and returns `Result[Order, LimitCheckResult]`. |

### 1.3 RiskManager position count never updated — CRITICAL

| Item | Detail |
|------|--------|
| **Severity** | Critical |
| **Component** | `src/risk_engine/manager.py`, execution path |
| **Root cause** | `can_place_order` uses `len(self.positions)`. No code path calls `add_position()` when an order is placed or filled, or `remove_position()` when closed. Positions are never updated from execution. |
| **Reproduction** | Place 20 orders (max_open_positions=10). Every order passes because `len(self.positions)` stays 0. |
| **Financial impact** | `max_open_positions` is ineffective; portfolio can accumulate unlimited open positions. |
| **Fix** | On order fill: call `risk_manager.add_position(Position(...))`. On close/square-off: call `risk_manager.remove_position(symbol, exchange)`. Persist positions from broker or from lifecycle so risk state is authoritative. |
| **Improvement** | Risk state (positions, daily_pnl) is the single source of truth, updated from execution events (fill/close) or reconciled with broker positions on startup. |

### 1.4 Zero/negative equity allows orders — CRITICAL

| Item | Detail |
|------|--------|
| **Severity** | Critical |
| **Component** | `src/risk_engine/limits.py` — `check_daily_loss` |
| **Root cause** | `if equity <= 0: return LimitCheckResult(True)` — orders are allowed when equity is zero or negative. |
| **Reproduction** | Set `risk_manager.equity = 0`; call `can_place_order(signal, 100, 100.0)` → allowed. |
| **Financial impact** | After a total loss or data error, system continues to allow orders. |
| **Fix** | `if equity <= 0: return LimitCheckResult(False, "zero_or_negative_equity")`. |
| **Improvement** | All limit checks explicitly reject when equity <= 0 (or when capital is not yet loaded). |

### 1.5 Negative quantity / zero price not rejected — HIGH

| Item | Detail |
|------|--------|
| **Severity** | High |
| **Component** | `src/risk_engine/manager.py` — `can_place_order` |
| **Root cause** | No validation that `quantity > 0` and `price > 0`. Negative quantity yields negative `position_value`; `check_position_size(equity, -x)` computes negative pct and does not trigger limit, so order is allowed. |
| **Reproduction** | `can_place_order(signal, quantity=-100, price=500)` → allowed. |
| **Financial impact** | Short orders could be placed via negative quantity if gateway interprets it; or garbage orders. |
| **Fix** | At start of `can_place_order`: if quantity <= 0 or price <= 0 return `LimitCheckResult(False, "invalid_quantity_or_price")`. |
| **Improvement** | Validate quantity (int/float > 0) and price > 0 in one place (e.g. RiskManager or OrderEntryService). |

### 1.6 Redis/Kafka failure handling — HIGH

| Item | Detail |
|------|--------|
| **Severity** | High |
| **Component** | `src/market_data/cache.py`, `src/market_data/streaming.py` |
| **Root cause** | Redis: no try/except on `setex`/`get`; connection failure raises and can crash the caller. Kafka: `publish_tick`/`publish_bar` catch Exception and log but do not retry or DLQ; failed publishes are silent data loss. |
| **Reproduction** | Kill Redis during set_tick → unhandled exception. Stop Kafka → publish fails once, logged; no retry, no backpressure. |
| **Financial impact** | Strategy sees stale or missing data; mispricing or no signal. |
| **Fix** | Redis: wrap in retry + fallback (e.g. skip cache or in-memory fallback); never let connection errors crash the process. Kafka: retry with backoff; on final failure write to DLQ or at least structured alert. |
| **Improvement** | Circuit breaker for Redis/Kafka; health checks; degrade gracefully (e.g. skip cache, or reject new orders if data stale). |

### 1.7 Circuit breaker not enforced at API — MEDIUM

| Item | Detail |
|------|--------|
| **Severity** | Medium |
| **Component** | API and any direct use of OrderRouter |
| **Root cause** | Circuit breaker state lives in `RiskManager._circuit_open`. API does not consult it; only code that explicitly calls `can_place_order` gets blocked. So if API is later wired to router but not to risk, circuit can be open and API still place orders. |
| **Fix** | Ensure the only path to place orders is through `RiskManager.can_place_order()` (which checks circuit). Document and enforce single order-entry path. |

---

## PHASE 2 — STRATEGY ENGINE AUDIT

### 2.1 Backtest uses future bar for execution (lookahead) — HIGH

| Item | Detail |
|------|--------|
| **Severity** | High |
| **Component** | `src/backtesting/engine.py` |
| **Root cause** | For bar index `i`, `state` is built from `window = bars[max(0,i-100):i+1]` and `latest = window[-1]` = bar at `i`. Signal is generated from that bar’s close; then we execute at `latest.close` (same bar). So we’re executing in the same bar we used for the signal — acceptable only if we assume fill at bar close. No latency_bars applied (config has `latency_bars: int = 0`). So no lookahead in the sense of “next bar”, but execution is same-bar close; any real latency would make fills occur on next bar. |
| **Reproduction** | Run backtest with 1m bars; compare to live with 1–2 bar delay; backtest will overstate performance. |
| **Fix** | Apply `latency_bars`: when signal is generated at bar i, execute fill at bar i+latency_bars (or at that bar’s open/close per policy). Use that bar’s price for fill. |
| **Improvement** | Document execution assumption (e.g. fill at bar close vs next open). Support configurable fill model. |

### 2.2 Backtest equity accounting wrong (capital not deducted) — CRITICAL

| Item | Detail |
|------|--------|
| **Severity** | Critical |
| **Component** | `src/backtesting/engine.py` |
| **Root cause** | On BUY: `cost = fill_price * (equity * 0.05 / fill_price)`; only `equity -= commission` is applied. Cash is never reduced by `cost`. Then `equity_curve.append(equity + position * latest.close)` adds MTM to the same (uncorrected) cash. So we never subtract cost of purchase → equity is overstated (free leverage). |
| **Reproduction** | Run backtest on any strategy that buys; compare equity curve to manual cash + position * price; backtest curve is higher. |
| **Financial impact** | Backtest metrics (Sharpe, drawdown, CAGR) are invalid; strategies promoted to live on false performance. |
| **Fix** | On BUY: `equity -= cost; equity -= commission`. Then `equity_curve.append(equity + (position * latest.close if position > 0 else 0))` is correct (cash + MTM). |
| **Improvement** | Separate cash and position in backtest state; compute equity = cash + position * mark; unit test with known path. |

### 2.3 Signal duplication / multiple strategies same symbol — MEDIUM

| Item | Detail |
|------|--------|
| **Severity** | Medium |
| **Component** | `src/strategy_engine/runner.py` |
| **Root cause** | Runner aggregates all signals and sorts by score. If EMA and MACD both emit BUY on same symbol, we get two signals. No deduplication by (symbol, side); no per-symbol cap. Downstream (e.g. “execute first signal only”) may send two orders for same symbol. |
| **Reproduction** | Enable EMA and MACD; feed state where both fire BUY → two BUY signals for same symbol. |
| **Fix** | Either: (a) aggregate per (symbol, side) (e.g. max score or weighted), or (b) enforce one order per (symbol, side) per cycle and take top signal per symbol. |
| **Improvement** | Define clear policy: one open position per symbol, or allow multiple strategies same symbol with explicit allocation. |

### 2.4 Bar indexing / warm check — LOW

| Item | Detail |
|------|--------|
| **Severity** | Low |
| **Component** | Strategies use `state.bars[-1]` and rolling; backtest passes `window = bars[:i+1]`. Bars are aligned (no off-by-one). RSI/MACD use `.iloc[-1]`, `.iloc[-2]` on pandas; consistent with “current bar” and “previous bar”. No clear indicator misalignment. |

---

## PHASE 3 — ML & FEATURE PIPELINE AUDIT

### 3.1 NaN propagation in ensemble and features — HIGH

| Item | Detail |
|------|--------|
| **Severity** | High |
| **Component** | `src/ai/models/ensemble.py`, feature pipeline outputs |
| **Root cause** | Ensemble sums `prob_up * w`; if any model returns NaN, result is NaN. No sanitization of `PredictionOutput`. Feature pipeline does not replace or drop NaN in computed features; NaN written to store and fed to models. |
| **Reproduction** | Return `PredictionOutput(prob_up=float('nan'), ...)` from one model → ensemble returns NaN; downstream may use it as score or size multiplier. |
| **Financial impact** | Incorrect position size or invalid signal; possible exception or extreme size if used naively. |
| **Fix** | Ensemble: after aggregation, clamp or replace NaN (e.g. prob_up = 0.5, confidence = 0). Feature pipeline: replace NaN with 0 or drop row; validate before write. |
| **Improvement** | Validate all model outputs (prob in [0,1], confidence in [0,1], no NaN/Inf); validate feature dict before inference. |

### 3.2 Incomplete feature vector at inference — MEDIUM

| Item | Detail |
|------|--------|
| **Severity** | Medium |
| **Component** | `src/ai/models/ensemble.py`, XGBPredictor |
| **Root cause** | XGBPredictor builds `X` from `features.get(n, 0)` for missing keys. So missing features become 0. Train might have had different missingness or no zeros → train/serve skew. |
| **Fix** | Require a fixed feature set; if any key is missing, return low-confidence prediction or skip inference; log and alert. |
| **Improvement** | Feature contract (schema) and validation at inference; version features and model together. |

---

## PHASE 4 — RISK ENGINE AUDIT

### 4.1 check_daily_loss allows orders when equity <= 0 — CRITICAL

| Item | Detail |
|------|--------|
| **Severity** | Critical |
| **Component** | `src/risk_engine/limits.py` |
| **Root cause** | See 1.4. |
| **Fix** | Reject when equity <= 0. |

### 4.2 No single-trade loss check in can_place_order — MEDIUM

| Item | Detail |
|------|--------|
| **Severity** | Medium |
| **Component** | `src/risk_engine/manager.py` |
| **Root cause** | `RiskLimits.check_single_trade_loss` exists but is never called in `can_place_order`. So max loss per trade is not enforced at order time. |
| **Fix** | Before placing, estimate worst-case loss for the order (e.g. quantity * price * stop or a fixed %). Call `check_single_trade_loss(equity, -worst_case_loss)` and reject if not allowed. |
| **Improvement** | Centralize all limit checks in one sequence; add tests for each limit. |

### 4.3 VaR/Kelly numerical edge cases — LOW

| Item | Detail |
|------|--------|
| **Severity** | Low |
| **Component** | `src/risk_engine/metrics.py` |
| **Root cause** | `kelly_fraction` clips to [0,1]; `var_parametric` uses scipy; returns with NaN/Inf could propagate. |
| **Fix** | Filter NaN/Inf from returns before VaR/Kelly; document that returns are simple (not log) for cumprod. |

---

## PHASE 5 — EXECUTION ENGINE AUDIT

### 5.1 No idempotency / duplicate order risk — HIGH

| Item | Detail |
|------|--------|
| **Severity** | High |
| **Component** | `src/execution/angel_one_gateway.py`, order placement path |
| **Root cause** | Every call to `place_order` generates new UUID and sends to broker. No idempotency key; retries or duplicate requests (e.g. double-click, replay) create duplicate orders. |
| **Reproduction** | Call place_order twice with same signal/quantity in quick succession → two orders. |
| **Fix** | Accept idempotency key (e.g. client-provided or hash(signal, ts, strategy_id)); persist (Redis/DB) “key → order_id”; on replay return existing order_id and do not call broker again. |
| **Improvement** | All order placement goes through an idempotent layer; broker gateway is called only once per key. |

### 5.2 OrderLifecycle overwrite for empty order_id — MEDIUM

| Item | Detail |
|------|--------|
| **Severity** | Medium |
| **Component** | `src/execution/lifecycle.py` |
| **Root cause** | `register(order)` uses `order.order_id or ""`. Two orders with `order_id=None` both map to key `""` and the second overwrites the first. |
| **Fix** | Reject registration when `order_id` is None or empty; or generate and set order_id before register. |
| **Improvement** | order_id is always set by gateway or by lifecycle before register. |

### 5.3 No partial fill / fill updates from broker — MEDIUM

| Item | Detail |
|------|--------|
| **Severity** | Medium |
| **Component** | Execution path |
| **Root cause** | Lifecycle has `update_status(order_id, status, filled_qty, avg_price)` but no code calls it when broker sends fill updates. So positions and risk state cannot be updated from real fills. |
| **Fix** | Webhook or poll from broker for order updates; on fill/partial fill call lifecycle.update_status and risk_manager.add_position/remove_position (or reconcile). |
| **Improvement** | Event-driven fill handling; risk state always in sync with broker. |

---

## PHASE 6 — BACKTEST ENGINE VALIDATION

### 6.1 Backtest equity accounting — CRITICAL

| Item | Detail |
|------|--------|
| **Severity** | Critical |
| **Component** | `src/backtesting/engine.py` |
| **Root cause** | See 2.2. |
| **Fix** | Deduct cost on BUY. |

### 6.2 Slippage model symmetric — LOW

| Item | Detail |
|------|--------|
| **Severity** | Low |
| **Component** | `src/backtesting/slippage.py` |
| **Root cause** | Slippage is symmetric (bps); real market often has worse sell slippage in stress. Acceptable for first order; document. |

---

## PHASE 7 — CONCURRENCY & ASYNC AUDIT

### 7.1 RiskManager not thread-safe — HIGH

| Item | Detail |
|------|--------|
| **Severity** | High |
| **Component** | `src/risk_engine/manager.py` |
| **Root cause** | `positions`, `daily_pnl`, `_circuit_open` are plain attributes. Two concurrent calls to `can_place_order` can both see the same position count and both pass; then two orders placed. |
| **Reproduction** | Run 10 concurrent place_order requests; max_open_positions=5; more than 5 can pass. |
| **Fix** | Protect risk checks and state updates with a lock (e.g. asyncio.Lock or threading.Lock). Serialize `can_place_order` and `add_position`/`remove_position`/`register_pnl`. |
| **Improvement** | Single-threaded risk service or strict locking; consider event-sourced risk state. |

### 7.2 OrderLifecycle dict not thread-safe — MEDIUM

| Item | Detail |
|------|--------|
| **Severity** | Medium |
| **Component** | `src/execution/lifecycle.py` |
| **Root cause** | `_orders` and `_placed_at` are dicts; concurrent register/update_status can cause races. |
| **Fix** | Use a lock around register and update_status, or use thread-safe structures. |

---

## PHASE 8 — SECURITY & CONFIG AUDIT

### 8.1 API keys in config/env — MEDIUM

| Item | Detail |
|------|--------|
| **Severity** | Medium |
| **Component** | `config/settings.example.yaml`, `src/core/config.py` |
| **Root cause** | Example shows commented `angel_one_api_key`; if real values are in settings.yaml and it is ever committed or copied, keys leak. |
| **Fix** | Never commit secrets; use env-only or secret manager for API keys; add settings.yaml to .gitignore if it can contain secrets. |
| **Improvement** | All secrets from env or vault; config files only non-sensitive. |

### 8.2 PlaceOrderRequest unbounded quantity — MEDIUM

| Item | Detail |
|------|--------|
| **Severity** | Medium |
| **Component** | `src/api/routers/orders.py` |
| **Root cause** | No validation on `body.quantity` (could be negative or huge). When API is wired to gateway, this could send invalid orders. |
| **Fix** | Validate quantity > 0 and quantity <= MAX_SAFE_INT or configurable cap; return 400 otherwise. |

---

## PHASE 9 — OBSERVABILITY GAPS

### 9.1 Risk rejections not always logged — MEDIUM

| Item | Detail |
|------|--------|
| **Severity** | Medium |
| **Component** | `src/risk_engine/manager.py` |
| **Root cause** | `can_place_order` returns LimitCheckResult(False, reason) but caller may not log it. So we can have no trace of why an order was rejected. |
| **Fix** | Log at INFO when can_place_order returns not allowed (symbol, reason, quantity, price). Or log inside can_place_order. |
| **Improvement** | Structured audit log for every order attempt (allowed/rejected + reason). |

### 9.2 Kafka publish failure is silent data loss — HIGH

| Item | Detail |
|------|--------|
| **Severity** | High |
| **Component** | `src/market_data/streaming.py` |
| **Root cause** | Exception is logged but not re-raised; caller does not know publish failed. Downstream consumers miss data; no alert. |
| **Fix** | At least increment a Prometheus counter (e.g. kafka_publish_failures_total); optionally retry and DLQ. |
| **Improvement** | Alert on publish failure rate; DLQ for failed messages. |

---

## PHASE 10 — CAPITAL DESTRUCTION SCENARIOS

### 10.1 Scenario: API wired to live gateway without risk — CRITICAL

| Item | Detail |
|------|--------|
| **Scenario** | Developer wires `POST /orders` to `OrderRouter` and Angel One live gateway but forgets to call `RiskManager.can_place_order`. |
| **Why** | orders.py currently has TODO and returns stub; when implemented, risk can be omitted. |
| **Result** | Any client can send quantity=1000000; order hits broker; capital loss. |
| **Mitigation** | Single order-entry service that always enforces risk; integration test that asserts order is rejected when risk says no. |

### 10.2 Scenario: Concurrent orders exceed max_open_positions — CRITICAL

| Item | Detail |
|------|--------|
| **Scenario** | 10 requests call can_place_order concurrently; each sees len(positions)==0; all pass; 10 orders placed. |
| **Why** | No lock; positions not updated from fills anyway, so even single-threaded we could exceed if we ever start updating positions. |
| **Result** | More positions than limit; concentration and leverage beyond policy. |
| **Mitigation** | Lock around risk check and state update; or single-threaded order executor. |

### 10.3 Scenario: Backtest approves strategy that loses money live — CRITICAL

| Item | Detail |
|------|--------|
| **Scenario** | Backtest shows high Sharpe because equity is overstated (cost not deducted). Strategy goes live and loses. |
| **Why** | Backtest equity accounting bug (2.2). |
| **Result** | Capital allocated to a strategy that only looked good due to bug. |
| **Mitigation** | Fix backtest; add unit test with known cash flows; compare backtest vs live logic. |

---

## Summary Table

| Severity  | Count |
|-----------|-------|
| Critical  | 7     |
| High      | 8     |
| Medium    | 11    |
| Low       | 3     |

**Immediate actions:** Fix all Critical and High items before any live trading. Enforce single order-entry path through risk, fix backtest accounting, add idempotency and locking, and sanitize NaN in AI outputs.
