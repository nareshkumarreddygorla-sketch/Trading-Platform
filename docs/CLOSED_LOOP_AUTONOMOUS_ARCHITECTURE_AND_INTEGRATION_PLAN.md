# Closed-Loop Autonomous Trading Platform — Architecture and Integration Plan

**Objective:** Transform the current system (strong execution core, stub broker, no autonomous loop) into a **real closed-loop autonomous trading platform** with actionable module-level changes, data flows, invariants, and proof requirements.

**Scope:** Architectural completion — broker implementation, live market ingestion, signal-to-order loop, drift/regime enforcement, capital hardening, stress/chaos, optional multi-instance, audit/security, and reassessment.

---

## PHASE 1 — REAL BROKER IMPLEMENTATION

### 1.1 Module-level changes

| Component | File(s) | Change |
|-----------|---------|--------|
| **Session / auth** | `src/execution/angel_one_gateway.py` | Add `_session: SmartApi` (or equivalent from `angel_one` / `smartapi` package). In `connect()`: instantiate SmartApi with `api_key`, `api_secret`, `access_token`; call login/session init; store session. On 401/403: clear session, set `_session_expired = True`; expose for heartbeat. |
| **place_order** | `src/execution/angel_one_gateway.py` | In live mode: call SmartAPI place order API (e.g. `session.place_order(...)` or REST `POST /rest/secure/angelone/v1/order/place`). Map request to broker params (symbol token from master, exchange NSE/BSE, order_type, quantity, price, trigger_price if any). Parse response → extract `order_id` (broker order id), `oms_order_id` or equivalent as our order_id; return `Order(order_id=..., broker_order_id=..., status=OrderStatus.LIVE or PENDING)`. On HTTP/connection error → raise; caller (OrderEntryService) already has timeout and retry at resilience layer if used. |
| **cancel_order** | `src/execution/angel_one_gateway.py` | Call broker cancel API with `broker_order_id` (or order_id as per broker docs). Return True if 200/success, False otherwise. |
| **get_order_status** | `src/execution/angel_one_gateway.py` | Call broker order book or order details API; map broker status string to `OrderStatus` (e.g. open → LIVE, filled → FILLED, cancelled → CANCELLED, rejected → REJECTED, partial → PARTIALLY_FILLED). |
| **get_positions** | `src/execution/angel_one_gateway.py` | Call broker positions API; map each row to `Position(symbol, exchange, side, quantity, avg_price, ...)`. Return list. |
| **get_orders** | `src/execution/angel_one_gateway.py` | Call broker order list API with optional status filter; map to list of `Order`. |
| **Fill delivery** | New: `src/execution/angel_one_order_listener.py` | Either (A) WebSocket: subscribe to order update stream, on message parse fill/reject/cancel, map to `FillEvent`, call `app.state.fill_handler.on_fill_event(event)`. Or (B) Polling: background task every T sec calls `get_order_status` for active order_ids from lifecycle, diff filled_qty/status, emit FillEvent for changes. Prefer WebSocket if broker supports. |
| **Resilience wrapper** | `src/execution/resilience.py` | Already has timeout/retry for place_order. Ensure all gateway methods used by loop or recovery are wrapped: place_order (already), get_positions, get_order_status with timeout (e.g. 10s) and bounded retry (e.g. 3 attempts, exponential backoff 1s, 2s, 4s). On final failure: raise; caller (recovery, heartbeat) handles. |

### 1.2 Broker order state → lifecycle FSM mapping

| Broker state (SmartAPI typical) | FillType / OrderStatus | Action |
|---------------------------------|------------------------|--------|
| open / pending | — | OrderStatus.LIVE |
| filled | FillType.FILL | lifecycle FILLED; persist; position upsert |
| partial_fill | FillType.PARTIAL_FILL | lifecycle PARTIALLY_FILLED; persist; position upsert |
| cancelled | FillType.CANCEL | lifecycle CANCELLED; persist |
| rejected | FillType.REJECT | lifecycle REJECTED; persist |

Emit one `FillEvent` per broker event; FillHandler already supports FILL, PARTIAL_FILL, CANCEL, REJECT and lifecycle + persist.

### 1.3 Error handling and fallback

- **Timeout:** All broker calls via `asyncio.wait_for(..., timeout=10)` (or 30 for place_order already in OrderEntryService). On timeout: raise; OrderEntryService releases reservation and updates idempotency to REJECTED; no duplicate submit.
- **Retry:** Only for idempotent reads: get_order_status, get_positions. place_order: no retry (idempotency key at API layer). Retry with backoff in resilience layer for get_*.
- **Partial connectivity:** If get_positions fails during recovery → safe_mode. If WebSocket disconnects → restart listener or fall back to polling; do not submit new orders until fill stream is re-established (optional: gate AutonomousTradingLoop on fill_consumer_healthy flag).
- **Session expiry:** On 401/403 in any broker call: clear session, set _session_expired; next connect() re-login. Heartbeat (Phase 2 or 7) can call get_positions; on failure enter safe_mode.

### 1.4 Observability

- **Metrics:** `broker_calls_total{method, status}`, `broker_latency_seconds{method}`, `broker_session_expired_total`, `broker_fill_events_total`. In gateway: increment on entry/exit and on fill emission.
- **Logging:** Log every place_order/cancel request and response (order_id, symbol, side, qty; redact credentials). Log fill events with order_id, filled_qty, avg_price.

### 1.5 Data flow (Phase 1)

```
API POST /orders → OrderEntryService.submit_order → OrderRouter.place_order → AngelOneExecutionGateway.place_order()
  → SmartAPI HTTP → broker
  → return Order → idempotency update → lifecycle.register → persist_order

Broker WebSocket/poll → AngelOneOrderListener → FillEvent → FillHandler.on_fill_event
  → lifecycle.update_status, add_or_merge_position, persist_fill
```

### 1.6 Invariants preserved

- No duplicate broker submission (idempotency unchanged).
- All orders still go through OrderEntryService (no direct gateway from API).
- FillHandler remains single consumer of fill events; lifecycle FSM and persistence unchanged.

### 1.7 New failure scenarios covered

- Broker timeout: already handled in OrderEntryService (release reservation, idempotency REJECTED).
- Broker reject/cancel: FillEvent REJECT/CANCEL → persist and lifecycle.
- Session expiry: re-login on next connect(); optional safe_mode if login fails.

---

## PHASE 2 — LIVE MARKET DATA INGESTION

### 2.1 Ingestion service structure

| Component | Location | Responsibility |
|-----------|----------|----------------|
| **Ingestion loop** | New: `src/market_data/ingestion_service.py` | Async service: connect to connector (Angel One or other); subscribe_ticks(symbols) and/or subscribe_bars(symbols, interval); loop over stream_ticks()/stream_bars(); normalize; publish to Kafka and update QuoteCache; optionally write bars to bar store. |
| **Connector (Angel One)** | `src/market_data/connectors/angel_one.py` | Implement real WebSocket: connect to SmartAPI WebSocket URL with auth; send subscription message for symbols; on message parse LTP/depth → Tick or aggregate to Bar; push to _tick_queue/_bar_queue. Implement get_historical_bars via REST for gap-fill. |
| **Normalizer** | Existing `src/market_data/normalizer.py` | Ensure Tick/Bar have correct exchange, ts (timezone UTC), symbol. |
| **Publish** | Existing `MarketDataStream` (Kafka) | ingestion_service calls `await stream.publish_tick(tick)` or `publish_bar(bar)` after normalize. |
| **QuoteCache** | Existing `src/market_data/cache.py` | Update cache on each tick (or bar): key by symbol (and exchange if needed); set last price, ts. |
| **Bar store** | New: `src/persistence/bar_repo.py` or use TimescaleDB | Optional: insert bar into bars table (symbol, exchange, interval, ts, o, h, l, c, volume). Used by strategies and feature store. |

### 2.2 Design details

- **Multi-symbol:** Ingestion service holds list of symbols from config or strategy registry; subscribe_ticks(symbols) and subscribe_bars(symbols, "1m"). Re-subscribe on reconnect.
- **Backpressure:** Kafka producer send is async; if producer slow, use bounded queue (e.g. 10k ticks) and drop or log overflow. Bar stream typically lower volume; queue 1k bars.
- **Reconnect:** On WebSocket disconnect or exception in stream loop: sleep(backoff), backoff = min(backoff * 2, 60), then connect() and subscribe again. Emit metric `market_data_reconnects_total`.
- **Gap detection:** Periodically (e.g. every 5 min) compare last bar ts per symbol with current time; if gap > 2 intervals, call get_historical_bars(symbol, interval, gap_start, gap_end) and publish bars. Record `market_data_gaps_filled_total`.

### 2.3 Failure handling

- Connector connect() fails: retry with backoff; do not start strategy loop until at least one successful connect (or allow “replay only” mode with historical bars only).
- Kafka down: log and optionally buffer in memory (bounded); if buffer full, drop oldest. Metric `market_data_kafka_drops_total`.
- Redis (QuoteCache) down: skip cache update; Kafka still gets data. Strategies that need cache can fall back to “no cache” or skip cycle.

### 2.4 Data integrity

- Bars: ensure ts monotonic per symbol; on gap fill, insert in order.
- Ticks: best-effort ordering; no strong guarantee across symbols.

### 2.5 Data flow (Phase 2)

```
Broker WebSocket → AngelOneConnector (stream_ticks/stream_bars)
  → IngestionService loop
  → Normalizer
  → MarketDataStream.publish_tick / publish_bar → Kafka
  → QuoteCache.set (Redis)
  → BarRepo.insert (optional) → TimescaleDB
```

### 2.6 Modules impacted

- `src/market_data/connectors/angel_one.py` (real WebSocket + get_historical_bars)
- New: `src/market_data/ingestion_service.py`
- Optional: `src/persistence/bar_repo.py`, `src/persistence/models.py` (BarModel)
- App startup: register IngestionService as background task (start after recovery).

---

## PHASE 3 — CLOSED-LOOP SIGNAL ORCHESTRATION

### 3.1 AutonomousTradingLoop service

**New module:** `src/execution/autonomous_loop.py`

**Responsibilities:**

1. **Schedule:** Run on interval (e.g. every 60s) or on “new bar” event (Kafka consumer of market.bars). Configurable: `loop_interval_seconds` and/or `trigger_on_bar = True`.
2. **Single cycle:**
   - **Market snapshot:** Read from QuoteCache (last price per symbol) or from bar store (last N bars per symbol). Build `MarketState` (symbol, exchange, bars, latest_price, volume) for each symbol in universe.
   - **Features:** Call FeatureStore.read(symbol, from_ts, to_ts) if strategies need features; else use bars only. If feature store empty, strategies use bars (e.g. EMA/MACD from bars).
   - **Run strategies:** `StrategyRunner.run(state)` per symbol (or one state with multi-symbol bars). Collect all signals.
   - **Alpha ranking (cross-sectional):** Sort signals by score descending; take top N (e.g. 10) by config. Optionally run ResearchPipeline.run_validation on current bars for “live alpha” (heavy) or use pre-computed scores; for minimal viable, use strategy score only.
   - **Select top N:** `candidates = sorted_signals[:top_n]`.
   - **MetaAllocator:** `MetaAllocator.allocate(strategy_ids, equity, current_drawdown_pct, regime_multiplier, meta_alpha_scale)` → weights per strategy. Map signal to strategy_id; weight = allocator weight for that strategy.
   - **Position sizing:** For each candidate signal, `quantity = dynamic_position_fraction(..., regime_multiplier=regime_mult, ...) * equity / price` or use risk_manager.max_quantity_for_signal(price). Cap by allocator weight (e.g. max 20% of equity per signal).
   - **Regime multiplier:** From RegimeClassifier.get_regime(state) → regime_multiplier (e.g. 0.5 in high vol). Apply to quantity.
   - **Drawdown scaling:** risk_manager.check_drawdown(peak, current); if near limit, scale quantity down (e.g. 0.5).
   - **Exposure multiplier:** risk_manager.effective_equity() already uses exposure_multiplier; use it in max_quantity_for_signal.
   - **Submit:** For each candidate (respecting max_open_positions and reservation): build `OrderEntryRequest(signal=Signal(...), quantity=rounded_qty, limit_price=..., idempotency_key=derive_key(...))`; call `await order_entry_service.submit_order(request)`. Do not call gateway directly.

3. **Enforcement:**
   - No bypass of risk engine: all orders go through OrderEntryService.submit_order (which calls can_place_order, reservation, etc.).
   - Respect reservation: if submit_order returns RESERVATION_FAILED, stop submitting more this cycle.
   - Respect lifecycle: existing open orders for same symbol/side can be skipped or reduced (e.g. “target position” minus current = order size); optional.

4. **Idempotent cycle:** Idempotency key per order: include symbol, side, quantity, price, and cycle_ts (e.g. bar ts or cycle start time). So same bar/cycle does not submit duplicate; next cycle new key.

5. **Already-open positions:** When computing “target” position from signals, subtract current position (from risk_manager.positions or get_positions). Order only delta (or skip if already at target).

### 3.2 Scheduler design

- **Option A (async task in app):** In app lifespan, after recovery and FillHandler ready, start `asyncio.create_task(autonomous_loop.run_forever())`. `run_forever()`: while True: await run_one_cycle(); await asyncio.sleep(loop_interval_seconds). On exception: log, increment `autonomous_loop_errors_total`, sleep(60), continue.
- **Option B (bar-driven):** Kafka consumer for topic market.bars; on message decode bar; call run_one_cycle(trigger_ts=bar.ts). Backpressure: process one bar at a time; skip if previous cycle still running (or queue bar).
- **Backoff if broker unhealthy:** If order_entry_service.submit_order returns TIMEOUT or IDEMPOTENCY_UNAVAILABLE repeatedly (e.g. 3 times in a row), set loop_paused = True; sleep 5 min; then try again. Emit `autonomous_loop_paused_total`.

### 3.3 Dependencies to inject

- OrderEntryService (app.state.order_entry_service)
- RiskManager (from order_entry_service.risk_manager)
- StrategyRunner + StrategyRegistry (from app.state or new)
- MetaAllocator (from app.state or new)
- RegimeClassifier (optional)
- FeatureStore (optional)
- Config: universe symbols, top_n, loop_interval_seconds, max_orders_per_cycle

### 3.4 Data flow (Phase 3)

```
Timer or Kafka bar
  → AutonomousTradingLoop.run_one_cycle()
  → QuoteCache / BarStore → MarketState
  → FeatureStore.read (optional)
  → StrategyRunner.run(state) → signals
  → sort by score, top N
  → MetaAllocator.allocate(...) → weights
  → position sizing + regime + drawdown + exposure
  → for each candidate: OrderEntryRequest → OrderEntryService.submit_order()
  → (existing path: idempotency → kill → circuit → risk → reserve → broker → lifecycle → persist)
```

### 3.5 Invariants preserved

- All orders via OrderEntryService only.
- Reservation and lifecycle counts unchanged (order entry path unchanged).
- No new bypass of risk or idempotency.

### 3.6 New invariants

- At most M orders submitted per cycle (config max_orders_per_cycle).
- Loop does not run if safe_mode or kill_switch armed (check at start of cycle).

### 3.7 Modules impacted

- New: `src/execution/autonomous_loop.py`
- App: register loop task in lifespan; inject dependencies.
- Optional: `src/api/routers/trading.py` — add POST /trading/loop/pause, POST /trading/loop/resume (set loop_paused flag).

---

## PHASE 4 — DRIFT & REGIME ENFORCEMENT IN LIVE PATH

### 4.1 Before submit_order in AutonomousTradingLoop

- **Drift check:** After generating signals and before building OrderEntryRequest, call `MultiLayerDriftDetector` or `ConceptDriftDetector.detect(current_features)`. If drifted:
  - **Action:** Set strategy_enabled[strategy_id] = False for affected strategy (or reduce allocation weight to 0 for that strategy this cycle). Log and emit `drift_disable_strategy_total{strategy_id}`.
  - **Determinism:** Use same threshold as config; no randomness. Same features → same outcome.
  - **Audit:** Log event: { ts, type: "drift_disable", strategy_id, reason } (see Phase 8 audit table).
- **Regime gating:** Call `RegimeClassifier.get_regime(state)` (or volatility_regime + trend). Get `strategies_for_regime(regime)`. Filter signals: keep only signals from strategies in that set. Deactivate others for this cycle (do not submit their signals).

### 4.2 No flapping

- Drift: require drift to be true for 2 consecutive cycles before disabling (or use hysteresis: enable only after 3 consecutive non-drift). Config: `drift_confirm_cycles = 2`.
- Regime: regime is computed per cycle; strategies allowed set changes with regime. No extra hysteresis needed if regime is stable at 1-min bar.

### 4.3 Audit trail

- Every drift_disable and strategy re-enable (if re-enable logic exists): insert into audit_events table (Phase 8).
- Regime change: log and optional metric `regime_current` (already exists).

### 4.4 Modules impacted

- `src/execution/autonomous_loop.py`: add drift check and regime filter before building requests.
- `src/ai/drift/multi_drift.py` or `src/ai/self_learning/drift.py`: expose detect(features) with deterministic threshold.
- `src/ai/regime/classifier.py`: expose get_regime(state), strategies_for_regime(regime).
- New or extend: audit_events (Phase 8).

---

## PHASE 5 — CAPITAL INTELLIGENCE HARDENING

### 5.1 Add to RiskManager / limits

| Feature | Location | Implementation |
|---------|----------|----------------|
| **Volatility-based exposure scaling** | `src/risk_engine/manager.py`, `limits.py` | Add `current_volatility: Optional[float]`, `volatility_scale_threshold: float`. If current_volatility > threshold, multiply effective equity by `max(0.5, 1.0 - (vol - threshold) / threshold)`. Call `update_volatility(vol)` from loop or market data (e.g. rolling std of returns). |
| **Per-symbol capital cap** | `src/risk_engine/limits.py`, `manager.py` | Add `max_position_per_symbol_pct: float = 10.0`. In can_place_order: sum position value for same symbol across sides; if (current_symbol_value + new_order_value) / equity > cap, reject. |
| **Sector exposure cap** | `src/risk_engine/limits.py`, `manager.py` | Add `max_sector_pct: float = 25.0`. Signal or position must have sector (from config or symbol→sector map). Sum position value by sector; if sector_value/equity > max_sector_pct, reject orders that add to that sector. |
| **Consecutive loss auto-disable** | New: `src/risk_engine/consecutive_loss.py` or in manager | Track per-strategy last K trade outcomes (from FillHandler or from PnL updates). If strategy has N consecutive losses and cumulative loss > threshold, set strategy disabled (in registry or in allocator). Emit metric and audit. |
| **Portfolio VaR cap** | `src/risk_engine/manager.py` | limits.var_limit_pct already exists. In can_place_order: compute current portfolio VaR (e.g. parametric from positions and vol); if VaR/equity > var_limit_pct, reject. Requires position list and vol; approximate is acceptable. |

### 5.2 Integration

- **can_place_order:** Add checks in order: volatility scale (already via effective_equity if we set exposure mult from vol), per-symbol cap, sector cap, VaR cap. Any fails → LimitCheckResult(False, reason).
- **AutonomousTradingLoop:** When sizing, use risk_manager.max_quantity_for_signal(price) which uses effective_equity; ensure effective_equity is already scaled by vol if implemented. Consecutive loss: loop checks strategy enabled (from registry or allocator) before including in candidates.

### 5.3 Modules impacted

- `src/risk_engine/limits.py`: add fields and optional check_sector, check_var.
- `src/risk_engine/manager.py`: add update_volatility; in can_place_order add per-symbol sum, sector sum, VaR check; call new limit checks.
- New: `src/risk_engine/consecutive_loss.py` (or inside allocator): track outcomes, disable strategy.
- StrategyRegistry or MetaAllocator: support “disabled by consecutive loss” and exclude from allocate.

### 5.4 Invariants

- exposure ≤ effective_equity * limits (unchanged).
- New: per-symbol exposure ≤ max_position_per_symbol_pct * equity; sector exposure ≤ max_sector_pct * equity; portfolio VaR ≤ var_limit_pct * equity.

---

## PHASE 6 — STRESS & CHAOS PROOF

### 6.1 Stress harness

**New:** `tests/stress/test_concurrent_fills.py` and `tests/stress/test_rapid_signals.py`, `tests/stress/test_restart_during_fill.py`.

- **100 concurrent fills:**
  - Setup: start app (or in-process OrderEntryService + FillHandler + PersistenceService). Create 100 FillEvents for same symbol (e.g. 100 partial fills) and for different symbols (e.g. 50 symbols × 2). Feed all to FillHandler concurrently (asyncio.gather). Assert: final position rows in DB match sum of fills per (symbol, exchange, side); no duplicate order_id in order_events; version column consistent (no lost update).
- **1000 rapid signal submissions:**
  - Call OrderEntryService.submit_order 1000 times with different idempotency keys (or same key for 1000 duplicates). Assert: with same key, only one order_id returned; with different keys, either all accepted (if risk allows) or rejected by reservation/risk; no duplicate broker call (mock broker counts calls).
- **Restart during fill storm:**
  - Start app; submit 10 orders; before all fill, kill process. Restart app; run recovery. Assert: recovery completes; invariant validation passes; active orders count + positions consistent; no duplicate order_id in DB.

### 6.2 Chaos injection

**New:** `tests/chaos/` or `scripts/chaos_inject.py`.

- **Random DB latency:** Use a proxy or mock that delays session.commit() by random 0–2s. Run 50 concurrent persist_fill; assert no lost update (OCC retry or symbol lock).
- **Random Redis outage:** During test, set Redis to refuse connections (or use fakeredis and toggle). Submit orders; assert 503 idempotency_unavailable; no broker call.
- **Random broker timeout:** Mock gateway.place_order to sleep(35) sometimes. Assert OrderEntryService returns TIMEOUT; reservation released; idempotency updated to REJECTED.
- **Forced exception in persist_fill:** Mock persistence to raise on every 3rd persist_fill. Assert: order path still returns success (persist is best-effort after broker success); metric orders_fill_persist_failed_total incremented; eventually no unhandled exception.

### 6.3 Validation criteria

- No invariant violation (run _validate_recovery_invariants after stress).
- No duplicate orders (broker mock call count == unique idempotency keys accepted).
- No exposure > limits (after cycle, sum position value ≤ equity * max_position_pct * N checked in risk).

### 6.4 Load limits assumptions

- Document: “Tested with 100 concurrent fills, 1000 submissions/min; single instance. Beyond that, run dedicated load test.”
- Assumption: Symbol lock pool 256 is sufficient; DB connection pool sized for 4–8 concurrent writes.

---

## PHASE 7 — MULTI-INSTANCE SAFETY (OPTIONAL)

### 7.1 Distributed risk lock

- **Option A — Redis lock:** Before entering “reserve + broker” section in OrderEntryService, acquire Redis lock `trading:order_entry_global` with TTL 60s. Release after order submitted or rejected. All instances use same key → only one order at a time cluster-wide. Downside: serializes all orders globally.
- **Option B — Per-symbol Redis lock:** Acquire `trading:order_entry:{symbol}:{exchange}` so different symbols can run in parallel across instances. Reservation and risk state are still per-process; so total open count is not shared. To be safe: persist “reservation” as a row (order_id, symbol, exchange, side, quantity, ts) and have can_place_order check DB count of active + reserved. Complex.
- **Recommendation for 10/10:** Use Redis global lock for order submission so only one process submits at a time. Reservation and lifecycle remain in-memory on that single process. Other instances can serve read-only (positions, orders list from DB) and health. Write path: one instance only.

### 7.2 Shared reservation state (alternative)

- Persist reservations in Redis: key `reservation:{order_id}` with TTL 120s. reserve() adds key; release() removes. can_place_order: count keys reservation:* and add to DB active count. Requires all instances to use same Redis and same counting logic.

### 7.3 Modules impacted

- New: `src/execution/order_entry/distributed_lock.py` (Redis lock acquire/release).
- `src/execution/order_entry/service.py`: wrap reserve→broker→release in distributed lock acquire/release when config multi_instance=True.

### 7.4 Invariants

- Cluster-wide: at most one order in “between reserve and broker response” at any time (if global lock). Or: total reserved + active ≤ max_open_positions (if shared reservation).

---

## PHASE 8 — AUDIT & SECURITY COMPLETION

### 8.1 Structured audit log table

**New model:** `src/persistence/models.py` — `AuditEventModel`:

- id, ts, event_type (str: submit_order, cancel_order, kill_switch_arm, kill_switch_disarm, safe_mode_clear, drift_disable, regime_change, admin_*), actor (str: user_id or "system"), payload (JSON: order_id, reason, strategy_id, etc.).

**Write points:**

- submit_order: on success and on reject (event_type=order_submit_success / order_submit_reject), actor=request state user or "api".
- cancel_order: event_type=cancel_order.
- kill_switch arm/disarm: event_type=kill_switch_arm, kill_switch_disarm; payload={reason, detail}.
- safe_mode clear: event_type=safe_mode_clear; payload={}.
- drift disable: event_type=drift_disable; payload={strategy_id, reason}.
- regime change: event_type=regime_change; payload={regime, previous_regime}.
- Admin endpoints: event_type=admin_*; actor=user_id from JWT.

### 8.2 Authentication and RBAC

- **Auth layer:** Use FastAPI dependency: `get_current_user` that reads JWT from Authorization header, verifies signature, returns user_id and roles. Optional: OAuth2 with JWT.
- **Protect endpoints:** All routes under /admin/* require role "admin". POST /orders require "trader" or "admin". GET /orders, /positions require "trader" or "readonly". Apply dependency to routers.
- **User isolation (multi-tenant):** Add tenant_id (or user_id) to Order, Position (in API layer or DB). Filter list_orders, list_positions by current_user.tenant_id. Order submission: set order.strategy_id or metadata with tenant_id for audit.

### 8.3 Modules impacted

- New: `src/persistence/audit_repo.py`, AuditEventModel in models.py.
- New: `src/api/auth.py` (JWT decode, get_current_user, require_role).
- `src/api/routers/orders.py`, `trading.py`, `health.py`: add Depends(get_current_user), audit log write on submit/cancel/admin.
- `src/execution/order_entry/service.py`: accept optional actor for audit; or audit from API layer after submit.

---

## PHASE 9 — PROOF REQUIREMENTS SUMMARY

| Area | Modules impacted | Invariants preserved | New invariants | Failure scenarios covered | Load assumptions |
|------|-------------------|----------------------|----------------|----------------------------|------------------|
| Phase 1 Broker | angel_one_gateway, new order_listener, resilience | No duplicate submit; single order path | — | Timeout, reject, cancel, session expiry | 1 place_order at a time per idempotency key |
| Phase 2 Ingestion | connectors/angel_one, ingestion_service, cache, optional bar_repo | — | At most one writer per symbol bar stream | Reconnect, Kafka/Redis down, gap fill | Backpressure queue 10k ticks, 1k bars |
| Phase 3 Loop | autonomous_loop, app lifespan | All orders via OrderEntryService; reservation respected | Max M orders/cycle; no run if safe_mode/kill | Broker unhealthy backoff; cycle error isolation | 1 cycle per interval; no overlapping cycles |
| Phase 4 Drift/Regime | autonomous_loop, drift, regime classifier, audit | — | Drift disable deterministic; audit trail | Flapping avoided by confirm_cycles | — |
| Phase 5 Capital | risk_engine/manager, limits, consecutive_loss | exposure ≤ limits | Per-symbol, sector, VaR caps | Consecutive loss disable | — |
| Phase 6 Stress | tests/stress, tests/chaos | Recovery invariants; no duplicate orders | — | 100 fills, 1000 submits, restart storm, chaos | Documented limits |
| Phase 7 Multi-instance | distributed_lock, service | — | One active submission cluster-wide (if global lock) | Two instances no duplicate broker call | Global lock TTL 60s |
| Phase 8 Audit/Auth | audit_repo, auth, routers | — | All admin actions logged; only authenticated | Unauthorized 401 | — |

---

## PHASE 10 — FINAL SCORE REASSESSMENT (POST-IMPLEMENTATION)

After **full** implementation of Phases 1–8:

| Score | Justification |
|-------|----------------|
| **Infrastructure maturity: 8** | Real broker, live ingestion, fill delivery, timeouts, retry, optional multi-instance lock. Stress and chaos tests exist. Remaining: no proof at 10k orders/min; single-region. |
| **Autonomous AI maturity: 8** | Closed loop: market → strategies → ranking → allocator → sizing → submit. Drift and regime in live path. Alpha research still batch; live ranking is strategy-score only unless ResearchPipeline wired to live. |
| **Capital safety: 9** | All Phase 5 caps; stress proof; audit log; no bypass. Remaining: multi-instance shared state is optional; full distributed reservation not implemented. |
| **SaaS readiness: 7** | Auth, RBAC, audit log, user/tenant isolation in place. Remaining: no rate limit per user; no billing; no formal compliance certification. |

**If Phase 7 (multi-instance) is skipped:** Infrastructure 7.5, Capital safety 8 (single-instance only).  
**If Phase 6 (stress/chaos) is minimal:** Capital safety 8 until proven under load.

---

## IMPLEMENTATION ORDER

1. **Phase 1** — Broker (required for any real trading).  
2. **Phase 8** — Audit table and auth (required before production; can do minimal auth first).  
3. **Phase 2** — Ingestion (required for loop).  
4. **Phase 3** — AutonomousTradingLoop (closes the loop).  
5. **Phase 4** — Drift/regime in loop.  
6. **Phase 5** — Capital hardening.  
7. **Phase 6** — Stress and chaos tests.  
8. **Phase 7** — Multi-instance if required.

---

## NEW FILES AND CLASSES CHECKLIST

| Phase | New file | New class / component |
|-------|----------|------------------------|
| 1 | `src/execution/angel_one_order_listener.py` | `AngelOneOrderListener`: WebSocket or poll loop; parse broker events → FillEvent; call fill_handler.on_fill_event. |
| 1 | — | Gateway methods in `angel_one_gateway.py`: real SmartAPI calls (use `smartapi` or `angel_one` Python SDK per broker docs). |
| 2 | `src/market_data/ingestion_service.py` | `MarketDataIngestionService`: connect, subscribe, loop stream_ticks/stream_bars, normalize, publish Kafka, update QuoteCache, optional bar_repo. |
| 2 | `src/persistence/bar_repo.py` (optional) | `BarRepository`: insert_bars(bars), get_bars(symbol, interval, from_ts, to_ts). |
| 3 | `src/execution/autonomous_loop.py` | `AutonomousTradingLoop`: run_one_cycle(), run_forever(); deps: order_entry_service, risk_manager, strategy_runner, meta_allocator, regime_classifier, feature_store. |
| 4 | — | In autonomous_loop: drift check + regime filter before submit. |
| 5 | `src/risk_engine/consecutive_loss.py` (optional) | `ConsecutiveLossTracker`: update(strategy_id, pnl), is_disabled(strategy_id). |
| 6 | `tests/stress/test_concurrent_fills.py` | Pytest: 100 concurrent FillHandler.on_fill_event; assert DB consistency. |
| 6 | `tests/stress/test_rapid_signals.py` | Pytest: 1000 submit_order; mock broker; assert no duplicate broker call. |
| 7 | `src/execution/order_entry/distributed_lock.py` | `RedisDistributedLock`: acquire(key, ttl), release(key). |
| 8 | `src/persistence/models.py` (add) | `AuditEventModel`: id, ts, event_type, actor, payload (JSON). |
| 8 | `src/persistence/audit_repo.py` | `AuditRepository`: append(event_type, actor, payload). |
| 8 | `src/api/auth.py` | `get_current_user`, `require_roles(["admin"])`, JWT decode. |

---

## SMARTAPI / ANGEL ONE SDK NOTES

- Official SmartAPI docs: https://smartapi.angelone.in/docs . Python SDK: confirm package name (`angel-one` or `smartapi` on PyPI) and add to requirements.txt.
- Session: JWT from login API; refresh before expiry. Store in gateway; on 401 refresh or re-login.
- Order place: REST POST with body (symbol token, exchange, order_type, quantity, price, product). Response: order_id, uniqueorderid. Map to Order(order_id=..., broker_order_id=..., status=LIVE).
- WebSocket order updates: map broker status to FillType and OrderStatus per §1.2; emit FillEvent; call fill_handler.on_fill_event.

---

**End of plan.** This closes the gap between "strong execution engine" and "real closed-loop autonomous AI trading platform" with concrete modules, data flows, invariants, and proof requirements. This closes the gap between “strong execution engine” and “real closed-loop autonomous AI trading platform” with concrete modules, data flows, invariants, and proof requirements.
