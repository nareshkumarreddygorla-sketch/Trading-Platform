# Execution Layer Redesign — Institutional-Grade Safety

## 1. Single Order Entry Pipe

**Design:** All order flows (API, AI, manual, test) MUST go through `OrderEntryService.submit_order(request)`.

**Pipeline (enforced in code):**
1. Validate input (quantity, price, symbol, side)
2. Idempotency check (Redis key → return existing order_id, do NOT call broker)
3. Global kill-switch check (if armed, allow only reduce-only orders)
4. Circuit breaker check (risk_manager.is_circuit_open())
5. RiskManager.can_place_order(signal, quantity, price)
6. Atomic position reservation (ExposureReservation.reserve under lock)
7. OrderRouter.place_order(signal, quantity, ...)
8. OrderLifecycle.register(order)
9. Persist order state (optional callback)
10. Publish to Kafka (optional callback)
11. Return order_id

**Preventing bypass:** No code path should call `OrderRouter.place_order` or gateway directly. The only way to get an order to the broker is via `OrderEntryService.submit_order`. Enforce by:
- API `POST /orders` resolves OrderEntryService from app state and calls submit_order only.
- AI engine / strategy runner receives OrderEntryService and calls submit_order only.
- OrderRouter and gateway are not exported to API or strategy layer; they are internal to execution.

**Class design:**
- `OrderEntryRequest(signal, quantity, order_type, limit_price, idempotency_key, source)`
- `OrderEntryResult(success, order_id, broker_order_id, reject_reason, reject_detail, latency_ms)`
- `OrderEntryService(risk_manager, order_router, lifecycle, idempotency_store, kill_switch, reservation, persist_order, publish_order_event)`
- Method: `async submit_order(request) -> OrderEntryResult`

---

## 2. Concurrency + Atomicity

**Lock strategy:** One global `asyncio.Lock` in OrderEntryService for steps 5–6 (risk check + reservation). This prevents two concurrent requests from both passing risk and exceeding limits.

**Reservation:** Before calling broker, `ExposureReservation.reserve(order_id_placeholder, ...)` counts reserved slots and position value. Risk check uses `len(positions) + reservation.reserved_count()` so that pending orders consume capacity. If broker fails, `reservation.release(placeholder)` frees the slot.

**Rollback:** On broker exception, `reservation.release(order_id_placeholder)` is called so the slot is available for the next order.

**Deadlock avoidance:** Lock is held only for risk check + reserve (in-memory). Broker call is outside the lock. No nested locks.

**Thread-safe RiskManager:** RiskManager itself is not thread-safe; the single global lock in OrderEntryService serializes all submissions, so only one thread/coro modifies risk state at a time. FillHandler updates (add_position, register_pnl) should also be serialized (e.g. same lock or a dedicated fill lock).

---

## 3. Idempotency System

**Flow:** Client sends `idempotency_key` (or server derives from request hash). Store in Redis: `idem:{key}` → `{order_id, broker_order_id, status, ts}` with TTL (e.g. 48h).

**Duplicate request:** If key exists and status is terminal or pending, return stored order_id and do NOT call broker again.

**Failure cases:**
- Broker success but network failure before response: client retries with same key → we return stored order_id (we must persist to Redis AFTER broker success; current design persists after broker return).
- Crash after broker success but before Redis set: client retries → we call broker again → duplicate order. Mitigation: persist order_id to Redis before calling broker (optimistic), or use broker idempotency key if supported.

**Expiration:** TTL 48h; after TTL key is gone, same key retry is treated as new (client should use new key for genuine retry after long delay).

**Memory pressure:** Redis eviction policy (volatile-lru); key size small (order_id + status).

**Crash recovery:** On startup, do not clear idempotency store; reconcile with broker positions so we know which order_ids are live.

---

## 4. Position + PnL Reconciliation

**FillHandler:** On broker WebSocket (or poll) fill event: update OrderLifecycle (status, filled_qty, avg_price), call risk_manager.add_position for new fill, and for close call remove_position + register_pnl(realized_pnl). Persist and emit metric.

**Partial fill:** Update lifecycle with filled_qty and remaining; add_position with filled_qty only (or accumulate; design depends on whether we track lots).

**Cancel:** Lifecycle update to CANCELLED; release reservation (reservation is already released when we don’t commit; for cancel we don’t add position).

**Reconciliation job:** Periodic (e.g. every 1–5 min) fetch broker positions via gateway.get_positions(), compare to risk_manager.positions. On mismatch: log, emit fill_mismatch_total, optionally arm kill switch (KillReason.FILL_MISMATCH) and open circuit.

**Emergency freeze:** If mismatch detected, CircuitAndKillController.trip_fill_mismatch() arms kill switch and opens circuit so no new orders until manual review.

---

## 5. Circuit Breaker + Global Kill Switch

**Hard max daily loss:** CircuitKillController.check_daily_loss_and_trip(current_equity) opens circuit and arms kill switch (KillReason.MAX_DAILY_LOSS).

**Max drawdown:** check_drawdown_and_trip(peak_equity, current_equity) opens circuit and arms kill (MAX_DRAWDOWN).

**Manual kill:** Admin endpoint `POST /admin/kill_switch/arm` with reason=MANUAL; KillSwitch.arm(KillReason.MANUAL).

**Auto kill on:** Rejection spike (N rejections in M seconds), broker latency spike (latency > threshold), fill mismatch (reconciliation), India VIX spike (e.g. VIX > 2x reference).

**Kill switch behavior:** When armed, OrderEntryService rejects new orders unless allow_order_reduce_only(symbol, side, qty, net_position) is True (e.g. SELL when long to reduce).

**Atomic and immediate:** Kill state is in-memory (KillSwitch._state under asyncio.Lock); check at step 3 of pipeline so no broker call after arm.

---

## 6. Backtest Execution Model

**FillModel:** Same interface for backtest and live sim: latency_bars (fill at bar i+latency_bars), slippage (bps), spread (bps), max_volume_participation_pct, commission_pct. execute_at_bar_index(signal_bar_index, bars, side, requested_qty, price_hint) returns (fill_bar, fill_price, fill_qty, commission).

**No same-bar fill:** Signal at bar i → fill at bar i+latency_bars using that bar’s open/close and volume.

**Volume participation:** fill_qty = min(requested_qty, bar_volume * max_volume_participation_pct / 100).

**BacktestEngine:** Uses FillModel for BUY/SELL; deducts cost on BUY, adds PnL on SELL; supports partial fills.

---

## 7. Broker Failure Resilience

**Timeout:** Order placement wrapped in asyncio.wait_for(..., timeout=15s). On timeout, release reservation and retry (BrokerGatewayResilient: max_retries=2).

**WebSocket disconnect:** Reconnect with exponential backoff; on reconnect subscribe again and optionally poll orders for missed updates.

**Partial fill with no callback:** Reconciliation job will detect (broker has position we don’t); trigger mismatch and freeze.

**Order accepted but no response:** Retry is dangerous (duplicate). Rely on idempotency key and broker-side idempotency if supported; otherwise poll order status and match by client order_id.

**Rate limiting:** Backoff on 429; track broker_error_total by type.

**DLQ for Kafka:** On publish failure after retries, write to DLQ topic or table; alert. Redis outage: idempotency fails open (no store) or fail closed (reject order). Recommend fail closed: if Redis down, reject new orders until Redis is back.

---

## 8. Observability + Alerting

**New metrics:**
- order_submission_latency_seconds (Histogram)
- risk_rejection_total{reason}
- duplicate_order_prevented_total
- exposure_reserved_total
- broker_error_total{type}
- fill_mismatch_total
- kill_switch_triggered_total{reason}
- idempotency_hit_ratio (Gauge)
- concurrent_order_wait_seconds (Histogram)

**Alert thresholds (example):**
- order_submission_latency p99 > 5s
- risk_rejection_total rate > 10/min
- fill_mismatch_total > 0
- kill_switch_triggered_total > 0
- broker_error_total rate > 5/min

---

## 9. Flash Crash + Extreme Scenarios

| Scenario | Expected behavior | Weakness / protection |
|----------|-------------------|------------------------|
| 5 simultaneous AI signals | Global lock serializes; each passes risk and reserve; 5 orders placed in sequence. Reservation prevents exceeding max_open_positions if total would exceed. | If 5 signals same symbol/side, we might place 5 orders (no dedupe by symbol). Add per-symbol or global throttle if needed. |
| India VIX 3x | CircuitKillController.check_india_vix_and_trip( vix ) arms kill switch; new orders blocked; reduce-only allowed. | VIX feed must be real-time; if delayed, orders may still go. |
| Liquidity vacuum | Slippage/spread in fill model; volume participation cap may result in partial fills. Live: large slippage possible. | Add max slippage check (reject if fill price worse than limit by X bps). |
| 50% broker API timeout | BrokerGatewayResilient retries; half of orders may still fail after retries. Reservation released on failure. | No automatic retry of rejected orders; manual or queue retry. |
| Duplicate WebSocket fill | Same fill event processed twice → add_position twice → position doubled. | Idempotency on fill: key = (order_id, fill_id or ts+qty+price). Dedupe in FillHandler. |
| Massive slippage | Fill price far from signal; PnL loss. | Hard limit: reject fill callback if avg_price worse than order limit by > X%; or alert only. |

**Additional protection:** Fill deduplication key (order_id, fill_ts, fill_qty); max slippage bps check on fill.

---

## 10. Institutional Redesign (Citadel / Tower / Jane Street / Two Sigma)

**What they would add or change:**

- **Co-location / low latency:** Order entry path would be single-digit ms; FIX or binary protocol; kernel bypass; dedicated NIC. Python async is too slow for HFT; would be C++/Rust for hot path.
- **Central risk (pre-trade):** Every order checked by a central risk service (not in-process); risk service returns allow/reject and reserved capacity. Order entry service never holds risk state; it calls risk API.
- **Order state machine:** Explicit states (NEW, PENDING_NEW, LIVE, PARTIAL, FILLED, CANCELLED, REJECTED) with transitions; no ambiguous states; persisted to DB and event log.
- **Audit log:** Every order attempt (including rejects) written to immutable audit log (Kafka + retention); replay for compliance.
- **Kill switch hierarchy:** Multiple levels (desk kill, risk kill, exchange kill); each can override; audit who armed and when.
- **Reconciliation:** Continuous (every tick or every order), not periodic; any mismatch freezes and alerts within seconds.
- **Model risk:** Separate validation of models that produce signals; backtest and live must use same fill model and same feature pipeline; versioned artifacts.
- **Capacity and rate limits:** Per-symbol, per-venue, per-strategy limits; token bucket or sliding window.

---

## 11. Prioritized Implementation Plan

**Tier 1 — Capital protection (must do before live):**
1. Wire API and all callers to OrderEntryService only; remove any direct router/gateway call.
2. Enforce idempotency on every order (client or derived key).
3. Ensure RiskManager positions and daily_pnl are updated on every fill/close (FillHandler + reconciliation).
4. Circuit breaker + kill switch checked in OrderEntryService; admin endpoint to arm/disarm kill.
5. Fix backtest accounting (already done) and use FillModel with latency_bars.

**Tier 2 — Execution reliability:**
6. Broker retry and timeout (BrokerGatewayResilient).
7. Reconciliation job every 1–5 min; on mismatch arm kill and alert.
8. Fill deduplication (order_id + fill_ts + fill_qty) to avoid double add_position.
9. Redis/DB persistence for idempotency and order state; crash recovery test.

**Tier 3 — Performance + scaling:**
10. Per-symbol or per-venue lock instead of global lock if contention is high.
11. Kafka publish with retry and DLQ; Redis fallback for idempotency (fail closed when Redis down).
12. Order state polling fallback when WebSocket is down.

**Tier 4 — Institutional-grade:**
13. Central risk service (out-of-process) and explicit order state machine.
14. Immutable audit log for every order attempt.
15. Multi-level kill switch and capacity/rate limits per symbol and strategy.
