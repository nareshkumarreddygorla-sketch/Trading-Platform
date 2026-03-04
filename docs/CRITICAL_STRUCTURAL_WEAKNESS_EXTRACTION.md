# Critical Structural Weakness Extraction

**Classification:** Structural truth only. Severity, production manifestation, worst-case capital impact, why architecture allows it.

---

## 1. Broker stubs or fake returns

| Severity | **Critical** |
|----------|--------------|
| Manifestation | In live mode gateway returns in-memory `Order()` with `uuid4()`; no HTTP call. Orders never reach exchange. |
| Worst-case capital impact | Zero execution; capital not at risk but strategy assumes filled positions; downstream PnL and risk state wrong. |
| Why allowed | Gateway live branch has comment "call SmartAPI" but code path identical to paper: `return Order(...)`. No integration test enforces real HTTP. |

---

## 2. Absence of fill producer

| Severity | **Critical** |
|----------|--------------|
| Manifestation | No WebSocket or polling; `FillHandler.on_fill_event` never called. Positions and lifecycle never updated from broker. |
| Worst-case capital impact | In-memory positions stale; risk limits based on stale state; over-leverage or wrong reduce-only decisions. |
| Why allowed | Fill delivery deferred; no component that subscribes to broker and calls FillHandler. |

---

## 3. Crash window after broker success but before persist

| Severity | **Critical** |
|----------|--------------|
| Manifestation | Process dies after broker accepts order, before `persist_order`. Order at broker; not in DB. On restart, order not in lifecycle; reconciliation log-only. |
| Worst-case capital impact | Ghost order at broker; we may place again (different idempotency key); double position; capital over-committed. |
| Why allowed | No write-ahead (persist SUBMITTING before broker); idempotency updated after broker so Redis has order_id but DB may not; recovery does not sync from broker. |

---

## 4. Idempotency instability (datetime-based key)

| Severity | **High** |
|----------|----------|
| Manifestation | When client omits key, `derive_key(..., datetime.now().isoformat())` used. Same logical order retried in same second can get different key → duplicate broker call. |
| Worst-case capital impact | Double order; double position; capital and slippage doubled. |
| Why allowed | No autonomous loop yet; when added, same bar must use bar_ts. Current API clients may send stable key; not enforced. |

---

## 5. Insufficient OCC retries

| Severity | **High** |
|----------|----------|
| Manifestation | `persist_fill` retries once on `PositionConcurrentUpdateError`. Second conflict raises; fill applied in-memory but not DB. |
| Worst-case capital impact | Position in risk_manager != DB; on restart position lost or wrong; PnL and limits wrong. |
| Why allowed | Single retry hardcoded; no config; no bounded backoff. |

---

## 6. Circuit breaker not wired

| Severity | **High** |
|----------|----------|
| Manifestation | `CircuitBreaker` class exists; never instantiated in app. Drawdown never auto-trips. Only manual `open_circuit()`. |
| Worst-case capital impact | Losses continue past drawdown limit; circuit intended to cap loss is inactive. |
| Why allowed | No wiring in lifespan; no call to `update_equity` on CircuitBreaker. |

---

## 7. Missing volatility / VaR / sector caps

| Severity | **High** |
|----------|----------|
| Manifestation | `max_sector_concentration_pct`, `var_limit_pct` in dataclass only; not checked in `can_place_order`. No volatility-based scaling. |
| Worst-case capital impact | Concentration in one sector; tail loss beyond VaR; no vol-adjusted size. |
| Why allowed | Limits defined but not enforced in order path. |

---

## 8. No write-ahead state

| Severity | **Critical** |
|----------|--------------|
| Manifestation | Order written to DB only after broker success. No SUBMITTING state; recovery cannot find in-flight orders. |
| Worst-case capital impact | Same as crash window: ghost orders, double submission. |
| Why allowed | Order status constraint does not include SUBMITTING; no code path persists before broker. |

---

## 9. No distributed lock

| Severity | **Critical** (for multi-instance) |
|----------|-----------------------------------|
| Manifestation | Two pods both call `submit_order`; separate reservation and lifecycle; total orders can exceed max_open_positions. |
| Worst-case capital impact | Over-leverage; breach of risk limits across cluster. |
| Why allowed | Single global asyncio lock per process; no Redis/DB lock. |

---

## 10. No shared reservation across pods

| Severity | **Critical** (for multi-instance) |
|----------|-----------------------------------|
| Manifestation | Reservation is in-memory per pod. Cluster-wide reserved + open can exceed limit. |
| Worst-case capital impact | Same as above. |
| Why allowed | No shared store for reservations. |

---

## 11. No broker heartbeat

| Severity | **High** |
|----------|----------|
| Manifestation | Broker unreachable at runtime not detected. safe_mode only at startup (reconciliation failure). |
| Worst-case capital impact | Orders sent to dead broker; timeouts and leaked reservations; no automatic pause. |
| Why allowed | No periodic health check; no safe_mode trigger on N failures. |

---

## 12. No session refresh logic

| Severity | **High** (when broker real) |
|----------|-----------------------------|
| Manifestation | JWT/session expires; subsequent calls 401; no refresh or re-login. |
| Worst-case capital impact | All orders fail; no recovery until restart. |
| Why allowed | No broker implementation yet; when added, session lifecycle not implemented. |

---

## 13. No immutable audit table

| Severity | **High** |
|----------|----------|
| Manifestation | No record of who armed kill_switch, cleared safe_mode, or placed/cancelled orders. |
| Worst-case capital impact | No accountability; compliance failure; cannot prove who did what. |
| Why allowed | No AuditEvent model; no audit_repo; no write on critical actions. |

---

## 14. No JWT / RBAC protection

| Severity | **Critical** |
|----------|--------------|
| Manifestation | All endpoints open. Anyone can POST /orders, arm/disarm kill_switch, clear safe_mode. |
| Worst-case capital impact | Malicious or mistaken admin action; unauthorized trading. |
| Why allowed | No auth layer; no role checks. |

---

## 15. Admin endpoints exposed

| Severity | **Critical** |
|----------|--------------|
| Manifestation | POST /admin/kill_switch/arm, disarm, POST /admin/safe_mode/clear unprotected. |
| Worst-case capital impact | Attacker disarms kill switch or clears safe_mode; trading resumes when it should not. |
| Why allowed | No dependency on auth or roles. |

---

## 16. No anomaly detection

| Severity | **Medium** |
|----------|------------|
| Manifestation | No alert on rejection spike, fill lag, broker latency spike. |
| Worst-case capital impact | Incidents undetected; delayed response. |
| Why allowed | No anomaly metrics or alerts. |

---

## 17. No stress tests

| Severity | **High** |
|----------|----------|
| Manifestation | 100 concurrent fills, 1000 submissions/min not tested. Unknown behavior under load. |
| Worst-case capital impact | Lock contention, OCC failures, or leaks under production load. |
| Why allowed | No stress test suite. |

---

## 18. No chaos tests

| Severity | **High** |
|----------|----------|
| Manifestation | Redis outage, DB latency, broker timeout, restart during fill not tested. |
| Worst-case capital impact | Recovery or idempotency fails in production; duplicate orders or lost state. |
| Why allowed | No chaos suite. |

---

## 19. Path that can leak capital

| Path | Severity | How |
|------|----------|-----|
| Over-reservation (multi-instance) | **Critical** | Two pods each reserve up to max; total positions exceed limit. |
| Double order (no stable idempotency) | **High** | Same logical order, different keys → two broker calls. |
| Stale positions (no fills) | **Critical** | Risk and kill-switch use stale positions; wrong reduce-only or over-size. |
| Persist failure after broker success | **High** | We return success; DB never has order; restart loses it; broker has order. |

---

## 20. Path that can duplicate broker call

| Path | Severity | How |
|------|----------|-----|
| Retry with different idempotency key | **High** | Client or loop retries without same key; NX set succeeds again; second place_order. |
| Two pods, same logical order | **Critical** | No distributed lock; both call broker. |

---

**End of extraction.** Each weakness is addressed in the transformation sections (2–9).
