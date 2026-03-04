# Institutional Resilience Upgrade Plan

**Objective:** Elevate the autonomous AI trading platform from ~9.5/10 to 10/10 institutional-grade resilience through systemic hardening and capital protection—no feature creep, no UI work, no model changes.

**Status key:** ✅ Implemented | 🔄 Planned | ⏳ Deferred

---

## PHASE 1 — Eliminate Remaining Structural Risk

### 1. Symbol-level locking (replace single-threaded DB write executor)

| Field | Value |
|-------|--------|
| **Status** | ✅ Implemented |
| **Why it improves safety** | Preserves no lost-update guarantee for same-symbol position upserts while allowing different symbols to persist in parallel, improving throughput under load without weakening consistency. |
| **Impacted modules** | `src/persistence/service.py` |
| **Implementation** | Bounded lock pool (256 asyncio locks) keyed by `hash(symbol, exchange, side) % 256`. `persist_fill` acquires the lock for that key then runs in a 4-worker executor. One retry on `PositionConcurrentUpdateError`. |
| **Rollback** | Revert to single-threaded executor and remove per-symbol lock; redeploy. |
| **Risk assessment** | Low. Same-symbol serialization preserved; different symbols scale. Lock pool bounds memory. |
| **Urgency** | High (throughput + consistency). |

---

### 2. Database-level constraints

| Field | Value |
|-------|--------|
| **Status** | ✅ Implemented |
| **Why it improves safety** | Prevents invalid data at the DB layer: duplicate idempotency keys, negative quantities, invalid status values. Last line of defense against application bugs or bad data. |
| **Impacted modules** | `src/persistence/models.py` |
| **Implementation** | `OrderModel`: `UniqueConstraint("idempotency_key")`, `CheckConstraint("quantity >= 0")`, `CheckConstraint("filled_qty >= 0")`, `CheckConstraint("status IN (...)")`. `PositionModel`: `CheckConstraint("quantity >= 0")`. `order_id` was already unique. |
| **Rollback** | Migration to drop constraints; application remains compatible. |
| **Risk assessment** | Low. Existing valid data satisfies constraints; only rejects invalid writes. |
| **Urgency** | High. |

---

### 3. Optimistic concurrency control (positions)

| Field | Value |
|-------|--------|
| **Status** | ✅ Implemented |
| **Why it improves safety** | Detects concurrent updates to the same position row; raises explicit `PositionConcurrentUpdateError` instead of silent lost update. Caller can retry. |
| **Impacted modules** | `src/persistence/models.py` (version column), `src/persistence/position_repo.py`, `src/persistence/service.py` (retry once) |
| **Implementation** | `PositionModel.version` (integer, default 0). On upsert: read row, update with `WHERE id=? AND version=?` and `SET version=version+1`; if `rowcount==0` raise `PositionConcurrentUpdateError`. Delete path: conditional delete on id+version. |
| **Rollback** | Remove version from model and repo logic; add migration to drop column. |
| **Risk assessment** | Low. Retry in `persist_fill` handles transient conflicts. |
| **Urgency** | High. |

---

### 4. Lifecycle state machine

| Field | Value |
|-------|--------|
| **Status** | ✅ Implemented |
| **Why it improves safety** | Only valid status transitions are applied (NEW→ACK→PARTIAL→FILLED; NEW/ACK/PARTIAL→CANCELLED/REJECTED). Prevents out-of-order or duplicate events from corrupting order state. |
| **Impacted modules** | `src/execution/lifecycle_transitions.py` (new), `src/execution/lifecycle.py`, `src/persistence/order_repo.py` |
| **Implementation** | `ALLOWED_DB_TRANSITIONS` map; `is_allowed_transition(from, to)`. Lifecycle `update_status` and order_repo `update_order_status` check before applying; illegal transition logs and skips/returns False. |
| **Rollback** | Remove checks; revert to unconditional update. |
| **Risk assessment** | Low. Only blocks invalid transitions. |
| **Urgency** | High. |

---

## PHASE 2 — Crash Consistency & Recovery Hardening

### 5. Write-ahead recovery marker (PENDING_SUBMIT)

| Field | Value |
|-------|--------|
| **Status** | 🔄 Planned |
| **Why it improves safety** | Before calling broker, persist a PENDING_SUBMIT order row (or recovery table). On restart, any PENDING_SUBMIT without broker confirmation can be reconciled (e.g. query broker or mark failed). Reduces window where order is in Redis but not in DB. |
| **Impacted modules** | `src/execution/order_entry/service.py`, `src/persistence/order_repo.py`, `src/startup/recovery.py` |
| **Implementation** | (1) Before `place_order`, call `persist_order` with status PENDING_SUBMIT or insert into `pending_submits` table. (2) After broker success, update to NEW/ACK. (3) Recovery: list PENDING_SUBMIT, for each query broker or mark REJECTED after timeout. |
| **Rollback** | Stop writing PENDING_SUBMIT; recovery ignores it. |
| **Risk assessment** | Medium. Adds write path and recovery logic; must not block order path. |
| **Urgency** | Medium. |

---

### 6. Startup invariant validation

| Field | Value |
|-------|--------|
| **Status** | ✅ Implemented |
| **Why it improves safety** | Fails startup if recovered state is inconsistent (e.g. position exposure >> equity, duplicate active order_ids), preventing trading on corrupted state. |
| **Impacted modules** | `src/startup/recovery.py` |
| **Implementation** | `_validate_recovery_invariants(risk_manager, positions, active_orders, lifecycle)`: (1) `sum(position qty*price) <= 1.5*equity`; (2) no duplicate order_id in active_orders. Raise RuntimeError if violated. |
| **Rollback** | Remove validation call; optional env to skip. |
| **Risk assessment** | Low. Threshold (1.5×) allows some leverage; duplicate check is definitive. |
| **Urgency** | High. |

---

### 7. Broker heartbeat monitor

| Field | Value |
|-------|--------|
| **Status** | 🔄 Planned |
| **Why it improves safety** | Detects broker connectivity degradation; can auto-enter SAFE MODE and expose `broker_connectivity_status` metric so ops can react. |
| **Impacted modules** | New `src/execution/broker_heartbeat.py`, `src/api/app.py` (background task), `src/monitoring/metrics.py` |
| **Implementation** | Periodic task (e.g. every 30s) calls `gateway.get_positions()` or lightweight ping; on failure N times, set app.state.safe_mode=True and gauge broker_connectivity_status=0. |
| **Rollback** | Disable background task; remove metric. |
| **Risk assessment** | Medium. Must not block main path; avoid hammering broker. |
| **Urgency** | Medium. |

---

### 8. Reconciliation drift thresholds

| Field | Value |
|-------|--------|
| **Status** | 🔄 Planned |
| **Why it improves safety** | If position mismatch count or magnitude exceeds threshold, auto SAFE MODE; after N cycles of persistent mismatch, escalate (alert/severity). |
| **Impacted modules** | `src/startup/recovery.py`, `src/persistence/reconciliation.py`, app state |
| **Implementation** | After reconciliation, if `len(mismatches) > threshold` or max position delta > X, set safe_mode=True. Counter for consecutive mismatch cycles; if > N, set severity and alert. |
| **Rollback** | Disable auto safe_mode; keep logging. |
| **Risk assessment** | Low. Conservative: only adds safety. |
| **Urgency** | Medium. |

---

## PHASE 3 — Concurrency & Load Hardening

### 9. Stress test harness

| Field | Value |
|-------|--------|
| **Status** | 🔄 Planned |
| **Why it improves safety** | Surfaces bugs under load (100 concurrent partial fills, restart during fill storm, Redis outage, DB latency spikes) before production. |
| **Impacted modules** | New `tests/stress/` or `scripts/stress_*.py` |
| **Implementation** | Scripts: (1) Simulate 100 partial fills for same/different symbols. (2) Restart mid-fill storm and assert deterministic state. (3) Disable Redis mid-trade, assert 503 and no duplicate broker call. (4) Inject DB latency, assert timeouts and no corruption. |
| **Rollback** | N/A (test only). |
| **Risk assessment** | Low. |
| **Urgency** | Medium. |

---

### 10. Fine-grained locking (replace global lock)

| Field | Value |
|-------|--------|
| **Status** | ⏳ Deferred |
| **Why it improves safety** | Could reduce contention by separating risk-mutation lock from lifecycle read lock, improving throughput while preserving correctness. |
| **Impacted modules** | `src/execution/order_entry/service.py`, `src/execution/fill_handler/handler.py` |
| **Implementation** | Introduce separate locks (e.g. risk_lock, lifecycle_lock); ensure no deadlock (consistent lock order) and all positions reads/writes still protected. Requires careful audit. |
| **Rollback** | Revert to single global lock. |
| **Risk assessment** | High. Deadlock or missed critical section if ordering wrong. |
| **Urgency** | Low (current global lock is correct; optimize only if proven bottleneck). |

---

### 11. Timeouts everywhere

| Field | Value |
|-------|--------|
| **Status** | Partially done |
| **Why it improves safety** | Prevents hung calls from holding resources (reservation, connection) and enables fail-fast. |
| **Impacted modules** | Broker (✅ 30s), DB (run_in_executor has no timeout—add optional), reconciliation (add timeout), background tasks (periodic risk snapshot already bounded). |
| **Implementation** | Wrap `run_in_executor` with `asyncio.wait_for(..., timeout)` where appropriate; reconciliation call with timeout; document all timeouts. |
| **Rollback** | Remove timeouts; accept possible hangs. |
| **Risk assessment** | Low. |
| **Urgency** | Medium. |

---

## PHASE 4 — Capital Guardrails Beyond Strategy

### 12. Circuit breaker escalation tiers

| Field | Value |
|-------|--------|
| **Status** | 🔄 Planned |
| **Why it improves safety** | Graduated response: Tier 1 reduce exposure multiplier, Tier 2 disable new entries, Tier 3 full kill switch. Avoids binary on/off. |
| **Impacted modules** | `src/risk_engine/circuit_breaker.py`, `src/risk_engine/manager.py`, limits |
| **Implementation** | Define tiers (e.g. drawdown 2% → tier 1, 4% → tier 2, 5% → tier 3). Tier 1: set exposure_multiplier to 0.5. Tier 2: open_circuit(). Tier 3: call kill_switch.arm(). |
| **Rollback** | Revert to single open/close. |
| **Risk assessment** | Low. |
| **Urgency** | Medium. |

---

### 13. Consecutive loss auto-disable

| Field | Value |
|-------|--------|
| **Status** | 🔄 Planned |
| **Why it improves safety** | Automatically disable alpha/strategy after N consecutive losses in a rolling window to limit drawdown. |
| **Impacted modules** | Risk/strategy layer, new tracker |
| **Implementation** | Track last K trade outcomes per strategy_id; if all losses and sum exceeds threshold, set strategy disabled or reduce weight. |
| **Rollback** | Feature flag to disable. |
| **Risk assessment** | Medium. Must not disable on transient noise. |
| **Urgency** | Low. |

---

### 14. Volatility-based exposure scaling

| Field | Value |
|-------|--------|
| **Status** | 🔄 Planned |
| **Why it improves safety** | In high volatility, reduce position size to limit tail risk. |
| **Impacted modules** | `src/risk_engine/manager.py`, limits, market data |
| **Implementation** | Compute rolling volatility (e.g. realized vol); if above threshold, scale max_position_pct or exposure_multiplier down. |
| **Rollback** | Disable scaling; use fixed limits. |
| **Risk assessment** | Medium. Volatility estimate must be robust. |
| **Urgency** | Low. |

---

### 15. Max capital at risk per symbol and sector

| Field | Value |
|-------|--------|
| **Status** | 🔄 Planned |
| **Why it improves safety** | Caps exposure to a single name or sector even if total equity allows more. |
| **Impacted modules** | `src/risk_engine/limits.py`, `src/risk_engine/manager.py` |
| **Implementation** | Add max_position_per_symbol_pct, max_sector_pct; in can_place_order sum position value by symbol/sector and reject if over. |
| **Rollback** | Set limits to 100% or disable checks. |
| **Risk assessment** | Low. |
| **Urgency** | Medium. |

---

## PHASE 5 — Operational Excellence

### 16. Structured audit log

| Field | Value |
|-------|--------|
| **Status** | 🔄 Planned |
| **Why it improves safety** | Audit trail for safe mode activation, kill switch toggle, recovery completion, manual overrides supports compliance and post-incident review. |
| **Impacted modules** | New audit log module (table or append-only file), app.py, kill_switch, recovery, safe_mode/clear |
| **Implementation** | Table or file: event_type, ts, payload (JSON). Log on: safe_mode True/False, kill_switch arm/disarm, recovery complete, POST /admin/safe_mode/clear. |
| **Rollback** | Stop writing; keep code paths. |
| **Risk assessment** | Low. |
| **Urgency** | Medium. |

---

### 17. Anomaly detection metrics

| Field | Value |
|-------|--------|
| **Status** | 🔄 Planned |
| **Why it improves safety** | Sudden reject rate increase, partial fill spike, position oscillation can indicate broker or strategy issues; metrics enable alerting. |
| **Impacted modules** | `src/monitoring/metrics.py`, optional detector |
| **Implementation** | Gauges or counters: rejection_rate_rolling, partial_fills_last_5m, position_change_count. Alert when delta exceeds threshold. |
| **Rollback** | Remove metrics. |
| **Risk assessment** | Low. |
| **Urgency** | Low. |

---

### 18. Self-test endpoint

| Field | Value |
|-------|--------|
| **Status** | ✅ Implemented |
| **Why it improves safety** | Single endpoint to verify DB, Redis, broker connectivity (and optionally invariants) for ops and CI. |
| **Impacted modules** | `src/api/routers/health.py` |
| **Implementation** | GET /health/self-test: check Redis (ping with timeout), DB (list_positions_sync with timeout), broker (get_positions with timeout if gateway present). Return 503 if any critical check fails. |
| **Rollback** | Remove route. |
| **Risk assessment** | Low. |
| **Urgency** | High. |

---

### 19. Chaos testing script in CI

| Field | Value |
|-------|--------|
| **Status** | 🔄 Planned |
| **Why it improves safety** | Random failure injection (kill process, disconnect Redis, delay DB) verifies recovery and deterministic state. |
| **Impacted modules** | CI config, new chaos script |
| **Implementation** | Script that runs tests while injecting failures (e.g. kill -9 after order submit, restart, assert no duplicate order). Run in CI on schedule. |
| **Rollback** | Disable in CI. |
| **Risk assessment** | Low. |
| **Urgency** | Low. |

---

### 20. Document all invariants

| Field | Value |
|-------|--------|
| **Status** | ✅ Implemented |
| **Why it improves safety** | Explicit invariants (exposure, reservation, lifecycle, persistence, recovery, idempotency, lock) give a single source of truth for what “valid” means and where it is enforced. |
| **Impacted modules** | `docs/INVARIANTS.md` |
| **Implementation** | Document exposure, reservation, lifecycle, persistence, recovery, idempotency, and lock invariants with statement, where enforced, and rationale. |
| **Rollback** | N/A. |
| **Risk assessment** | None. |
| **Urgency** | High. |

---

## Summary

- **Implemented in this pass:** 1 (symbol-level locking), 2 (DB constraints), 3 (OCC version), 4 (lifecycle state machine), 6 (startup invariants), 18 (self-test), 20 (invariants doc).
- **Planned next:** 5 (write-ahead recovery marker), 7 (broker heartbeat), 8 (reconciliation drift thresholds), 9 (stress harness), 11 (timeouts), 12 (circuit tiers), 15 (per-symbol/sector caps), 16 (audit log).
- **Deferred:** 10 (fine-grained locking) until proven bottleneck; 13, 14 (consecutive loss, volatility scaling) as later capital guardrails; 17, 19 (anomaly metrics, chaos CI) as operational polish.

All changes preserve existing guarantees and do not simplify or weaken the architecture.
