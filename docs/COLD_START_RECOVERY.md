# Cold Start Recovery — Architecture

## Step 1 — Architecture Review

**Understanding:** On restart, the app must rebuild runtime state from Postgres (active orders, positions), warm RiskManager and OrderLifecycle, and optionally reconcile with broker. Trading is not allowed until recovery completes. If persistence is configured, DB unavailability must fail startup. If broker is unreachable in live mode, system enters SAFE MODE (trading disabled).

**Impacted modules:**
- **API lifespan (app.py):** Orchestrate recovery; set startup_lock, recovery_complete, safe_mode; require DB when DATABASE_URL set; run recovery after RiskManager/Lifecycle exist.
- **Persistence (order_repo):** Add `list_active_orders()` (status IN NEW, ACK, PARTIAL). Position list already exists.
- **RiskManager:** Add `load_positions_for_recovery(positions)` for cold start only; no change to normal risk logic.
- **OrderLifecycle:** Add `load_for_recovery(orders)` to repopulate in-memory state from DB; no re-submit.
- **Trading ready (trading router):** Gate on startup_lock, recovery_complete, safe_mode; fix await on kill_switch.get_state().
- **Metrics:** startup_recovery_duration_seconds, startup_recovery_failures_total, startup_recovery_mismatches_total.
- **New:** Cold-start recovery orchestration (run in lifespan or dedicated module).

**Risks mitigated:**
- **Execution integrity:** We never call OrderEntryService.submit_order during recovery; only load orders into lifecycle for visibility.
- **Risk integrity:** RiskManager is warmed from persisted positions; no bypass.
- **Duplicate protection:** No re-submit; no double-count; exposure set once from DB.
- **Persistence:** When DATABASE_URL set, DB failure fails startup (no silent degraded mode).
- **Broker unreachable:** Sets safe_mode; /trading/ready returns NOT_READY; no silent correction.

## Step 2 — Design

**Data flow:**
1. Lifespan starts → set startup_lock=True, recovery_complete=False, safe_mode=False.
2. If DATABASE_URL set: create engine/repos (on failure → raise, startup fails).
3. Create gateway, risk_manager, lifecycle, idempotency, kill_switch, reservation.
4. If persistence: run cold_start_recovery (sync load active orders + positions in executor; risk_manager.load_positions_for_recovery; lifecycle.load_for_recovery; if live mode fetch broker + reconcile; on broker error set safe_mode, increment failures).
5. Create OrderEntryService with same risk_manager/lifecycle.
6. Set recovery_complete=True, startup_lock=False, safe_mode from recovery; record duration metric.
7. /trading/ready: if startup_lock or not recovery_complete → 503; if safe_mode → 503; else existing checks (order_entry, kill_switch, circuit, equity).

**Transaction boundaries:** DB reads are read-only; no write during recovery. Reconciliation is read-only (log + metric).

**Error handling:** DB required when DATABASE_URL set → fail fast. Broker unreachable → safe_mode, log, metric.

**Concurrency:** Recovery runs once during startup (single-threaded lifespan).

## Step 4 — Verification

**Unit test strategy**
- Mock OrderRepository.list_active_orders and PositionRepository.list_positions; call run_cold_start_recovery; assert risk_manager.positions and lifecycle.list_recent match loaded data.
- Mock get_broker_positions to raise; assert safe_mode True and startup_recovery_failures_total incremented.

**Manual test plan**
1. With DATABASE_URL set and DB up: start app; GET /trading/ready until 200 (after recovery); confirm orders/positions from DB visible via GET /orders and GET /positions.
2. With DATABASE_URL set and DB down: start app; expect startup to fail (exception in lifespan).
3. With DATABASE_URL set, DB up, gateway in live mode and broker unreachable: start app; expect safe_mode True; GET /trading/ready returns 503 reason "safe_mode_broker_unreachable".
4. Without DATABASE_URL: start app; recovery_complete True, no recovery run; GET /trading/ready depends on order_entry only.

**Edge cases**
- No active orders / no positions in DB: recovery loads empty lists; RiskManager and Lifecycle empty; no error.
- Broker get_positions raises (timeout, auth): safe_mode set; no silent correction.

**Failure scenarios**
- DB down with DATABASE_URL set: startup fails (raise); no degraded silent mode.
- Broker down in live mode: safe_mode True; /trading/ready 503; no trading until operator resolves and restarts or clears safe_mode (latter not implemented; safe_mode is process-local).

**Restart scenarios**
- Restart with same DB: active orders and positions reloaded; no re-submit; exposure correct.
- Duplicate protection: we never call submit_order during recovery; idempotency unchanged.
