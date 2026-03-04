# System Invariants — Autonomous AI Trading Platform

These invariants must hold for the system to be in a valid, capital-safe state. Violations should fail fast (startup) or be corrected/reconciled.

---

## 1. Exposure invariant

- **Statement:** Sum of (position quantity × price) across all positions must not exceed a multiple of equity (e.g. 1.5×) at startup.
- **Where enforced:** Cold start recovery (`_validate_recovery_invariants`). Startup fails if `sum(position exposure) > 1.5 * equity`.
- **Rationale:** Prevents obviously corrupted state (e.g. positions from a different account or bug) from being loaded.

---

## 2. Reservation invariant

- **Statement:** The number of reserved slots (pending orders not yet acked by broker) plus the number of active lifecycle orders (NEW, ACK, PARTIAL) plus open position count must not exceed `max_open_positions`.
- **Where enforced:** `ExposureReservation.reserve()` with `active_order_count=lifecycle.count_active()`. Order entry rejects when `open_count + reserved_count + active_order_count >= max_open_positions`.
- **Rationale:** Ensures we never exceed intended concurrency and position count.

---

## 3. Lifecycle invariant

- **Statement:** Order status may only transition along allowed edges:
  - NEW → ACK, PARTIAL, FILLED, REJECTED, CANCELLED  
  - ACK → PARTIAL, FILLED, REJECTED, CANCELLED  
  - PARTIAL → FILLED, CANCELLED  
  - FILLED, REJECTED, CANCELLED are terminal.
- **Where enforced:** `OrderLifecycle.update_status()` (in-memory) and `OrderRepository.update_order_status()` (DB). Illegal transitions are rejected/logged and not applied.
- **Rationale:** Prevents out-of-order or duplicate events from corrupting order state.

---

## 4. Persistence invariant

- **Statement:** Order and position writes are consistent: (a) no empty `order_id` inserted; (b) order status update and position upsert for a fill occur in a single transaction; (c) position updates use optimistic concurrency (version) so concurrent updates for the same row fail explicitly instead of lost update.
- **Where enforced:** `OrderRepository.create_order` (guard on empty order_id); `PersistenceService.persist_fill` (single `session_scope`; position_repo uses version column and raises `PositionConcurrentUpdateError` on conflict).
- **Rationale:** Ensures DB state is auditable and never silently overwritten.

---

## 5. Recovery invariant

- **Statement:** After cold start, active orders loaded from DB have no duplicate `order_id`; positions and equity/daily_pnl are restored from risk_snapshot when present.
- **Where enforced:** `_validate_recovery_invariants` (duplicate order_id check); `run_cold_start_recovery` (load positions, risk_snapshot, lifecycle).
- **Rationale:** Prevents duplicate orders in lifecycle and ensures risk state is restored for loss limits.

---

## 6. Idempotency invariant

- **Statement:** When Redis is available, no two requests with the same idempotency key can both reach the broker; when Redis is unavailable, order entry rejects (503).
- **Where enforced:** `OrderEntryService.submit_order`: `idempotency.is_available()` → reject if False; `set()` returns False when Redis down; duplicate key returns stored result.
- **Rationale:** Prevents duplicate broker orders and double fill.

---

## 7. Lock invariant

- **Statement:** All reads and writes of `risk_manager.positions` occur under the same asyncio lock (order-entry lock) when shared with FillHandler; kill-switch reduce-only reads net position under that lock.
- **Where enforced:** FillHandler holds `order_lock` when calling `add_or_merge_position` / `remove_position`; OrderEntryService holds `_global_lock` for risk check, reservation, and kill-switch net position read.
- **Rationale:** Prevents races between order entry and fill handler that could corrupt exposure or allow/deny reduce-only incorrectly.
