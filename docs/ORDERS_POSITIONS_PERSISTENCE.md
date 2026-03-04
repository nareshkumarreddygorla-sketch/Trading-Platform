# Persistent Orders & Positions — Architecture

## Step 1 — Architecture Review

**Task:** Replace in-memory orders/positions with Postgres-backed persistence while preserving full lifecycle integrity.

**Components impacted:**
- **New:** `src/persistence/` — DB models (Order, OrderEvent, Position), repositories, async persistence service, reconciliation helper.
- **OrderEntryService:** Receives `persist_order` callback; already invokes it after lifecycle.register. We wire a callback that persists Order (status=NEW) via new persistence layer.
- **FillHandler:** Already has `on_fill_persist(event)`; we wire a callback that updates Order, inserts OrderEvent, and upserts Position.
- **API (orders router):** GET /orders → paginated from DB; GET /orders/{id} → by order_id; GET /positions → from DB. Fallback to lifecycle/risk_manager when DB not configured.
- **RiskManager:** Unchanged. Still holds in-memory positions for can_place_order and reservation. Position updates continue to flow via risk_manager.add_position in FillHandler before we persist to DB.
- **Reconciliation:** New `reconcile_positions(gateway, position_repo)` compares broker vs DB, logs discrepancies, increments metric. No auto-correction.
- **Metrics:** New Prometheus counters: orders_total, orders_rejected_total, orders_filled_total, reconciliation_mismatches_total.

**Risks:**
- Order entry path remains single (OrderEntryService); no bypass.
- Idempotency remains in Redis; DB persistence is append/update only after broker success.
- On broker failure, OrderEntryService already releases reservation and updates idempotency to REJECTED; we do not persist a failed order as NEW (persist_order is called only after successful place_order and lifecycle.register).
- DB transaction boundaries: create_order in one transaction; update_order_status + order_event + position in one transaction for each fill.

## Step 2 — Design Approach

**Data flow:**
1. **Place order:** API → OrderEntryService.submit_order → validate → idempotency → kill → circuit → risk → reserve → router.place_order → lifecycle.register → **persist_order(order)** → idempotency.update. Persist writes Order row (status=NEW) and optionally first OrderEvent(NEW).
2. **Fill/cancel/reject:** Broker → FillHandler.on_fill_event → lifecycle.update_status + risk_manager.add_position (for fill) → **on_fill_persist(event)**. Persist: update Order (status, filled_qty, avg_price), insert OrderEvent, upsert Position (on fill only).

**Error handling:**
- persist_order and on_fill_persist are best-effort; exceptions are logged and do not fail the main pipeline (order already in lifecycle / risk already updated).
- DB sessions use explicit commit/rollback; on exception rollback and re-raise in repo; caller (callback) catches and logs.

**State consistency:**
- Order created only after broker returns and lifecycle.register succeeds.
- Position in DB updated only after risk_manager.add_position (same process); no bypass of risk.
- Idempotency: duplicate client request returns existing order_id from Redis; we never create a second Order row for the same logical order (order_id is unique in DB).

**Risk integrity:**
- RiskManager.positions and exposure reservation logic unchanged.
- Position table is a persistent record; risk checks continue to use in-memory RiskManager.positions.

## Step 4 — Verification

**How to test**
1. **DB and migrations:** Set `DATABASE_URL` (e.g. `postgresql://user:pass@localhost:5432/trading`). Run `alembic upgrade head` (or rely on app startup `Base.metadata.create_all` when using persistence). Ensure tables `orders`, `order_events`, `positions` exist.
2. **Place order:** `POST /api/v1/orders` with valid body. With persistence configured, `GET /api/v1/orders` and `GET /api/v1/orders/{order_id}` should return the order from DB. Without `DATABASE_URL`, orders still flow through OrderEntryService and are returned from in-memory lifecycle.
3. **Positions:** `GET /api/v1/positions` returns from DB when persistence is configured, else from `risk_manager.positions`. After a fill is processed via FillHandler (with `on_fill_persist`), position should appear/update in DB.
4. **Reconciliation:** `POST /api/v1/reconcile/positions` calls `reconcile_positions(gateway.get_positions, position_repo, track_reconciliation_mismatches_total)`. Check logs for "Reconciliation MISMATCH" and Prometheus `reconciliation_mismatches_total` when broker and DB differ. No automatic correction.

**Edge cases**
- Duplicate order_id: idempotency returns existing result; DB `create_order` is idempotent (skips if order_id exists).
- Broker failure: reservation released, idempotency updated to REJECTED; no Order row created (persist_order is only called after success).
- DB unavailable: persist_order and on_fill_persist log and do not fail the pipeline; lifecycle and risk remain consistent.

**Failure scenarios**
- DB down: Order entry and fills still succeed; GET /orders and GET /positions fall back to lifecycle/risk_manager when persistence_service is None.
- Reconcile with broker error: returns `in_sync=False`, mismatches include `broker_fetch_error`.
