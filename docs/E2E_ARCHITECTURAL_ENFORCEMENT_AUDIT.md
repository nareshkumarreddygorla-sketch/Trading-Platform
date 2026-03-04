# E2E Architectural Enforcement Audit

**Objective:** Eliminate illusion. Validate reality. Enforce institutional-grade autonomous integrity.

**Method:** Trace code only. No assumptions. No trust. Only traceable execution paths.

---

## PART 1 — EXECUTION PATH TRUTH VERIFICATION

### Full order path (traced)

| Step | Location | Action |
|------|----------|--------|
| 1 | `src/api/routers/orders.py` → `place_order()` | HTTP POST /orders → build Signal + OrderEntryRequest → `order_entry.submit_order(entry_request)` |
| 2 | `OrderEntryService.submit_order()` | validate(); price/quantity checks → reject + track_orders_rejected_total on fail |
| 3 | same | idem_key = request.idempotency_key or IdempotencyStore.derive_key(..., datetime.now().isoformat()) |
| 4 | same | if not await idempotency.is_available(): return REJECT IDEMPOTENCY_UNAVAILABLE (503) |
| 5 | same | existing = await idempotency.get(idem_key); if existing return stored order_id (no broker) |
| 6 | same | idempotency.set(idem_key, placeholder, None, "PENDING") NX; if not reserved, get again or reject idempotency_reserve_failed |
| 7 | same | kill_switch.is_armed(); if armed: async with _global_lock: net_pos = _net_position(); allow_reduce_only_order() or reject KILL_SWITCH |
| 8 | same | risk_manager.is_circuit_open() → reject CIRCUIT_BREAKER if True |
| 9 | same | async with _global_lock: can_place_order(); active_order_count = lifecycle.count_active(); reservation.reserve(..., active_order_count); reject RESERVATION_FAILED if not ok |
| 10 | same | await asyncio.wait_for(order_router.place_order(...), 30); on timeout/exception: reservation.release(), idempotency.update(REJECTED), return |
| 11 | same | reservation.commit(); idempotency.update(real_order_id, broker_order_id, status); lifecycle.register(order); _persist_order_with_retry(order); publish_order_event if set; track_orders_total(); return success |

**Broker invocation:** Only via `order_router.place_order()` which calls `gateway.place_order()`. No other code path calls gateway.place_order in the running app (resilience.py wraps gateway but is not used in app lifespan).

### Execution truth table

| Claim | Verified? | Evidence | Gap |
|-------|-----------|----------|-----|
| Exactly one broker call path | **YES** | Grep: only OrderEntryService.submit_order → order_router.place_order → gateway.place_order. No other caller of place_order in app flow. | — |
| All broker calls through OrderEntryService | **YES** | orders.py is only HTTP entry; it calls submit_order. No strategy/allocator/loop calls submit_order. | — |
| Idempotency fails closed when Redis unavailable | **YES** | is_available() = Redis ping; if False → return IDEMPOTENCY_UNAVAILABLE. set() returns False when client None or exception → reject idempotency_reserve_failed. | — |
| Reservation under shared lock with risk checks | **YES** | can_place_order(), lifecycle.count_active(), reservation.reserve() all inside `async with self._global_lock`. | — |
| Kill switch reduce-only uses net position under lock | **YES** | `async with self._global_lock: net_pos = self._net_position(...)` then allow_reduce_only_order(state, ..., net_pos). | — |
| Circuit breaker can auto-trip | **NO** | risk_manager.is_circuit_open() checked. open_circuit() exists. CircuitBreaker class has update_equity() → check_drawdown → trip(). **CircuitBreaker is never instantiated in app.py.** Nothing calls update_equity on a CircuitBreaker or check_drawdown_and_trip. | **CRITICAL:** Circuit is manual-only. Drawdown does not auto-trip. |
| Live mode calls real SmartAPI | **NO** | angel_one_gateway.py: if self.paper return Order(...); else return Order(...) with same structure. No HTTP call. Comment: "Live: call SmartAPI place order API" but code returns fake Order. | **CRITICAL** |
| cancel_order calls broker | **NO** | Paper: return True. Live: return False. TODO: SmartAPI cancel. | **CRITICAL** |
| get_positions calls broker | **NO** | Paper: return []. Live: return []. TODO: SmartAPI positions. | **CRITICAL** |
| Stub returning fake Order | **YES** | place_order (paper and live) both return in-memory Order with uuid4() order_id, status=PENDING. No network. | **CRITICAL** |
| Component that produces FillEvent | **NONE** | FillHandler.on_fill_event exists. No WebSocket, no poll loop, no gateway callback. Grep: no caller of on_fill_event in codebase. | **CRITICAL:** Fill path is dead. |
| FillHandler invoked in running system | **NO** | Only registered on app.state.fill_handler. Nothing invokes it. | **CRITICAL** |
| Silent bypass paths | **NONE** | No alternate path to broker found. | — |
| Race: reservation vs broker vs idempotency update | **LOW** | Lock released before broker call. After broker success: commit reservation, then idempotency.update, then lifecycle, then persist. If crash after broker success but before idempotency.update: retry with same key could get existing from Redis if another request already updated; if crash before any update, Redis has placeholder — second request would get placeholder or race. Idempotency.update overwrites placeholder with real_order_id immediately after broker (before persist). | **MEDIUM:** Crash after broker success, before idempotency.update: lost update; retry might re-call broker if key not yet updated (unlikely given single process). |
| persist_order retry | **YES** | _persist_order_with_retry: 3 attempts, delays (0.5, 1, 2)s; on final failure track_orders_persist_failed_total() then raise. | — |
| persist_fill atomicity | **YES** | Single session_scope: order_repo.update_order_status + position_repo.upsert_from_fill in one transaction. Symbol lock around _run_write_sync(_do). | — |
| OCC retry limit and failure behavior | **1 retry** | except PositionConcurrentUpdateError: logger.warning then await _run_write_sync(_do) once. Second failure propagates. No bounded retry count. | **HIGH:** One retry only; second OCC conflict raises. |

### Gaps summary (Part 1)

| Severity | Item |
|----------|------|
| **Critical** | Live broker: no SmartAPI; stub returns fake Order. cancel_order/get_positions/get_order_status same. |
| **Critical** | No fill producer: FillHandler never invoked. No WebSocket/poll. |
| **Critical** | Circuit breaker never auto-trips (CircuitBreaker not wired). |
| **High** | OCC: one retry only; second conflict raises. |
| **Medium** | Crash window after broker success before idempotency.update (theoretical in single process). |

---

## PART 2 — RECOVERY & RESTART INTEGRITY

### Startup sequence (traced)

- lifespan: startup_lock=True, recovery_complete=False, safe_mode=False.
- If DATABASE_URL: create engine, Base.metadata.create_all, repos, PersistenceService; on exception **raise** (startup fails).
- Gateway paper=True → is_live=False → get_broker_positions=None.
- run_cold_start_recovery(order_repo, position_repo, risk_manager, lifecycle, get_broker_positions=None, get_risk_snapshot=persistence_service.get_risk_snapshot_sync).
- Recovery: list_active_orders, list_positions (executor); load_positions_for_recovery(positions); get_risk_snapshot() → update_equity, daily_pnl; lifecycle.load_for_recovery(active_orders); _validate_recovery_invariants (raise RuntimeError if exposure > 1.5*equity or duplicate order_id). Reconciliation skipped (get_broker_positions None). safe_mode remains False unless live and reconcile raises.
- OrderEntryService, FillHandler constructed; recovery_complete=True, startup_lock=False.
- Periodic risk snapshot: every 60s save_risk_snapshot_sync(equity, daily_pnl).

### Recovery truth table

| Claim | Verified? | Evidence |
|-------|-----------|----------|
| startup_lock blocks trading | **YES** | /trading/ready returns 503 when startup_lock True. place_order does not check startup_lock; readiness is for probes. Orders accepted after recovery_complete. | 
| recovery_complete gates readiness | **YES** | trading/ready: if not recovery_complete return 503. |
| safe_mode blocks trading | **YES** | trading/ready: if safe_mode return 503. place_order does not check safe_mode; readiness blocks. |
| DB required when DATABASE_URL set | **YES** | try/except around persistence; on failure raise. Startup fails. |
| DB down at startup | **FAIL** | Exception in get_engine/create_all/repos → raise → app fails to start. |
| Redis down at startup | **NO FAIL** | Redis not used until first submit_order. IdempotencyStore lazy-connects. Startup succeeds; first order fails idempotency.is_available(). |
| Broker down at startup | **Paper: N/A** | is_live=False; get_broker_positions not called. Live: reconcile_positions(get_broker_positions) would run; on exception safe_mode=True, track_startup_recovery_failure(). |
| Recovery loads positions | **YES** | position_repo.list_positions() → risk_manager.load_positions_for_recovery(positions). |
| Recovery loads active orders | **YES** | order_repo.list_active_orders() → lifecycle.load_for_recovery(active_orders). |
| Recovery loads equity | **YES** | get_risk_snapshot_sync() → (equity, daily_pnl) → risk_manager.update_equity(equity), risk_manager.daily_pnl = daily_pnl. |
| Recovery loads daily_pnl | **YES** | Same as above. |
| Invariant: exposure ≤ 1.5*equity | **YES** | _validate_recovery_invariants: position_value > equity*1.5 → RuntimeError. |
| Invariant: no duplicate active order_id | **YES** | len(order_ids) != len(set(order_ids)) → RuntimeError. |
| Reconciliation auto-correcting | **NO** | reconcile_positions: log mismatches; does NOT write to DB. Log-only. |
| PENDING_SUBMIT gap risk | **YES** | If process dies after broker accepts but before persist_order: order in Redis (placeholder then updated to real_id). On restart, order not in DB; not in lifecycle (new process). Broker has order. No write-ahead marker. Reconciliation would show broker position not in DB; no auto-fix. | **HIGH** |
| Periodic risk snapshot | **YES** | Every 60s save_risk_snapshot_sync(equity, daily_pnl). |

### Crash-between states

| Crash after | Lost | Duplicate risk | Assumption |
|-------------|------|----------------|------------|
| Broker success, before lifecycle.register | Order at broker. Not in lifecycle. Not in DB. Reservation already committed. Idempotency may have been updated (real_order_id) after broker in code order — actually code does: commit, idempotency.update, lifecycle.register, persist. So crash before idempotency.update: Redis has placeholder; retry with same key could set again (NX fails) or get; crash after idempotency.update but before persist: DB has no order; lifecycle has no order; broker has order. Retry returns stored order_id. Order in broker but not in our DB/lifecycle. | Retry returns same order_id; no duplicate broker call. | Order at broker not in our state until next fill or reconciliation. |
| Broker success, before persist_order | As above if after idempotency.update. Order in broker and Redis; not in DB. | No duplicate broker. | Persist is best-effort after success; recovery does not re-persist from broker. |
| Broker success, before idempotency.update | Reservation committed. Lifecycle not updated. Redis has placeholder. Next request with same key: get() could return placeholder (another request) or set could fail. Unlikely single-process. | Theoretically two requests; one could have called broker. | Single process; one request at a time for same key. |

### Restart failure matrix

| Scenario | Result |
|----------|--------|
| DB down at startup (DATABASE_URL set) | Startup fails (raise). |
| Redis down at startup | Startup succeeds. First order reject idempotency_unavailable. |
| Broker down at startup (live) | safe_mode=True; recovery_complete still True; trading/ready 503. |
| Crash after broker, before persist | Order at broker; not in DB; in Redis (idempotency). Restart: order not in active_orders; not in lifecycle. **Lost from our state.** |
| Crash after persist_order | Order in DB; in lifecycle if register was before persist. Restart: loaded from DB. Safe. |

### What can be lost

- Order accepted by broker but not yet persisted: not in DB; not in lifecycle after restart. Broker has it; we don't until reconciliation (log-only) or fill (no fill producer).

### What can duplicate

- With same idempotency key: no duplicate broker call (return existing). With different keys (e.g. derived with now()): two orders possible from client. No server-side duplicate from single path.

### What is only assumed safe

- Single process (no second instance).
- Crash after broker before persist: we assume operational reconciliation or manual fix; no write-ahead or re-sync from broker.

---

## PART 3 — AUTONOMY VERIFICATION

### Closed loop exists?

**No.** Target: Market → Features → Strategy → Ranking → Allocation → OrderEntryService → Broker → Fill → Risk → Feedback.

| Segment | Exists? | Wired? | Evidence |
|---------|---------|--------|----------|
| Market (live) | No | — | No process ingests live ticks. Connectors stub. Kafka not fed by real feed. |
| Features (live) | No | — | FeatureStore file-based; no live pipeline. |
| Strategy → signals | Yes (code) | No | StrategyRunner exists; never called in a loop that leads to orders. |
| Ranking | Yes (code) | No | MetaAllocator, alpha research exist; not called before submit_order. |
| Allocation → OrderEntryRequest | No | — | No code builds OrderEntryRequest from allocator output and calls submit_order. |
| OrderEntryService → Broker | Yes | Yes | submit_order → router.place_order. Only from HTTP. |
| Broker → Fill | No | — | No fill delivery; stub broker. |
| Fill → Risk | Yes (code) | No producer | FillHandler would update risk; nothing calls FillHandler. |
| Feedback | No | — | No loop. |

### Repo search: who calls submit_order?

- **Only:** `src/api/routers/orders.py` in `place_order()` → `order_entry.submit_order(entry_request)`.
- No scheduled job, no asyncio loop, no event-driven caller, no strategy/allocator code path.

### Strategy output → OrderEntryRequest?

- **No.** StrategyRunner and MetaAllocator are never invoked with live data and their output is never passed to submit_order.

### MetaAllocator in live trading?

- **No.** Allocator is used in backtest/API contexts only. No live flow uses allocate() then submit_order().

### Regime classifier influences live orders?

- **No.** Regime classifier not in order path. Not called before can_place_order or submit_order.

### Drift detection disables/scales strategies live?

- **No.** Drift detectors exist; no code disables or scales strategy in the order path.

### Market ingestion: live ticks?

- **No.** No running process that connects to broker/vendor and pushes ticks to Kafka or cache.

### Kafka fed by real connectors?

- **No.** Angel One connector connect/stream_ticks/stream_bars are stubs (empty/no real read).

### FeatureStore live usage?

- **File-based read/write.** No live pipeline that continuously writes from market and is read by strategies in a loop.

### Backtest/live parity?

- **No.** Backtest uses synthetic bars when no store; one signal per bar; no parity test.

### Idempotent cycle design (bar_ts keys)?

- **N/A.** No cycle. If added, would need bar_ts (or equivalent) in idempotency key to avoid duplicate orders per bar.

### Autonomy truth table

| Question | Answer |
|----------|--------|
| Closed loop exists? | **NO** |
| Any scheduler/loop calling submit_order? | **NO** |
| Strategy → submit_order? | **NO** |
| Allocator → submit_order? | **NO** |
| Regime in live path? | **NO** |
| Drift in live path? | **NO** |
| Live market ingestion? | **NO** |
| FillHandler ever invoked? | **NO** |

### Components that exist but are unwired

- StrategyRunner, MetaAllocator, AutonomousTradingController, RegimeClassifier, drift detectors, ResearchPipeline, FillHandler, MarketDataStream (Kafka producer), Angel One connector (stub).

### Components that must be built to close loop

- Live ingestion service (broker or vendor → Kafka/cache).
- AutonomousTradingLoop (scheduler or bar-driven): snapshot/features → strategies → rank → allocator → sizing → OrderEntryRequest → submit_order.
- Broker fill delivery (WebSocket or poll) → FillEvent → FillHandler.on_fill_event.
- Optional: drift/regime gating before submit.

### New failure modes once loop is active

- Stale data → bad signals/sizing.
- Loop + broker slow/down → timeouts, reservation buildup; need backoff/health gate.
- Strategy/allocator bug → order flood; need rate/per-strategy caps.
- Drift/regime flapping; need hysteresis and audit.
- Fill storm during loop → lock contention; need stress test.

---

## PART 4 — CONCURRENCY & MULTI-INSTANCE SAFETY

### In-memory state (process-local)

| Object | Location | Process-local? |
|--------|----------|----------------|
| RiskManager.positions | risk_engine/manager.py | **YES** |
| OrderLifecycle._orders, _placed_at | execution/lifecycle.py | **YES** |
| ExposureReservation._reservations | execution/order_entry/reservation.py | **YES** |
| KillSwitch._state | execution/order_entry/kill_switch.py | **YES** |
| IdempotencyStore | Redis-backed | **NO** (shared) |

### Assumptions: 2 pods, 1000 signals/hour, 100 concurrent fills

- **Duplicate broker calls:** Two pods; two reservation pools; two lifecycles. Different idempotency keys (e.g. different timestamp in derived key) → two orders for same logical intent. Same key → Redis NX prevents double reserve but only one pod wins; the other gets get() after. So same key is safe; different keys are not. Loop with bar_ts+strategy+symbol+side would need to be same across pods or only one pod runs loop.
- **Reservation drift:** Each pod has its own reservation count. open_count from risk_manager.positions (local). So total reserved + open can exceed max_open_positions across pods.
- **Over max_open_positions:** Yes. No shared limit.
- **Symbol lock pool:** 256 locks by hash(symbol, exchange, side)%256. Collision: different symbols same bucket serialize. Same symbol always same lock. Process-local; second pod has its own pool.
- **Global lock contention:** One lock per process. Under 100 concurrent fills, FillHandler and order entry contend on _global_lock; latency and tail latency can grow.
- **Distributed lock:** **None.** No Redis or DB lock for "single placer" across pods.

### Single-instance safety statement

- **Single instance:** One order path; idempotency fail-closed; reservation and risk under one lock; kill-switch net position under lock; lifecycle FSM; persist_fill atomic with OCC (one retry). Safe for one process assuming Redis and DB available. Not proven under 100 concurrent fills (no stress test).

### Multi-instance failure scenarios

- Both pods run loop → duplicate orders (different idempotency keys).
- Both pods accept orders → open_count + reserved per pod; total positions can exceed max_open_positions.
- Restart one pod → that pod’s lifecycle and reservation empty; the other still has state; inconsistent view.

### Required distributed primitives

- Distributed mutex for order submission (one active submission cluster-wide or per symbol).
- Shared or persisted reservation/lifecycle state, or single-writer (leader) for order placement.

---

## PART 5 — RISK ENGINE INTEGRITY

| Check | Implemented? | Evidence |
|-------|----------------|----------|
| can_place_order logic | **YES** | quantity/price > 0; circuit_open; equity > 0; check_daily_loss; check_open_positions; check_position_size (effective_equity). |
| daily_loss enforcement | **YES** | limits.check_daily_loss(equity, daily_pnl) in can_place_order. |
| Circuit breaker auto-trip | **NO** | CircuitBreaker not instantiated. Only risk_manager.open_circuit() manual or via unwired circuit_and_kill. |
| Volatility-based scaling | **NO** | Not in limits or manager. |
| Per-symbol caps | **NO** | Only max_position_pct per order; no per-symbol aggregate cap. |
| Sector caps | **NO** | max_sector_concentration_pct in RiskLimits dataclass but not used in can_place_order or limits methods. |
| Consecutive-loss auto-disable | **NO** | Not in codebase. |
| VaR enforcement | **NO** | var_limit_pct in RiskLimits; no check in can_place_order. |
| Equity updates on fill | **NO** | register_pnl called from FillHandler.on_close_position (and add_or_merge/remove). Equity not updated from fill price/mark. update_equity only in recovery from snapshot. |
| Exposure multiplier | **YES** | effective_equity() = equity * _exposure_multiplier; used in can_place_order and position size check. set_exposure_multiplier caps [0.5, 1.5]. |

### Risk control completeness score

- **Implemented:** can_place_order (daily loss, open positions, position size with effective equity), exposure multiplier, circuit state (manual open/close).
- **Theoretical:** sector_concentration, var_limit_pct, max_correlation_exposure in dataclass only; not enforced in can_place_order.
- **Missing:** Volatility-based scaling, per-symbol cap, sector cap enforcement, consecutive-loss disable, VaR check, circuit auto-trip, equity update from fill/mark.

---

## PART 6 — OBSERVABILITY & AUDIT

### Prometheus metrics coverage

| Metric | Exists? | Used? |
|--------|---------|--------|
| order accepted | **YES** (orders_total) | **YES** (submit_order success) |
| order rejected | **YES** (orders_rejected_total) | **YES** (all reject paths) |
| persist failure | **YES** (orders_persist_failed_total) | **YES** (after 3 retries) |
| fill persist failure | **YES** (orders_fill_persist_failed_total) | **YES** (FillHandler._run_persist except) |
| recovery duration | **YES** (startup_recovery_duration_seconds) | **YES** (recovery.py) |
| recovery failures | **YES** (startup_recovery_failures_total) | **YES** (recovery.py) |
| reconciliation mismatches | **YES** (reconciliation_mismatches_total, startup_recovery_mismatches_total) | **YES** (recovery, reconciliation router) |

### Structured audit table

- **NO.** No AuditEventModel, no audit_repo. No table for submit/cancel/kill_switch/safe_mode/admin.

### Logging of kill_switch arm/disarm, safe_mode clear, admin

- **Partial.** kill_switch.arm/disarm log via logger.warning/info. safe_mode clear logs in orders.py. No structured audit record; no actor (user/id). Admin endpoints not protected.

### Anomaly detection

- **NO.** No anomaly metrics or alerts in code.

### Broker heartbeat

- **NO.** No periodic broker check. Safe_mode only at startup (reconciliation failure).

### Readiness endpoint logic

- **YES.** /trading/ready: startup_lock, recovery_complete, safe_mode, order_entry_service, risk_manager, kill_switch.armed, circuit_open, equity > 0. Returns 503 if any block.

### Auditability score

- **4/10.** Metrics good for orders and recovery. No audit table; no who/when for admin; no broker heartbeat.

### Operational blind spots

- Who cleared safe_mode / armed or disarmed kill_switch.
- No audit trail for rejections by reason (only counter).
- No proof of fill delivery (no fills in current system).

### Attack surface

- POST /admin/kill_switch/arm, disarm, POST /admin/safe_mode/clear, POST /orders — **no authentication.** Anyone can place orders or change safety state.

---

## PART 7 — SAAS & SECURITY

| Item | Exists? |
|------|---------|
| Authentication | **NO** |
| RBAC | **NO** |
| Tenant isolation | **NO** |
| Per-user limits | **NO** |
| Audit log for user actions | **NO** |
| Admin endpoints protected | **NO** |
| Schema tenant_id / user_id | **NO** (orders, positions have strategy_id only) |

### SaaS readiness score: **0/10**

### Security critical gaps

- All endpoints unauthenticated. Admin and order entry publicly callable.

### Monetization blockers

- No auth, no tenants, no per-user data or limits, no audit of who did what.

---

## PART 8 — STRESS & CHAOS PROOF

| Test / proof | Exists? |
|--------------|--------|
| Load tests | **NO** (no stress/ in repo) |
| 100 concurrent fill test | **NO** |
| Restart-during-fill test | **NO** |
| Redis outage test | **NO** |
| DB deadlock test | **NO** |
| Broker timeout flood test | **NO** |
| Deterministic recovery under chaos | **NOT PROVEN** |
| Idempotency under retry storms | **NOT PROVEN** |

### Stress proof score: **0/10**

### Chaos readiness score: **0/10**

### Capital deployment risk

- **High.** No evidence that 100 concurrent fills, restart during storm, or Redis/DB/broker failure are handled correctly. Execution design is sound on paper; not validated under load or chaos.

---

## PART 9 — SCORE WITH EVIDENCE (0–10)

| Domain | Score | Evidence |
|--------|-------|----------|
| **Execution Core** | **7** | Single path; idempotency fail-closed; kill switch under lock; reservation under lock; FSM; persist retry 3x; persist_fill atomic; OCC 1 retry. Broker stub; no fill producer; no stress proof. |
| **Broker Realism** | **1** | place_order/cancel_order/get_positions/get_order_status: paper returns fake/local; live returns same without HTTP. No SmartAPI. No fill delivery. |
| **Autonomy** | **0** | No loop; no caller of submit_order except HTTP. No strategy/allocator → order. No live ingestion. |
| **Risk Hardening** | **5** | can_place_order with daily loss, open positions, position size; exposure multiplier. No volatility/sector/VaR/consecutive-loss; no circuit auto-trip; sector/var in dataclass only. |
| **Concurrency Safety** | **6** | Single-instance: lock and reservation correct. Multi-instance: no distributed lock; reservation/lifecycle local. Not proven under 100 fills. |
| **Recovery Integrity** | **6** | Recovery loads positions, orders, equity, daily_pnl; invariants enforced; log-only reconciliation; PENDING_SUBMIT gap; no write-ahead. |
| **Observability** | **5** | Orders/reject/persist/recovery metrics present. No audit table; kill_switch/safe_mode not in audit; no broker heartbeat. |
| **SaaS Security** | **0** | No auth, no RBAC, no tenant_id, admin open. |
| **Institutional Maturity** | **4** | Execution design institutional-style; broker stub, no autonomy, no stress/chaos, no audit, no auth. |
| **Overall Capital Deployment Safety** | **3** | For paper/single-operator: path is strict. For real capital: no broker, no fill path, no stress proof, no audit, no auth. |

No rounding up. No optimism.

---

## PART 10 — TRANSFORMATION GAP MAP

For each domain scoring below 9: module changes, new invariants, new failure modes, testing, and criteria for 9+.

| Domain | Current | Module changes | New invariants | New failure modes | Testing | Criteria for 9+ | What blocks 10 |
|--------|---------|----------------|----------------|-------------------|---------|------------------|----------------|
| **Execution Core** | 7 | Real gateway (SmartAPI place/cancel/status/positions); fill delivery → FillHandler; optional OCC retry config (e.g. 3). | Same. | Broker latency/failure; fill reorder/duplicate. | Integration with mock broker; fill ordering; OCC under load. | Broker real; fills reach handler; stress test passed. | Chaos and multi-instance proof. |
| **Broker Realism** | 1 | angel_one_gateway.py: HTTP to SmartAPI; session refresh; new module angel_one_order_listener (WebSocket or poll) → FillEvent → fill_handler.on_fill_event. | Fill order/dedup. | Stale/duplicate fills; reconnect storms. | Mock broker + fill injection; timeout/retry tests. | Live calls broker; cancel/status/positions work; fills delivered. | Production broker SLA and monitoring. |
| **Autonomy** | 0 | autonomous_loop.py: schedule or bar trigger; snapshot → strategies → rank → allocator → sizing → OrderEntryRequest → submit_order. Wire drift/regime before submit. Ingestion service for Kafka/cache. | Idempotent cycle (bar_ts+strategy+symbol+side); no bypass. | Stale data; loop/broker contention; strategy flood. | Loop unit; integration with mock broker; rate/backoff. | Loop runs and submits only via OrderEntryService; idempotent per bar. | Full alpha pipeline and live feature store. |
| **Risk Hardening** | 5 | RiskManager/limits: volatility scaling; per-symbol/sector caps; consecutive_loss tracker; VaR check in can_place_order. Wire CircuitBreaker in lifespan and call update_equity on equity change. | Exposure and concentration limits. | Over-scaling; wrong vol/VaR. | Limit and regression tests. | All caps enforced; circuit auto-trips on drawdown. | Backtesting and live calibration. |
| **Concurrency Safety** | 6 | Redis (or DB) distributed lock for order submission; optional shared reservation or single-writer. | One active submission per scope cluster-wide. | Lock expiry; split-brain. | Two-pod tests; lock failure. | Multi-instance safe; no over-reservation. | Full distributed reservation and lifecycle. |
| **Recovery Integrity** | 6 | Write-ahead marker (e.g. PENDING_SUBMIT) or broker sync after load; optional reconciliation auto-fix with guardrails. | No order at broker missing from our state after recovery. | Sync failure; overwrite. | Recovery tests; crash-between tests. | No PENDING_SUBMIT gap or documented runbook + proof. | Full broker-state sync and reconciliation policy. |
| **Observability** | 5 | AuditEventModel + audit_repo; write on submit/cancel/kill_switch/safe_mode/drift/regime/admin. Broker heartbeat (periodic get_positions or health). | All critical actions logged. | Log failure; volume. | Audit coverage; heartbeat timeout. | Audit table populated; broker heartbeat. | Anomaly detection and escalation. |
| **SaaS Security** | 0 | auth.py: JWT validation; require_roles; protect admin and order endpoints. Schema: tenant_id/user_id; per-user limits. | All admin and order endpoints require auth; tenant isolation. | Auth bypass; token leak. | Auth and RBAC tests. | All sensitive endpoints protected; tenant_id in schema. | Billing and compliance. |
| **Institutional Maturity** | 4 | Sum of broker, autonomy, risk, concurrency, recovery, observability, security. | As above. | As above. | Stress and chaos suite. | All domains ≥ 9 with evidence. | 10/10: chaos proof, multi-instance, compliance. |
| **Capital Deployment Safety** | 3 | All of the above. | No deployment with real capital until broker real, fill path live, stress test passed, audit and auth. | As above. | Full regression + stress + chaos. | Scores ≥ 9 in execution, broker, risk, recovery, observability, security; autonomy optional for manual-only. | 10: autonomy proven, chaos proven, multi-instance. |

---

## FINAL REQUIREMENTS — COMPLIANCE

| Requirement | Status |
|-------------|--------|
| Reject any claim not proven by code path | **DONE.** All claims in this doc traced to code or marked unproven. |
| Mark illusion vs implementation | **DONE.** "Institutional-grade", "autonomous", "live trading ready" marked as illusion where not implemented. |
| Separate "component exists" from "component wired" | **DONE.** StrategyRunner, MetaAllocator, FillHandler, CircuitBreaker, etc. listed as existing but unwired. |
| Architectural drift (docs vs behavior) | **STATED.** Docs imply autonomous/live/institutional; behavior: single path, stub broker, no loop, no auth. |
| What is safe to deploy | **Single-instance, paper or manual-only live (when broker is real), trusted network, no multi-tenant.** |
| What is not safe to deploy | **Real capital without broker implementation, fill delivery, stress test, audit, and auth. Multi-instance. Public SaaS.** |

No praise. No encouragement. Only structural truth.

---

**End of E2E Architectural Enforcement Audit.**

Use this document after each major phase; compare scores; fix the weakest critical domain before advancing. When every domain scores ≥ 9 with evidence, the system is near 10/10. Until then, it is still building.
