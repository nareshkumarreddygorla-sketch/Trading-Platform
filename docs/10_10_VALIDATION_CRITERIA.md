# 10/10 Institutional Validation Criteria

A domain is marked 10 only when: **code exists**, **tests exist**, **metrics exist**, **invariant is stated**, **failure mode is modeled**, and **chaos test passes**. No rounding up.

## Execution Core
- **Code**: OrderEntryService single path; write-ahead SUBMITTING; recovery reconciles SUBMITTING.
- **Tests**: test_risk_hardening, test_fill_listener, test_broker_gateway; recovery in startup.
- **Metrics**: order_submission_latency, orders_total, orders_rejected_total, broker_latency_seconds, fill_events_total.
- **Invariant**: No broker-accepted order without durable record (write-ahead).
- **Chaos**: test_chaos_suite (duplicate fill, idempotency storm, lock expiry, cluster reservation).

## Broker Realism
- **Code**: AngelOneHttpClient (login, refresh, place, cancel, get_order_details, get_order_book, get_position); AngelOneExecutionGateway live path uses client; heartbeat + safe_mode.
- **Tests**: test_broker_gateway (timeout, session, cancel, status mapping, no fake Order in live).
- **Metrics**: broker_latency_seconds, broker_failure_total, broker_session_expired_total.
- **Invariant**: No fake Order in live mode.

## Recovery Integrity
- **Code**: SUBMITTING persisted before broker; on recovery SUBMITTING → REJECTED; list_submitting_orders.
- **Tests**: Recovery reconciliation in startup; crash-after-broker covered by write-ahead.
- **Invariant**: No broker-accepted order may exist without durable record.

## Risk Hardening
- **Code**: Sector cap, VaR, per-symbol cap, consecutive-loss in can_place_order; config-driven limits; circuit wired; equity on fill callback; volatility scaling.
- **Tests**: test_risk_hardening (drawdown trip, sector/VaR/consecutive/per-symbol, vol scaling).
- **Invariant**: No order bypasses enhanced risk checks.

## Distributed Concurrency Safety
- **Code**: RedisDistributedLock; RedisClusterReservation; OrderEntryService uses both; release on all paths + finally.
- **Tests**: test_chaos_suite (lock expiry, cluster over-reservation).
- **Invariant**: Cluster behaves as single logical execution engine.

## Observability & Audit
- **Code**: AuditEventModel; AuditRepository append_sync; audit on order_submit_success/reject, kill_switch, safe_mode_clear, broker_failure.
- **Invariant**: All critical actions traceable to actor.

## SaaS Security
- **Code**: JWT get_current_user; require_roles(["admin"]); admin endpoints protected.
- **Invariant**: Admin endpoints require admin role.

## Autonomous Loop
- **Code**: AutonomousLoop (bar-based tick, safe_mode, drift/regime gates); stable idempotency key derive_key_bar_stable.
- **Invariant**: Autonomy deterministic and idempotent (same bar_ts + strategy + symbol + side → no duplicate order).

## Readiness
- **Code**: /ready (Redis); /health/self-test (Redis, DB, broker).
- **Invariant**: 503 when critical deps down.

## Capital Deployment Safety
- **Code**: Write-ahead, idempotency, cluster reservation, risk caps, circuit breaker, kill switch, audit.
- **Invariant**: No capital leakage path; no duplicate broker call; no position corruption under concurrency.
