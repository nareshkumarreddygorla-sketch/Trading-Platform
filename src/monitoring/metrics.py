"""Prometheus metrics for latency, P&L, orders, risk."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_registry = None


def get_metrics_registry():
    global _registry
    if _registry is None:
        try:
            from prometheus_client import Counter, Histogram, Gauge, REGISTRY
            _registry = REGISTRY
            _registry.order_total = Counter("trading_orders_total", "Total orders", ["strategy_id", "status"])
            _registry.order_latency = Histogram("trading_order_latency_seconds", "Order latency")
            _registry.pnl_gauge = Gauge("trading_pnl_total", "Cumulative P&L")
            _registry.risk_circuit = Gauge("trading_risk_circuit_open", "Circuit breaker 1=open")
            # AI layer metrics
            _registry.model_confidence = Gauge("trading_ai_model_confidence", "Ensemble/model confidence", ["model_id"])
            _registry.regime_current = Gauge("trading_ai_regime", "Current regime enum value", ["regime"])
            _registry.drift_detected = Counter("trading_ai_drift_detected_total", "Concept drift events")
            _registry.retrain_total = Counter("trading_ai_retrain_total", "Retrain runs", ["model_id", "replaced"])
            # Execution layer
            _registry.order_submission_latency = Histogram("order_submission_latency_seconds", "Order entry pipeline latency")
            _registry.risk_rejection_total = Counter("risk_rejection_total", "Orders rejected by risk", ["reason"])
            _registry.duplicate_order_prevented_total = Counter("duplicate_order_prevented_total", "Idempotency hit")
            _registry.exposure_reserved_total = Counter("exposure_reserved_total", "Exposure reservations")
            _registry.broker_error_total = Counter("broker_error_total", "Broker errors", ["type"])
            _registry.broker_latency_seconds = Histogram(
                "broker_latency_seconds",
                "Broker API call latency",
                ["operation"],
                buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            )
            _registry.broker_failure_total = Counter("broker_failure_total", "Broker call failures", ["operation", "reason"])
            _registry.broker_session_expired_total = Counter("broker_session_expired_total", "Broker session expiry events")
            _registry.fill_mismatch_total = Counter("fill_mismatch_total", "Reconciliation mismatches")
            _registry.orders_total = Counter("orders_total", "Total orders submitted (accepted)")
            _registry.orders_rejected_total = Counter("orders_rejected_total", "Total orders rejected")
            _registry.orders_filled_total = Counter("orders_filled_total", "Total orders filled")
            _registry.orders_persist_failed_total = Counter("orders_persist_failed_total", "Order persistence failures after retries")
            _registry.orders_fill_persist_failed_total = Counter("orders_fill_persist_failed_total", "Fill persistence failures (order/position update)")
            _registry.reconciliation_mismatches_total = Counter("reconciliation_mismatches_total", "Position reconciliation mismatches")
            # Fill pipeline
            _registry.fill_events_total = Counter("fill_events_total", "Fill events processed", ["fill_type"])
            _registry.duplicate_fill_total = Counter("duplicate_fill_total", "Duplicate fills skipped")
            _registry.fill_latency_seconds = Histogram(
                "fill_latency_seconds",
                "Time from fill at broker to applied",
                buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
            )
            _registry.fill_reconciliation_mismatch_total = Counter("fill_reconciliation_mismatch_total", "Fill vs position reconciliation mismatches")
            _registry.startup_recovery_duration_seconds = Histogram(
                "startup_recovery_duration_seconds",
                "Cold start recovery duration in seconds",
                buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
            )
            _registry.startup_recovery_failures_total = Counter("startup_recovery_failures_total", "Cold start recovery failures (e.g. broker unreachable)")
            _registry.startup_recovery_mismatches_total = Counter("startup_recovery_mismatches_total", "Cold start reconciliation mismatches")
            _registry.kill_switch_triggered_total = Counter("kill_switch_triggered_total", "Kill switch armed", ["reason"])
            _registry.idempotency_hit_ratio = Gauge("idempotency_hit_ratio", "Idempotency hits / total requests")
            _registry.concurrent_order_wait_seconds = Histogram("concurrent_order_wait_seconds", "Wait time for global order lock")
            # Phase 14: portfolio heat, drawdown, execution quality, drift/sharpe
            _registry.portfolio_heat = Gauge("trading_portfolio_heat", "Portfolio heat (sum |position|/equity)")
            _registry.drawdown_pct = Gauge("trading_drawdown_pct", "Current drawdown %")
            _registry.slippage_ratio = Gauge("trading_slippage_ratio", "Realized/expected slippage ratio")
            _registry.rejection_rate = Gauge("trading_rejection_rate", "Order rejection rate")
            _registry.sharpe_rolling = Gauge("trading_sharpe_rolling", "Rolling Sharpe (e.g. 20d)")
            _registry.drift_psi = Gauge("trading_drift_psi", "PSI drift value")
        except ImportError:
            logger.warning("prometheus_client not installed; metrics no-op")
            _registry = object()
    return _registry


def track_order_latency(seconds: float) -> None:
    r = get_metrics_registry()
    if hasattr(r, "order_latency"):
        r.order_latency.observe(seconds)


def track_pnl(pnl: float) -> None:
    r = get_metrics_registry()
    if hasattr(r, "pnl_gauge"):
        r.pnl_gauge.set(pnl)


def track_model_confidence(model_id: str, confidence: float) -> None:
    r = get_metrics_registry()
    if hasattr(r, "model_confidence"):
        r.model_confidence.labels(model_id=model_id).set(confidence)


def track_regime(regime: str) -> None:
    r = get_metrics_registry()
    if hasattr(r, "regime_current"):
        r.regime_current.labels(regime=regime).set(1)


def track_drift() -> None:
    r = get_metrics_registry()
    if hasattr(r, "drift_detected"):
        r.drift_detected.inc()


def track_retrain(model_id: str, replaced: bool) -> None:
    r = get_metrics_registry()
    if hasattr(r, "retrain_total"):
        r.retrain_total.labels(model_id=model_id, replaced=str(replaced).lower()).inc()


def track_order_submission_latency(seconds: float) -> None:
    r = get_metrics_registry()
    if hasattr(r, "order_submission_latency"):
        r.order_submission_latency.observe(seconds)


def track_risk_rejection(reason: str) -> None:
    r = get_metrics_registry()
    if hasattr(r, "risk_rejection_total"):
        r.risk_rejection_total.labels(reason=reason).inc()


def track_duplicate_prevented() -> None:
    r = get_metrics_registry()
    if hasattr(r, "duplicate_order_prevented_total"):
        r.duplicate_order_prevented_total.inc()


def track_exposure_reserved() -> None:
    r = get_metrics_registry()
    if hasattr(r, "exposure_reserved_total"):
        r.exposure_reserved_total.inc()


def track_broker_error(error_type: str) -> None:
    r = get_metrics_registry()
    if hasattr(r, "broker_error_total"):
        r.broker_error_total.labels(type=error_type).inc()


def track_broker_latency(operation: str, seconds: float) -> None:
    r = get_metrics_registry()
    if hasattr(r, "broker_latency_seconds"):
        r.broker_latency_seconds.labels(operation=operation).observe(seconds)


def track_broker_failure(operation: str, reason: str) -> None:
    r = get_metrics_registry()
    if hasattr(r, "broker_failure_total"):
        r.broker_failure_total.labels(operation=operation, reason=reason).inc()


def track_broker_session_expired() -> None:
    r = get_metrics_registry()
    if hasattr(r, "broker_session_expired_total"):
        r.broker_session_expired_total.inc()


def track_fill_mismatch() -> None:
    r = get_metrics_registry()
    if hasattr(r, "fill_mismatch_total"):
        r.fill_mismatch_total.inc()


def track_orders_total() -> None:
    r = get_metrics_registry()
    if hasattr(r, "orders_total"):
        r.orders_total.inc()


def track_orders_rejected_total() -> None:
    r = get_metrics_registry()
    if hasattr(r, "orders_rejected_total"):
        r.orders_rejected_total.inc()


def track_orders_filled_total() -> None:
    r = get_metrics_registry()
    if hasattr(r, "orders_filled_total"):
        r.orders_filled_total.inc()


def track_orders_persist_failed_total() -> None:
    r = get_metrics_registry()
    if hasattr(r, "orders_persist_failed_total"):
        r.orders_persist_failed_total.inc()


def track_orders_fill_persist_failed_total() -> None:
    r = get_metrics_registry()
    if hasattr(r, "orders_fill_persist_failed_total"):
        r.orders_fill_persist_failed_total.inc()


def track_reconciliation_mismatches_total(count: int = 1) -> None:
    r = get_metrics_registry()
    if hasattr(r, "reconciliation_mismatches_total"):
        for _ in range(count):
            r.reconciliation_mismatches_total.inc()


def track_startup_recovery_duration(seconds: float) -> None:
    r = get_metrics_registry()
    if hasattr(r, "startup_recovery_duration_seconds"):
        r.startup_recovery_duration_seconds.observe(seconds)


def track_startup_recovery_failure() -> None:
    r = get_metrics_registry()
    if hasattr(r, "startup_recovery_failures_total"):
        r.startup_recovery_failures_total.inc()


def track_startup_recovery_mismatches(count: int = 1) -> None:
    r = get_metrics_registry()
    if hasattr(r, "startup_recovery_mismatches_total"):
        for _ in range(count):
            r.startup_recovery_mismatches_total.inc()


def track_kill_switch(reason: str) -> None:
    r = get_metrics_registry()
    if hasattr(r, "kill_switch_triggered_total"):
        r.kill_switch_triggered_total.labels(reason=reason).inc()


def track_concurrent_order_wait(seconds: float) -> None:
    r = get_metrics_registry()
    if hasattr(r, "concurrent_order_wait_seconds"):
        r.concurrent_order_wait_seconds.observe(seconds)


def track_portfolio_heat(heat: float) -> None:
    r = get_metrics_registry()
    if hasattr(r, "portfolio_heat"):
        r.portfolio_heat.set(heat)


def track_drawdown_pct(pct: float) -> None:
    r = get_metrics_registry()
    if hasattr(r, "drawdown_pct"):
        r.drawdown_pct.set(pct)


def track_slippage_ratio(ratio: float) -> None:
    r = get_metrics_registry()
    if hasattr(r, "slippage_ratio"):
        r.slippage_ratio.set(ratio)


def track_rejection_rate(rate: float) -> None:
    r = get_metrics_registry()
    if hasattr(r, "rejection_rate"):
        r.rejection_rate.set(rate)


def track_sharpe_rolling(sharpe: float) -> None:
    r = get_metrics_registry()
    if hasattr(r, "sharpe_rolling"):
        r.sharpe_rolling.set(sharpe)


def track_drift_psi(psi: float) -> None:
    r = get_metrics_registry()
    if hasattr(r, "drift_psi"):
        r.drift_psi.set(psi)


def track_fill_event(fill_type: str) -> None:
    r = get_metrics_registry()
    if hasattr(r, "fill_events_total"):
        r.fill_events_total.labels(fill_type=fill_type).inc()


def track_duplicate_fill() -> None:
    r = get_metrics_registry()
    if hasattr(r, "duplicate_fill_total"):
        r.duplicate_fill_total.inc()


def track_fill_latency(seconds: float) -> None:
    r = get_metrics_registry()
    if hasattr(r, "fill_latency_seconds"):
        r.fill_latency_seconds.observe(seconds)


def track_fill_reconciliation_mismatch() -> None:
    r = get_metrics_registry()
    if hasattr(r, "fill_reconciliation_mismatch_total"):
        r.fill_reconciliation_mismatch_total.inc()
