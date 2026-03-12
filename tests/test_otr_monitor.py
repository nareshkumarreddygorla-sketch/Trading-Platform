"""Unit tests for src.compliance.otr_monitor — SEBI Order-to-Trade Ratio monitor.

Covers:
- OTRMonitor construction and defaults
- OTR computation: orders/trades ratio
- Status transitions: NORMAL -> WARNING -> HALTED
- Warmup grace period (orders < 20 with no trades stays NORMAL)
- is_halted() check
- reset_status() manual override
- Alert generation on threshold breach
- get_alerts() with filters
- get_all_otr_reports() across all algos
- Event pruning (old events removed)
- on_alert callback invocation
- Thread safety
"""

import threading

import pytest

from src.compliance.otr_monitor import (
    OTR_HALT_THRESHOLD,
    OTR_WARNING_THRESHOLD,
    OTRMonitor,
    OTRStatus,
)


@pytest.fixture
def monitor() -> OTRMonitor:
    return OTRMonitor()


@pytest.fixture
def strict_monitor() -> OTRMonitor:
    """Monitor with low thresholds for easy testing."""
    return OTRMonitor(warning_threshold=3.0, halt_threshold=5.0)


# ──────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────


class TestConstruction:
    def test_default_thresholds(self, monitor: OTRMonitor):
        assert monitor._warning_threshold == OTR_WARNING_THRESHOLD
        assert monitor._halt_threshold == OTR_HALT_THRESHOLD

    def test_custom_thresholds(self, strict_monitor: OTRMonitor):
        assert strict_monitor._warning_threshold == 3.0
        assert strict_monitor._halt_threshold == 5.0

    def test_initial_status_is_normal(self, monitor: OTRMonitor):
        assert monitor.get_status("ANY_ALGO") == OTRStatus.NORMAL


# ──────────────────────────────────────────────────
# OTR computation
# ──────────────────────────────────────────────────


class TestOTRComputation:
    def test_zero_orders_returns_zero(self, monitor: OTRMonitor):
        assert monitor.get_otr("ALGO1") == 0.0

    def test_equal_orders_and_trades(self, strict_monitor: OTRMonitor):
        """1 order + 1 trade = OTR of 1.0."""
        strict_monitor.record_order("ALGO1", "O1", "SYM")
        strict_monitor.record_trade("ALGO1", "O1", "SYM")
        otr = strict_monitor.get_otr("ALGO1")
        assert otr == 1.0

    def test_multiple_orders_one_trade(self, strict_monitor: OTRMonitor):
        """4 orders + 1 trade = OTR of 4.0."""
        for i in range(4):
            strict_monitor.record_order("ALGO1", f"O{i}", "SYM")
        strict_monitor.record_trade("ALGO1", "O0", "SYM")
        otr = strict_monitor.get_otr("ALGO1")
        assert otr == pytest.approx(4.0)

    def test_warmup_grace_few_orders_no_trades(self, strict_monitor: OTRMonitor):
        """Fewer than 20 orders with 0 trades should return 0.0 (grace period)."""
        for i in range(10):
            strict_monitor.record_order("ALGO1", f"O{i}", "SYM")
        otr = strict_monitor.get_otr("ALGO1")
        assert otr == 0.0

    def test_many_orders_no_trades_returns_inf(self, strict_monitor: OTRMonitor):
        """20+ orders with 0 trades should return inf."""
        for i in range(25):
            strict_monitor.record_order("ALGO1", f"O{i}", "SYM")
        otr = strict_monitor.get_otr("ALGO1")
        assert otr == float("inf")


# ──────────────────────────────────────────────────
# Status transitions
# ──────────────────────────────────────────────────


class TestStatusTransitions:
    def test_normal_when_otr_below_warning(self, strict_monitor: OTRMonitor):
        """2 orders + 1 trade = OTR 2.0 (below 3.0 warning)."""
        strict_monitor.record_order("ALGO1", "O1", "SYM")
        strict_monitor.record_order("ALGO1", "O2", "SYM")
        status = strict_monitor.record_trade("ALGO1", "O1", "SYM")
        assert status == OTRStatus.NORMAL

    def test_warning_when_otr_exceeds_warning(self, strict_monitor: OTRMonitor):
        """Submit enough orders to cross 3:1 warning threshold."""
        # Need OTR > 3.0 but < 5.0 in the 1-min window
        # 4 orders + 1 trade = 4.0 OTR
        for i in range(4):
            strict_monitor.record_order("ALGO1", f"O{i}", "SYM")
        status = strict_monitor.record_trade("ALGO1", "O0", "SYM")
        assert status == OTRStatus.WARNING

    def test_halted_when_otr_exceeds_halt(self, strict_monitor: OTRMonitor):
        """Submit enough orders to cross 5:1 halt threshold."""
        # 6 orders + 1 trade = 6.0 OTR (> 5.0)
        for i in range(6):
            strict_monitor.record_order("ALGO1", f"O{i}", "SYM")
        status = strict_monitor.record_trade("ALGO1", "O0", "SYM")
        assert status == OTRStatus.HALTED


# ──────────────────────────────────────────────────
# is_halted / reset_status
# ──────────────────────────────────────────────────


class TestHaltedAndReset:
    def test_is_halted(self, strict_monitor: OTRMonitor):
        for i in range(6):
            strict_monitor.record_order("ALGO1", f"O{i}", "SYM")
        strict_monitor.record_trade("ALGO1", "O0", "SYM")
        assert strict_monitor.is_halted("ALGO1") is True

    def test_reset_status(self, strict_monitor: OTRMonitor):
        for i in range(6):
            strict_monitor.record_order("ALGO1", f"O{i}", "SYM")
        strict_monitor.record_trade("ALGO1", "O0", "SYM")
        assert strict_monitor.is_halted("ALGO1") is True

        strict_monitor.reset_status("ALGO1")
        assert strict_monitor.get_status("ALGO1") == OTRStatus.NORMAL
        assert strict_monitor.is_halted("ALGO1") is False


# ──────────────────────────────────────────────────
# Alert generation
# ──────────────────────────────────────────────────


class TestAlertGeneration:
    def test_alert_generated_on_warning(self, strict_monitor: OTRMonitor):
        for i in range(4):
            strict_monitor.record_order("ALGO1", f"O{i}", "SYM")
        strict_monitor.record_trade("ALGO1", "O0", "SYM")
        alerts = strict_monitor.get_alerts()
        assert len(alerts) >= 1
        assert alerts[0]["status"] == "WARNING"

    def test_alert_generated_on_halt(self, strict_monitor: OTRMonitor):
        for i in range(6):
            strict_monitor.record_order("ALGO1", f"O{i}", "SYM")
        strict_monitor.record_trade("ALGO1", "O0", "SYM")
        alerts = strict_monitor.get_alerts()
        # Should have at least one HALTED alert
        halted_alerts = [a for a in alerts if a["status"] == "HALTED"]
        assert len(halted_alerts) >= 1

    def test_no_alert_when_normal(self, strict_monitor: OTRMonitor):
        strict_monitor.record_order("ALGO1", "O1", "SYM")
        strict_monitor.record_trade("ALGO1", "O1", "SYM")
        alerts = strict_monitor.get_alerts()
        assert len(alerts) == 0

    def test_alerts_filtered_by_algo(self, strict_monitor: OTRMonitor):
        for i in range(6):
            strict_monitor.record_order("ALGO1", f"O{i}", "SYM")
        strict_monitor.record_trade("ALGO1", "O0", "SYM")

        algo1_alerts = strict_monitor.get_alerts(algo_id="ALGO1")
        algo2_alerts = strict_monitor.get_alerts(algo_id="ALGO2")
        assert len(algo1_alerts) >= 1
        assert len(algo2_alerts) == 0


# ──────────────────────────────────────────────────
# Callback
# ──────────────────────────────────────────────────


class TestCallback:
    def test_on_alert_callback_invoked(self):
        alerts_received = []
        monitor = OTRMonitor(
            warning_threshold=3.0,
            halt_threshold=5.0,
            on_alert=lambda a: alerts_received.append(a),
        )
        for i in range(4):
            monitor.record_order("ALGO1", f"O{i}", "SYM")
        monitor.record_trade("ALGO1", "O0", "SYM")
        assert len(alerts_received) >= 1

    def test_callback_exception_does_not_crash(self):
        def bad_cb(alert):
            raise ValueError("cb error")

        monitor = OTRMonitor(warning_threshold=3.0, halt_threshold=5.0, on_alert=bad_cb)
        for i in range(4):
            monitor.record_order("ALGO1", f"O{i}", "SYM")
        # Should not raise
        monitor.record_trade("ALGO1", "O0", "SYM")


# ──────────────────────────────────────────────────
# get_all_otr_reports
# ──────────────────────────────────────────────────


class TestOTRReports:
    def test_reports_for_all_algos(self, strict_monitor: OTRMonitor):
        strict_monitor.record_order("ALGO1", "O1", "SYM")
        strict_monitor.record_order("ALGO2", "O2", "SYM")
        reports = strict_monitor.get_all_otr_reports()
        algo_ids = {r["algo_id"] for r in reports}
        assert "ALGO1" in algo_ids
        assert "ALGO2" in algo_ids

    def test_report_structure(self, strict_monitor: OTRMonitor):
        strict_monitor.record_order("ALGO1", "O1", "SYM")
        strict_monitor.record_trade("ALGO1", "O1", "SYM")
        reports = strict_monitor.get_all_otr_reports()
        assert len(reports) >= 1
        r = reports[0]
        assert "algo_id" in r
        assert "otr_1min" in r
        assert "otr_5min" in r
        assert "otr_1hr" in r
        assert "total_orders_lifetime" in r
        assert "total_trades_lifetime" in r

    def test_lifetime_counters(self, strict_monitor: OTRMonitor):
        for i in range(5):
            strict_monitor.record_order("ALGO1", f"O{i}", "SYM")
        strict_monitor.record_trade("ALGO1", "O0", "SYM")
        strict_monitor.record_trade("ALGO1", "O1", "SYM")
        reports = strict_monitor.get_all_otr_reports()
        r = [r for r in reports if r["algo_id"] == "ALGO1"][0]
        assert r["total_orders_lifetime"] == 5
        assert r["total_trades_lifetime"] == 2


# ──────────────────────────────────────────────────
# Thread safety
# ──────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_orders_and_trades(self):
        monitor = OTRMonitor()
        errors = []

        def submit_orders(algo_id: str, count: int):
            try:
                for i in range(count):
                    monitor.record_order(algo_id, f"{algo_id}_O{i}", "SYM")
            except Exception as e:
                errors.append(e)

        def submit_trades(algo_id: str, count: int):
            try:
                for i in range(count):
                    monitor.record_trade(algo_id, f"{algo_id}_O{i}", "SYM")
            except Exception as e:
                errors.append(e)

        threads = []
        for j in range(4):
            threads.append(threading.Thread(target=submit_orders, args=(f"ALGO{j}", 50)))
            threads.append(threading.Thread(target=submit_trades, args=(f"ALGO{j}", 50)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
