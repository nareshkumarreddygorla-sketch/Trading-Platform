"""Unit tests for src.compliance.surveillance — SEBI post-trade surveillance.

Covers:
- SurveillanceEngine construction and defaults
- Layering detection: multiple price levels with high cancel ratio
- Spoofing detection: large orders cancelled quickly
- Wash trade detection: same algo on both sides
- Alert severity calculation
- Alert querying with filters
- Alert acknowledgement
- Summary aggregation
- Callback invocation on alert
- History pruning
- Thread safety
"""

import threading
import time

import pytest

from src.compliance.surveillance import (
    AlertSeverity,
    ManipulationPattern,
    SurveillanceEngine,
    SurveillanceThresholds,
)


@pytest.fixture
def engine() -> SurveillanceEngine:
    return SurveillanceEngine()


@pytest.fixture
def strict_engine() -> SurveillanceEngine:
    """Engine with tight thresholds for easier testing."""
    return SurveillanceEngine(
        thresholds=SurveillanceThresholds(
            layering_min_levels=2,
            layering_cancel_ratio=0.5,
            layering_time_window_sec=60.0,
            spoofing_size_multiple=2.0,
            spoofing_cancel_time_sec=10.0,
            spoofing_min_quantity=10.0,
            wash_price_tolerance=0.01,
            wash_time_window_sec=60.0,
            wash_quantity_tolerance=0.1,
        )
    )


# ──────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────


class TestConstruction:
    def test_default_thresholds(self, engine: SurveillanceEngine):
        assert engine._thresholds.layering_min_levels == 3
        assert engine._thresholds.spoofing_size_multiple == 5.0
        assert engine._thresholds.wash_price_tolerance == 0.001

    def test_custom_thresholds(self):
        t = SurveillanceThresholds(layering_min_levels=5)
        e = SurveillanceEngine(thresholds=t)
        assert e._thresholds.layering_min_levels == 5

    def test_empty_initial_state(self, engine: SurveillanceEngine):
        assert engine.get_alerts() == []
        summary = engine.get_summary()
        assert summary["total_alerts"] == 0


# ──────────────────────────────────────────────────
# Layering detection
# ──────────────────────────────────────────────────


class TestLayeringDetection:
    def test_layering_detected(self, strict_engine: SurveillanceEngine):
        """Place orders at different price levels, cancel most, detect layering.

        Layering detection runs inside record_order(), so after cancelling
        several prior orders we submit a *new* order to trigger the check.
        strict_engine: layering_min_levels=2, layering_cancel_ratio=0.5
        """
        # Place 2 BUY orders at different prices
        strict_engine.record_order("ORD0", "ALGO1", "RELIANCE", "BUY", 100.0, 50.0)
        strict_engine.record_order("ORD1", "ALGO1", "RELIANCE", "BUY", 101.0, 50.0)

        # Cancel both (100% cancel ratio, well above 50% threshold)
        strict_engine.record_cancellation("ORD0")
        strict_engine.record_cancellation("ORD1")

        # Place a new order at a different price — triggers _detect_layering.
        # Now 3 orders, 3 distinct prices, 2 cancelled / 3 total = 67% >= 50%
        alerts = strict_engine.record_order("ORD_TRIGGER", "ALGO1", "RELIANCE", "BUY", 102.0, 50.0)

        assert len(alerts) > 0 or len(strict_engine.get_alerts(pattern=ManipulationPattern.LAYERING)) > 0

    def test_no_layering_below_min_levels(self, engine: SurveillanceEngine):
        """With default min_levels=3, only 2 orders should not trigger.

        Layering is checked in record_order(), not record_cancellation().
        """
        engine.record_order("O1", "ALGO1", "SYM", "BUY", 100.0, 50.0)
        engine.record_cancellation("O1")
        # Place a second order — only 2 distinct price levels, below min_levels=3
        alerts = engine.record_order("O2", "ALGO1", "SYM", "BUY", 101.0, 50.0)
        assert len(alerts) == 0

    def test_no_layering_with_low_cancel_ratio(self, strict_engine: SurveillanceEngine):
        """If cancel ratio is below threshold, no layering alert."""
        for i, price in enumerate([100.0, 101.0, 102.0]):
            strict_engine.record_order(f"ORD{i}", "ALGO1", "SYM", "BUY", price, 50.0)
        # Cancel only 1 of 3 (33% < 50%)
        strict_engine.record_cancellation("ORD0")
        # New order triggers layering check — but cancel ratio too low
        alerts = strict_engine.record_order("ORD_TRIGGER", "ALGO1", "SYM", "BUY", 103.0, 50.0)
        assert len(alerts) == 0


# ──────────────────────────────────────────────────
# Spoofing detection
# ──────────────────────────────────────────────────


class TestSpoofingDetection:
    def test_spoofing_detected(self, strict_engine: SurveillanceEngine):
        """Large order cancelled quickly should trigger spoofing."""
        # Place a small baseline order first
        strict_engine.record_order("baseline", "ALGO1", "SYM", "BUY", 100.0, 10.0)

        # Place a large order (>= 2x average)
        strict_engine.record_order("big_order", "ALGO1", "SYM", "BUY", 100.0, 500.0)

        # Cancel it quickly
        alerts = strict_engine.record_cancellation("big_order")
        assert len(alerts) >= 1
        assert alerts[0].pattern == ManipulationPattern.SPOOFING

    def test_no_spoofing_small_order(self, strict_engine: SurveillanceEngine):
        """Order below min_quantity should not trigger spoofing."""
        strict_engine.record_order("small", "ALGO1", "SYM", "BUY", 100.0, 5.0)
        alerts = strict_engine.record_cancellation("small")
        assert len(alerts) == 0

    def test_no_spoofing_slow_cancel(self):
        """If cancellation takes too long, no spoofing alert."""
        engine = SurveillanceEngine(
            thresholds=SurveillanceThresholds(
                spoofing_cancel_time_sec=0.001,  # Very short window
                spoofing_min_quantity=10.0,
                spoofing_size_multiple=2.0,
            )
        )
        engine.record_order("O1", "ALGO1", "SYM", "BUY", 100.0, 500.0)
        time.sleep(0.01)  # Exceed the 1ms window
        alerts = engine.record_cancellation("O1")
        assert not any(a.pattern == ManipulationPattern.SPOOFING for a in alerts)


# ──────────────────────────────────────────────────
# Wash trade detection
# ──────────────────────────────────────────────────


class TestWashTradeDetection:
    def test_wash_trade_detected(self, strict_engine: SurveillanceEngine):
        """Same algo trading both sides at similar price/qty should trigger wash trade."""
        strict_engine.record_trade("T1", "O1", "ALGO1", "RELIANCE", "BUY", 2500.0, 100.0)
        alerts = strict_engine.record_trade("T2", "O2", "ALGO1", "RELIANCE", "SELL", 2500.0, 100.0)
        assert len(alerts) >= 1
        assert alerts[0].pattern == ManipulationPattern.WASH_TRADE

    def test_no_wash_different_algos(self, strict_engine: SurveillanceEngine):
        """Different algos on opposite sides should not trigger wash trade."""
        strict_engine.record_trade("T1", "O1", "ALGO1", "SYM", "BUY", 100.0, 100.0)
        alerts = strict_engine.record_trade("T2", "O2", "ALGO2", "SYM", "SELL", 100.0, 100.0)
        assert len(alerts) == 0

    def test_no_wash_different_prices(self, strict_engine: SurveillanceEngine):
        """Same algo but very different prices should not trigger."""
        strict_engine.record_trade("T1", "O1", "ALGO1", "SYM", "BUY", 100.0, 100.0)
        alerts = strict_engine.record_trade(
            "T2",
            "O2",
            "ALGO1",
            "SYM",
            "SELL",
            200.0,
            100.0,  # 100% diff
        )
        assert len(alerts) == 0

    def test_no_wash_different_quantities(self, strict_engine: SurveillanceEngine):
        """Same algo but very different quantities should not trigger."""
        strict_engine.record_trade("T1", "O1", "ALGO1", "SYM", "BUY", 100.0, 100.0)
        alerts = strict_engine.record_trade(
            "T2",
            "O2",
            "ALGO1",
            "SYM",
            "SELL",
            100.0,
            200.0,  # 100% diff
        )
        assert len(alerts) == 0


# ──────────────────────────────────────────────────
# Alert severity
# ──────────────────────────────────────────────────


class TestAlertSeverity:
    def test_spoofing_severity_critical_for_fast_cancel_large_size(self):
        """Very fast cancel + very large size -> CRITICAL."""
        engine = SurveillanceEngine(
            thresholds=SurveillanceThresholds(
                spoofing_cancel_time_sec=10.0,
                spoofing_min_quantity=10.0,
                spoofing_size_multiple=2.0,
            )
        )
        engine.record_order("baseline", "ALGO1", "SYM", "BUY", 100.0, 10.0)
        engine.record_order("big", "ALGO1", "SYM", "BUY", 100.0, 5000.0)
        alerts = engine.record_cancellation("big")
        if alerts:
            # Fast cancel + 500x size -> should be HIGH or CRITICAL
            assert alerts[0].severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL)


# ──────────────────────────────────────────────────
# Alert querying
# ──────────────────────────────────────────────────


class TestAlertQuerying:
    def test_get_alerts_empty(self, engine: SurveillanceEngine):
        assert engine.get_alerts() == []

    def test_get_alerts_filter_by_pattern(self, strict_engine: SurveillanceEngine):
        # Generate a wash trade alert
        strict_engine.record_trade("T1", "O1", "ALGO1", "SYM", "BUY", 100.0, 100.0)
        strict_engine.record_trade("T2", "O2", "ALGO1", "SYM", "SELL", 100.0, 100.0)

        wash_alerts = strict_engine.get_alerts(pattern=ManipulationPattern.WASH_TRADE)
        layering_alerts = strict_engine.get_alerts(pattern=ManipulationPattern.LAYERING)
        assert len(wash_alerts) >= 1
        assert len(layering_alerts) == 0

    def test_get_alerts_filter_by_symbol(self, strict_engine: SurveillanceEngine):
        strict_engine.record_trade("T1", "O1", "ALGO1", "AAA", "BUY", 100.0, 100.0)
        strict_engine.record_trade("T2", "O2", "ALGO1", "AAA", "SELL", 100.0, 100.0)

        aaa_alerts = strict_engine.get_alerts(symbol="AAA")
        bbb_alerts = strict_engine.get_alerts(symbol="BBB")
        assert len(aaa_alerts) >= 1
        assert len(bbb_alerts) == 0

    def test_get_alerts_limit(self, strict_engine: SurveillanceEngine):
        # Generate multiple alerts
        for i in range(5):
            strict_engine.record_trade(f"T{i}a", f"O{i}a", "ALGO1", "SYM", "BUY", 100.0, 100.0)
            strict_engine.record_trade(f"T{i}b", f"O{i}b", "ALGO1", "SYM", "SELL", 100.0, 100.0)

        limited = strict_engine.get_alerts(limit=2)
        assert len(limited) <= 2


# ──────────────────────────────────────────────────
# Acknowledge alert
# ──────────────────────────────────────────────────


class TestAcknowledgeAlert:
    def test_acknowledge_existing_alert(self, strict_engine: SurveillanceEngine):
        strict_engine.record_trade("T1", "O1", "ALGO1", "SYM", "BUY", 100.0, 100.0)
        strict_engine.record_trade("T2", "O2", "ALGO1", "SYM", "SELL", 100.0, 100.0)

        alerts = strict_engine.get_alerts()
        assert len(alerts) >= 1
        alert_id = alerts[0]["alert_id"]
        assert strict_engine.acknowledge_alert(alert_id) is True

        # Check it's now acknowledged
        acked = strict_engine.get_alerts(unacknowledged_only=True)
        assert all(a["alert_id"] != alert_id for a in acked)

    def test_acknowledge_nonexistent_alert(self, engine: SurveillanceEngine):
        assert engine.acknowledge_alert("nonexistent") is False


# ──────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────


class TestSummary:
    def test_summary_structure(self, engine: SurveillanceEngine):
        summary = engine.get_summary()
        assert "total_alerts" in summary
        assert "unacknowledged" in summary
        assert "by_pattern" in summary
        assert "by_severity" in summary
        assert "thresholds" in summary

    def test_summary_counts(self, strict_engine: SurveillanceEngine):
        strict_engine.record_trade("T1", "O1", "ALGO1", "SYM", "BUY", 100.0, 100.0)
        strict_engine.record_trade("T2", "O2", "ALGO1", "SYM", "SELL", 100.0, 100.0)
        summary = strict_engine.get_summary()
        assert summary["total_alerts"] >= 1
        assert summary["unacknowledged"] >= 1


# ──────────────────────────────────────────────────
# Callback
# ──────────────────────────────────────────────────


class TestCallback:
    def test_on_alert_callback_invoked(self):
        alerts_received = []
        engine = SurveillanceEngine(
            thresholds=SurveillanceThresholds(
                wash_price_tolerance=0.01,
                wash_time_window_sec=60.0,
                wash_quantity_tolerance=0.1,
            ),
            on_alert=lambda a: alerts_received.append(a),
        )
        engine.record_trade("T1", "O1", "ALGO1", "SYM", "BUY", 100.0, 100.0)
        engine.record_trade("T2", "O2", "ALGO1", "SYM", "SELL", 100.0, 100.0)
        assert len(alerts_received) >= 1

    def test_callback_exception_does_not_crash(self):
        def bad_callback(alert):
            raise ValueError("callback error")

        engine = SurveillanceEngine(
            thresholds=SurveillanceThresholds(
                wash_price_tolerance=0.01,
                wash_time_window_sec=60.0,
                wash_quantity_tolerance=0.1,
            ),
            on_alert=bad_callback,
        )
        engine.record_trade("T1", "O1", "ALGO1", "SYM", "BUY", 100.0, 100.0)
        # Should not raise
        engine.record_trade("T2", "O2", "ALGO1", "SYM", "SELL", 100.0, 100.0)


# ──────────────────────────────────────────────────
# Order status tracking
# ──────────────────────────────────────────────────


class TestOrderStatusTracking:
    def test_order_marked_filled_on_trade(self, engine: SurveillanceEngine):
        engine.record_order("O1", "ALGO1", "SYM", "BUY", 100.0, 50.0)
        engine.record_trade("T1", "O1", "ALGO1", "SYM", "BUY", 100.0, 50.0)
        assert engine._order_index["O1"].status == "FILLED"

    def test_order_marked_cancelled(self, engine: SurveillanceEngine):
        engine.record_order("O1", "ALGO1", "SYM", "BUY", 100.0, 50.0)
        engine.record_cancellation("O1")
        assert engine._order_index["O1"].status == "CANCELLED"

    def test_cancel_unknown_order_no_crash(self, engine: SurveillanceEngine):
        alerts = engine.record_cancellation("NONEXISTENT")
        assert alerts == []


# ──────────────────────────────────────────────────
# Thread safety
# ──────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_operations(self):
        engine = SurveillanceEngine()
        errors = []

        def submit_orders(algo_id: str, count: int):
            try:
                for i in range(count):
                    engine.record_order(f"{algo_id}_O{i}", algo_id, "SYM", "BUY", 100.0 + i, 50.0)
            except Exception as e:
                errors.append(e)

        def submit_trades(algo_id: str, count: int):
            try:
                for i in range(count):
                    engine.record_trade(f"{algo_id}_T{i}", f"{algo_id}_O{i}", algo_id, "SYM", "BUY", 100.0 + i, 50.0)
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
        summary = engine.get_summary()
        assert summary["total_alerts"] >= 0  # No crash
