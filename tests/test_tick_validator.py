"""Unit tests for src.data_pipeline.tick_validator — tick-level data integrity checks.

Covers:
- Valid tick passthrough
- Negative/zero price rejection
- Price bounds vs previous close
- Volume bounds vs ADV
- Future/stale timestamp rejection
- Circuit limit breach
- Duplicate tick detection
- Thread safety (concurrent validation)
- Stats tracking & summary
- Reset behaviour
"""

import threading
from datetime import UTC, datetime, timedelta

import pytest

from src.data_pipeline.tick_validator import (
    TickRejectReason,
    TickValidator,
)


@pytest.fixture
def validator() -> TickValidator:
    """Validator with default settings."""
    return TickValidator()


@pytest.fixture
def now_utc() -> datetime:
    return datetime.now(UTC)


# ──────────────────────────────────────────────────
# Valid tick passthrough
# ──────────────────────────────────────────────────


class TestValidTick:
    def test_valid_tick_passes(self, validator: TickValidator, now_utc: datetime):
        result = validator.validate_tick("RELIANCE", 2500.0, 100, now_utc)
        assert result.is_valid is True
        assert result.reject_reasons == []
        assert result.symbol == "RELIANCE"
        assert result.price == 2500.0
        assert result.volume == 100

    def test_valid_tick_updates_stats(self, validator: TickValidator, now_utc: datetime):
        validator.validate_tick("RELIANCE", 2500.0, 100, now_utc)
        stats = validator.get_stats("RELIANCE")
        assert stats["total_ticks"] == 1
        assert stats["valid_ticks"] == 1
        assert stats["rejected_ticks"] == 0
        assert stats["last_valid_price"] == 2500.0

    def test_naive_timestamp_gets_utc(self, validator: TickValidator):
        """Naive timestamps should be treated as UTC."""
        naive_ts = datetime(2025, 1, 15, 10, 30, 0)
        result = validator.validate_tick("INFY", 1500.0, 50, naive_ts)
        # Will be stale (it's in the past), but the check for tzinfo should pass
        assert result.timestamp.tzinfo is not None


# ──────────────────────────────────────────────────
# Price validation
# ──────────────────────────────────────────────────


class TestPriceValidation:
    def test_zero_price_rejected(self, validator: TickValidator, now_utc: datetime):
        result = validator.validate_tick("TCS", 0, 100, now_utc)
        assert result.is_valid is False
        assert TickRejectReason.ZERO_PRICE in result.reject_reasons

    def test_negative_price_rejected(self, validator: TickValidator, now_utc: datetime):
        result = validator.validate_tick("TCS", -10.0, 100, now_utc)
        assert result.is_valid is False
        assert TickRejectReason.NEGATIVE_PRICE in result.reject_reasons

    def test_price_exceeds_bound_with_previous_close(self, validator: TickValidator, now_utc: datetime):
        validator.set_previous_close("TCS", 100.0)
        # price_bound_multiplier defaults to 10x, so >1000 should fail
        result = validator.validate_tick("TCS", 1100.0, 100, now_utc)
        assert result.is_valid is False
        assert TickRejectReason.PRICE_EXCEEDS_BOUND in result.reject_reasons

    def test_price_within_bound_passes(self, validator: TickValidator, now_utc: datetime):
        validator.set_previous_close("TCS", 100.0)
        # Price must also be within circuit limit (20%), so use 110 (within both)
        result = validator.validate_tick("TCS", 110.0, 100, now_utc)
        assert result.is_valid is True

    def test_price_bound_not_checked_without_previous_close(self, validator: TickValidator, now_utc: datetime):
        # No previous close set — should not trigger PRICE_EXCEEDS_BOUND
        result = validator.validate_tick("UNKNOWN", 999999.0, 100, now_utc)
        assert TickRejectReason.PRICE_EXCEEDS_BOUND not in result.reject_reasons


# ──────────────────────────────────────────────────
# Volume validation
# ──────────────────────────────────────────────────


class TestVolumeValidation:
    def test_negative_volume_rejected(self, validator: TickValidator, now_utc: datetime):
        result = validator.validate_tick("HDFC", 1000.0, -50, now_utc)
        assert result.is_valid is False
        assert TickRejectReason.NEGATIVE_VOLUME in result.reject_reasons

    def test_volume_exceeds_adv_bound(self, validator: TickValidator, now_utc: datetime):
        validator.set_average_daily_volume("HDFC", 10_000)
        # volume_bound_multiplier defaults to 100x → 1_000_000
        result = validator.validate_tick("HDFC", 1000.0, 1_100_000, now_utc)
        assert result.is_valid is False
        assert TickRejectReason.VOLUME_EXCEEDS_BOUND in result.reject_reasons

    def test_volume_within_adv_bound_passes(self, validator: TickValidator, now_utc: datetime):
        validator.set_average_daily_volume("HDFC", 10_000)
        result = validator.validate_tick("HDFC", 1000.0, 500_000, now_utc)
        assert TickRejectReason.VOLUME_EXCEEDS_BOUND not in result.reject_reasons

    def test_zero_volume_passes(self, validator: TickValidator, now_utc: datetime):
        """Zero volume is valid (some exchanges report this for reference prices)."""
        result = validator.validate_tick("HDFC", 1000.0, 0, now_utc)
        assert TickRejectReason.NEGATIVE_VOLUME not in result.reject_reasons


# ──────────────────────────────────────────────────
# Timestamp validation
# ──────────────────────────────────────────────────


class TestTimestampValidation:
    def test_future_timestamp_rejected(self, validator: TickValidator):
        future_ts = datetime.now(UTC) + timedelta(seconds=10)
        result = validator.validate_tick("INFY", 1500.0, 100, future_ts)
        assert result.is_valid is False
        assert TickRejectReason.FUTURE_TIMESTAMP in result.reject_reasons

    def test_slight_future_within_clock_skew_passes(self, validator: TickValidator):
        """Timestamps within 2 seconds into the future are allowed for clock skew."""
        near_future = datetime.now(UTC) + timedelta(seconds=1)
        result = validator.validate_tick("INFY", 1500.0, 100, near_future)
        assert TickRejectReason.FUTURE_TIMESTAMP not in result.reject_reasons

    def test_stale_timestamp_rejected(self, validator: TickValidator):
        stale_ts = datetime.now(UTC) - timedelta(seconds=600)
        result = validator.validate_tick("INFY", 1500.0, 100, stale_ts)
        assert result.is_valid is False
        assert TickRejectReason.STALE_TIMESTAMP in result.reject_reasons

    def test_fresh_timestamp_passes(self, validator: TickValidator):
        fresh_ts = datetime.now(UTC) - timedelta(seconds=10)
        result = validator.validate_tick("INFY", 1500.0, 100, fresh_ts)
        assert TickRejectReason.STALE_TIMESTAMP not in result.reject_reasons

    def test_custom_stale_threshold(self):
        validator = TickValidator(stale_seconds=30.0)
        stale_ts = datetime.now(UTC) - timedelta(seconds=60)
        result = validator.validate_tick("INFY", 1500.0, 100, stale_ts)
        assert TickRejectReason.STALE_TIMESTAMP in result.reject_reasons


# ──────────────────────────────────────────────────
# Circuit limit breach
# ──────────────────────────────────────────────────


class TestCircuitLimitBreach:
    def test_circuit_limit_breach_upper(self, validator: TickValidator, now_utc: datetime):
        validator.set_previous_close("SBIN", 500.0)
        # Default circuit limit is 20%. 500 * 1.20 = 600. Price 610 exceeds.
        result = validator.validate_tick("SBIN", 610.0, 100, now_utc)
        assert TickRejectReason.CIRCUIT_LIMIT_BREACH in result.reject_reasons

    def test_circuit_limit_breach_lower(self, validator: TickValidator, now_utc: datetime):
        validator.set_previous_close("SBIN", 500.0)
        # 500 * 0.80 = 400. Price 390 breaches lower circuit.
        result = validator.validate_tick("SBIN", 390.0, 100, now_utc)
        assert TickRejectReason.CIRCUIT_LIMIT_BREACH in result.reject_reasons

    def test_within_circuit_limit_passes(self, validator: TickValidator, now_utc: datetime):
        validator.set_previous_close("SBIN", 500.0)
        result = validator.validate_tick("SBIN", 550.0, 100, now_utc)
        assert TickRejectReason.CIRCUIT_LIMIT_BREACH not in result.reject_reasons

    def test_custom_circuit_limit(self, now_utc: datetime):
        validator = TickValidator(circuit_limit_pct=5.0)
        validator.set_previous_close("SBIN", 500.0)
        # 5% limit: 500 * 1.05 = 525. Price 530 should breach.
        result = validator.validate_tick("SBIN", 530.0, 100, now_utc)
        assert TickRejectReason.CIRCUIT_LIMIT_BREACH in result.reject_reasons


# ──────────────────────────────────────────────────
# Duplicate tick detection
# ──────────────────────────────────────────────────


class TestDuplicateTick:
    def test_duplicate_tick_rejected(self, validator: TickValidator, now_utc: datetime):
        validator.validate_tick("WIPRO", 450.0, 200, now_utc)
        result = validator.validate_tick("WIPRO", 450.0, 200, now_utc)
        assert TickRejectReason.DUPLICATE_TICK in result.reject_reasons

    def test_same_timestamp_different_price_passes(self, validator: TickValidator, now_utc: datetime):
        validator.validate_tick("WIPRO", 450.0, 200, now_utc)
        result = validator.validate_tick("WIPRO", 451.0, 200, now_utc)
        assert TickRejectReason.DUPLICATE_TICK not in result.reject_reasons

    def test_same_price_different_timestamp_passes(self, validator: TickValidator):
        t1 = datetime.now(UTC)
        t2 = t1 + timedelta(seconds=1)
        validator.validate_tick("WIPRO", 450.0, 200, t1)
        result = validator.validate_tick("WIPRO", 450.0, 200, t2)
        assert TickRejectReason.DUPLICATE_TICK not in result.reject_reasons


# ──────────────────────────────────────────────────
# Multiple rejection reasons
# ──────────────────────────────────────────────────


class TestMultipleRejections:
    def test_multiple_reasons_collected(self, validator: TickValidator):
        """A tick can have multiple rejection reasons simultaneously."""
        validator.set_previous_close("BAD", 100.0)
        stale_ts = datetime.now(UTC) - timedelta(seconds=600)
        # Negative price + stale timestamp
        result = validator.validate_tick("BAD", -5.0, 100, stale_ts)
        assert result.is_valid is False
        assert TickRejectReason.NEGATIVE_PRICE in result.reject_reasons
        assert TickRejectReason.STALE_TIMESTAMP in result.reject_reasons


# ──────────────────────────────────────────────────
# Statistics & summary
# ──────────────────────────────────────────────────


class TestStats:
    def test_get_stats_unknown_symbol(self, validator: TickValidator):
        stats = validator.get_stats("NONEXISTENT")
        assert stats["symbol"] == "NONEXISTENT"
        assert stats["total_ticks"] == 0

    def test_valid_pct_calculation(self, validator: TickValidator, now_utc: datetime):
        validator.validate_tick("A", 100.0, 10, now_utc)
        validator.validate_tick("A", 0, 10, now_utc)  # rejected (zero price)
        stats = validator.get_stats("A")
        assert stats["total_ticks"] == 2
        assert stats["valid_ticks"] == 1
        assert stats["rejected_ticks"] == 1
        assert stats["valid_pct"] == 50.0

    def test_get_all_stats(self, validator: TickValidator, now_utc: datetime):
        validator.validate_tick("X", 100.0, 10, now_utc)
        validator.validate_tick("Y", 200.0, 20, now_utc)
        all_stats = validator.get_all_stats()
        assert "X" in all_stats
        assert "Y" in all_stats

    def test_summary_aggregation(self, validator: TickValidator, now_utc: datetime):
        validator.validate_tick("X", 100.0, 10, now_utc)
        validator.validate_tick("Y", 0, 10, now_utc)  # rejected
        summary = validator.get_summary()
        assert summary["total_symbols"] == 2
        assert summary["total_ticks"] == 2
        assert summary["valid_ticks"] == 1
        assert summary["rejected_ticks"] == 1

    def test_reject_counts_tracked(self, validator: TickValidator, now_utc: datetime):
        validator.validate_tick("X", 0, 10, now_utc)
        validator.validate_tick("X", -1.0, 10, now_utc)
        stats = validator.get_stats("X")
        assert stats["reject_counts"].get("ZERO_PRICE", 0) >= 1
        assert stats["reject_counts"].get("NEGATIVE_PRICE", 0) >= 1


# ──────────────────────────────────────────────────
# Reset
# ──────────────────────────────────────────────────


class TestReset:
    def test_reset_single_symbol(self, validator: TickValidator, now_utc: datetime):
        validator.set_previous_close("RST", 100.0)
        validator.set_average_daily_volume("RST", 50_000)
        validator.validate_tick("RST", 105.0, 100, now_utc)
        validator.reset_stats("RST")
        stats = validator.get_stats("RST")
        assert stats["total_ticks"] == 0
        # Previous close and ADV should be preserved
        assert stats["previous_close"] == 100.0
        assert stats["average_daily_volume"] == 50_000

    def test_reset_all(self, validator: TickValidator, now_utc: datetime):
        validator.validate_tick("A", 100.0, 10, now_utc)
        validator.validate_tick("B", 200.0, 20, now_utc)
        validator.reset_stats()
        assert validator.get_all_stats() == {}


# ──────────────────────────────────────────────────
# Thread safety
# ──────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_validation(self, validator: TickValidator):
        """Multiple threads validating concurrently should not crash or corrupt state."""
        errors = []

        def validate_many(symbol: str, count: int):
            try:
                for i in range(count):
                    ts = datetime.now(UTC) - timedelta(seconds=i % 10)
                    validator.validate_tick(symbol, 100.0 + i, 100, ts)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=validate_many, args=(f"SYM{j}", 100)) for j in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        summary = validator.get_summary()
        assert summary["total_ticks"] == 500
        assert summary["total_symbols"] == 5


# ──────────────────────────────────────────────────
# to_dict serialization
# ──────────────────────────────────────────────────


class TestSerialization:
    def test_to_dict(self, validator: TickValidator, now_utc: datetime):
        result = validator.validate_tick("SER", 100.0, 10, now_utc)
        d = result.to_dict()
        assert d["is_valid"] is True
        assert d["symbol"] == "SER"
        assert d["price"] == 100.0
        assert isinstance(d["timestamp"], str)
        assert d["reject_reasons"] == []
