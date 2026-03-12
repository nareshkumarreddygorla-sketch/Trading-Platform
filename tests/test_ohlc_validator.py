"""Unit tests for src.data_pipeline.ohlc_validator — OHLCV bar data integrity checks.

Covers:
- Valid bar passthrough
- OHLC consistency checks (high/low vs open/close)
- Negative/zero price rejection
- Negative volume rejection
- Gap detection (intraday and overnight)
- Stale bar detection
- Missing bar detection
- Extreme candle filtering (shadow/body ratio)
- Stats tracking & summary
- Reset behaviour
- Thread safety (concurrent validation)
"""

import threading
from datetime import UTC, datetime, timedelta

import pytest

from src.data_pipeline.ohlc_validator import (
    OHLCRejectReason,
    OHLCValidator,
    OHLCWarning,
)


@pytest.fixture
def validator() -> OHLCValidator:
    """Validator with default settings."""
    return OHLCValidator()


@pytest.fixture
def now_utc() -> datetime:
    return datetime.now(UTC)


# ──────────────────────────────────────────────────
# Valid bar passthrough
# ──────────────────────────────────────────────────


class TestValidBar:
    def test_valid_bar_passes(self, validator: OHLCValidator, now_utc: datetime):
        result = validator.validate_bar(
            "RELIANCE", open_=2500.0, high=2520.0, low=2490.0, close=2510.0, volume=10000, timestamp=now_utc
        )
        assert result.is_valid is True
        assert result.reject_reasons == []
        assert result.symbol == "RELIANCE"

    def test_valid_bar_updates_stats(self, validator: OHLCValidator, now_utc: datetime):
        validator.validate_bar("RELIANCE", 2500.0, 2520.0, 2490.0, 2510.0, 10000, now_utc)
        stats = validator.get_stats("RELIANCE")
        assert stats["total_bars"] == 1
        assert stats["valid_bars"] == 1
        assert stats["rejected_bars"] == 0
        assert stats["last_close"] == 2510.0

    def test_doji_candle_valid(self, validator: OHLCValidator, now_utc: datetime):
        """A doji (open==close) is valid if OHLC relationships hold."""
        result = validator.validate_bar("DOJI", 100.0, 105.0, 95.0, 100.0, 500, now_utc)
        assert result.is_valid is True

    def test_naive_timestamp_gets_utc(self, validator: OHLCValidator):
        """Naive timestamps should be treated as UTC."""
        naive_ts = datetime(2025, 1, 15, 10, 30, 0)
        result = validator.validate_bar("INFY", 1500.0, 1510.0, 1490.0, 1505.0, 1000, naive_ts)
        assert result.timestamp.tzinfo is not None


# ──────────────────────────────────────────────────
# OHLC consistency
# ──────────────────────────────────────────────────


class TestOHLCConsistency:
    def test_high_less_than_low_rejected(self, validator: OHLCValidator, now_utc: datetime):
        result = validator.validate_bar("BAD", 100.0, 90.0, 95.0, 98.0, 500, now_utc)
        assert result.is_valid is False
        assert OHLCRejectReason.HIGH_LESS_THAN_LOW in result.reject_reasons

    def test_high_less_than_open_rejected(self, validator: OHLCValidator, now_utc: datetime):
        result = validator.validate_bar("BAD", 105.0, 100.0, 95.0, 98.0, 500, now_utc)
        assert result.is_valid is False
        assert OHLCRejectReason.HIGH_LESS_THAN_OPEN in result.reject_reasons

    def test_high_less_than_close_rejected(self, validator: OHLCValidator, now_utc: datetime):
        result = validator.validate_bar("BAD", 100.0, 100.0, 95.0, 105.0, 500, now_utc)
        assert result.is_valid is False
        assert OHLCRejectReason.HIGH_LESS_THAN_CLOSE in result.reject_reasons

    def test_low_greater_than_open_rejected(self, validator: OHLCValidator, now_utc: datetime):
        result = validator.validate_bar("BAD", 95.0, 110.0, 100.0, 105.0, 500, now_utc)
        assert result.is_valid is False
        assert OHLCRejectReason.LOW_GREATER_THAN_OPEN in result.reject_reasons

    def test_low_greater_than_close_rejected(self, validator: OHLCValidator, now_utc: datetime):
        result = validator.validate_bar("BAD", 100.0, 110.0, 98.0, 95.0, 500, now_utc)
        assert result.is_valid is False
        assert OHLCRejectReason.LOW_GREATER_THAN_CLOSE in result.reject_reasons

    def test_flat_bar_valid(self, validator: OHLCValidator, now_utc: datetime):
        """All OHLC equal (flat bar) is valid."""
        result = validator.validate_bar("FLAT", 100.0, 100.0, 100.0, 100.0, 500, now_utc)
        assert result.is_valid is True


# ──────────────────────────────────────────────────
# Price and volume checks
# ──────────────────────────────────────────────────


class TestPriceVolumeChecks:
    def test_zero_price_rejected(self, validator: OHLCValidator, now_utc: datetime):
        result = validator.validate_bar("ZERO", 0, 100.0, 0, 0, 500, now_utc)
        assert result.is_valid is False
        assert OHLCRejectReason.ZERO_PRICE in result.reject_reasons

    def test_negative_price_rejected(self, validator: OHLCValidator, now_utc: datetime):
        result = validator.validate_bar("NEG", -10.0, 100.0, -10.0, -5.0, 500, now_utc)
        assert result.is_valid is False
        assert OHLCRejectReason.NEGATIVE_PRICE in result.reject_reasons

    def test_negative_volume_rejected(self, validator: OHLCValidator, now_utc: datetime):
        result = validator.validate_bar("VOL", 100.0, 110.0, 95.0, 105.0, -500, now_utc)
        assert result.is_valid is False
        assert OHLCRejectReason.NEGATIVE_VOLUME in result.reject_reasons

    def test_zero_volume_valid(self, validator: OHLCValidator, now_utc: datetime):
        """Zero volume is valid (pre-market or no trades)."""
        result = validator.validate_bar("ZVOL", 100.0, 105.0, 98.0, 103.0, 0, now_utc)
        assert result.is_valid is True


# ──────────────────────────────────────────────────
# Gap detection
# ──────────────────────────────────────────────────


class TestGapDetection:
    def test_intraday_gap_detected(self, validator: OHLCValidator, now_utc: datetime):
        # First bar closes at 100
        validator.validate_bar("GAP", 98.0, 102.0, 97.0, 100.0, 1000, now_utc - timedelta(minutes=1))
        # Second bar opens at 110 (10% gap, > 5% default threshold)
        result = validator.validate_bar("GAP", 110.0, 115.0, 108.0, 112.0, 1000, now_utc)
        # Gap is a warning, not a rejection
        assert result.is_valid is True
        assert any(OHLCWarning.GAP_DETECTED.value in w for w in result.warnings)

    def test_small_gap_not_flagged(self, validator: OHLCValidator, now_utc: datetime):
        validator.validate_bar("NOGAP", 98.0, 102.0, 97.0, 100.0, 1000, now_utc - timedelta(minutes=1))
        # 2% gap (below 5% threshold)
        result = validator.validate_bar("NOGAP", 102.0, 105.0, 101.0, 104.0, 1000, now_utc)
        assert not any(OHLCWarning.GAP_DETECTED.value in w for w in result.warnings)

    def test_overnight_gap_not_flagged(self, validator: OHLCValidator):
        """Overnight gaps should not be flagged for intraday intervals."""
        yesterday = datetime(2025, 3, 10, 15, 29, 0, tzinfo=UTC)
        today = datetime(2025, 3, 11, 9, 15, 0, tzinfo=UTC)
        validator.validate_bar("OVER", 98.0, 102.0, 97.0, 100.0, 1000, yesterday)
        # 15% gap (normally flagged), but crosses date boundary
        result = validator.validate_bar("OVER", 115.0, 120.0, 113.0, 118.0, 1000, today)
        assert not any(OHLCWarning.GAP_DETECTED.value in w for w in result.warnings)


# ──────────────────────────────────────────────────
# Stale bar detection
# ──────────────────────────────────────────────────


class TestStaleBar:
    def test_stale_bar_rejected(self, validator: OHLCValidator):
        # Default stale threshold is 120 seconds
        stale_ts = datetime.now(UTC) - timedelta(seconds=300)
        result = validator.validate_bar("STALE", 100.0, 105.0, 98.0, 103.0, 500, stale_ts)
        assert result.is_valid is False
        assert OHLCRejectReason.STALE_BAR in result.reject_reasons

    def test_fresh_bar_passes(self, validator: OHLCValidator):
        fresh_ts = datetime.now(UTC) - timedelta(seconds=10)
        result = validator.validate_bar("FRESH", 100.0, 105.0, 98.0, 103.0, 500, fresh_ts)
        assert OHLCRejectReason.STALE_BAR not in result.reject_reasons

    def test_custom_stale_threshold(self):
        validator = OHLCValidator(stale_seconds=30.0)
        stale_ts = datetime.now(UTC) - timedelta(seconds=60)
        result = validator.validate_bar("STALE", 100.0, 105.0, 98.0, 103.0, 500, stale_ts)
        assert OHLCRejectReason.STALE_BAR in result.reject_reasons


# ──────────────────────────────────────────────────
# Missing bar detection
# ──────────────────────────────────────────────────


class TestMissingBars:
    def test_missing_bars_flagged(self, validator: OHLCValidator, now_utc: datetime):
        # First bar at t
        t1 = now_utc - timedelta(minutes=10)
        validator.validate_bar("MISS", 100.0, 105.0, 98.0, 103.0, 500, t1, interval="1m")
        # Next bar at t+5min (should have had 4 bars between them)
        t2 = now_utc - timedelta(minutes=5)
        result = validator.validate_bar("MISS", 103.0, 108.0, 101.0, 106.0, 500, t2, interval="1m")
        # Gap of 300s with 60s intervals → 4 missing bars
        assert result.missing_bar_count > 0
        assert any(OHLCWarning.MISSING_BARS.value in w for w in result.warnings)

    def test_consecutive_bars_no_missing(self, validator: OHLCValidator, now_utc: datetime):
        t1 = now_utc - timedelta(minutes=2)
        t2 = now_utc - timedelta(minutes=1)
        validator.validate_bar("SEQ", 100.0, 105.0, 98.0, 103.0, 500, t1, interval="1m")
        result = validator.validate_bar("SEQ", 103.0, 108.0, 101.0, 106.0, 500, t2, interval="1m")
        assert result.missing_bar_count == 0


# ──────────────────────────────────────────────────
# Extreme candle filtering
# ──────────────────────────────────────────────────


class TestExtremeCandleFilter:
    def test_extreme_shadow_rejected(self, validator: OHLCValidator, now_utc: datetime):
        """Very long shadows relative to body should be flagged as extreme candles."""
        # Body: |close - open| = |100 - 101| = 1
        # Shadows: high - max(open, close) = 130 - 101 = 29, min(open, close) - low = 100 - 70 = 30
        # Shadow/body = 59/1 = 59, which is > 5.0 default threshold
        result = validator.validate_bar("EXT", 100.0, 130.0, 70.0, 101.0, 500, now_utc)
        assert result.is_valid is False
        assert OHLCRejectReason.EXTREME_CANDLE in result.reject_reasons

    def test_normal_candle_passes(self, validator: OHLCValidator, now_utc: datetime):
        # Typical candle: body is significant relative to shadows
        result = validator.validate_bar("NORM", 100.0, 108.0, 97.0, 106.0, 500, now_utc)
        assert OHLCRejectReason.EXTREME_CANDLE not in result.reject_reasons


# ──────────────────────────────────────────────────
# Statistics & summary
# ──────────────────────────────────────────────────


class TestStats:
    def test_get_stats_unknown_symbol(self, validator: OHLCValidator):
        stats = validator.get_stats("NONEXISTENT")
        assert stats["symbol"] == "NONEXISTENT"
        assert stats["total_bars"] == 0

    def test_valid_pct_calculation(self, validator: OHLCValidator, now_utc: datetime):
        validator.validate_bar("A", 100.0, 105.0, 98.0, 103.0, 500, now_utc)
        validator.validate_bar("A", 0, 0, 0, 0, 500, now_utc)  # rejected (zero price)
        stats = validator.get_stats("A")
        assert stats["total_bars"] == 2
        assert stats["valid_bars"] == 1
        assert stats["rejected_bars"] == 1
        assert stats["valid_pct"] == 50.0

    def test_summary_aggregation(self, validator: OHLCValidator, now_utc: datetime):
        validator.validate_bar("X", 100.0, 105.0, 98.0, 103.0, 500, now_utc)
        validator.validate_bar("Y", 0, 0, 0, 0, 500, now_utc)  # rejected
        summary = validator.get_summary()
        assert summary["total_symbols"] == 2
        assert summary["total_bars"] == 2
        assert summary["valid_bars"] == 1
        assert summary["rejected_bars"] == 1

    def test_gap_and_missing_bar_counts(self, validator: OHLCValidator, now_utc: datetime):
        t1 = now_utc - timedelta(minutes=10)
        t2 = now_utc - timedelta(minutes=5)
        validator.validate_bar("GM", 100.0, 105.0, 98.0, 100.0, 500, t1, interval="1m")
        # Large gap: opens at 120 (20% gap) + 5 min elapsed for 1m bars
        validator.validate_bar("GM", 120.0, 125.0, 118.0, 122.0, 500, t2, interval="1m")
        stats = validator.get_stats("GM")
        assert stats["total_gaps_detected"] >= 1
        assert stats["total_missing_bars"] >= 1


# ──────────────────────────────────────────────────
# Reset
# ──────────────────────────────────────────────────


class TestReset:
    def test_reset_single_symbol(self, validator: OHLCValidator, now_utc: datetime):
        validator.validate_bar("RST", 100.0, 105.0, 98.0, 103.0, 500, now_utc)
        validator.reset_stats("RST")
        stats = validator.get_stats("RST")
        assert stats["total_bars"] == 0
        # last_close should be preserved for gap detection continuity
        assert stats["last_close"] == 103.0

    def test_reset_all(self, validator: OHLCValidator, now_utc: datetime):
        validator.validate_bar("A", 100.0, 105.0, 98.0, 103.0, 500, now_utc)
        validator.validate_bar("B", 200.0, 210.0, 195.0, 205.0, 500, now_utc)
        validator.reset_stats()
        assert validator.get_all_stats() == {}


# ──────────────────────────────────────────────────
# Thread safety
# ──────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_validation(self, validator: OHLCValidator):
        errors = []

        def validate_many(symbol: str, count: int):
            try:
                for i in range(count):
                    ts = datetime.now(UTC) - timedelta(seconds=i % 10)
                    validator.validate_bar(
                        symbol,
                        100.0 + i,
                        110.0 + i,
                        90.0 + i,
                        105.0 + i,
                        1000,
                        ts,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=validate_many, args=(f"SYM{j}", 100)) for j in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        summary = validator.get_summary()
        assert summary["total_bars"] == 500
        assert summary["total_symbols"] == 5


# ──────────────────────────────────────────────────
# to_dict serialization
# ──────────────────────────────────────────────────


class TestSerialization:
    def test_to_dict(self, validator: OHLCValidator, now_utc: datetime):
        result = validator.validate_bar("SER", 100.0, 105.0, 98.0, 103.0, 500, now_utc)
        d = result.to_dict()
        assert d["is_valid"] is True
        assert d["symbol"] == "SER"
        assert d["open"] == 100.0
        assert d["high"] == 105.0
        assert d["low"] == 98.0
        assert d["close"] == 103.0
        assert isinstance(d["timestamp"], str)
        assert d["reject_reasons"] == []
        assert d["missing_bar_count"] == 0
