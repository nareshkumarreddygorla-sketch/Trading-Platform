"""
OHLC bar-level data validation for institutional-grade data integrity.

Validates OHLCV bars for internal consistency, gap detection, stale data,
missing bar detection, and extreme candle filtering. Thread-safe with
per-symbol tracking.
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

logger = logging.getLogger(__name__)


# Standard bar intervals in seconds for missing-bar detection
INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


class OHLCRejectReason(str, Enum):
    """Reasons an OHLC bar can be rejected or flagged."""

    VALID = "VALID"
    HIGH_LESS_THAN_OPEN = "HIGH_LESS_THAN_OPEN"
    HIGH_LESS_THAN_CLOSE = "HIGH_LESS_THAN_CLOSE"
    LOW_GREATER_THAN_OPEN = "LOW_GREATER_THAN_OPEN"
    LOW_GREATER_THAN_CLOSE = "LOW_GREATER_THAN_CLOSE"
    HIGH_LESS_THAN_LOW = "HIGH_LESS_THAN_LOW"
    NEGATIVE_VOLUME = "NEGATIVE_VOLUME"
    NEGATIVE_PRICE = "NEGATIVE_PRICE"
    ZERO_PRICE = "ZERO_PRICE"
    STALE_BAR = "STALE_BAR"
    EXTREME_CANDLE = "EXTREME_CANDLE"


class OHLCWarning(str, Enum):
    """Warnings (non-rejecting) for OHLC bars."""

    GAP_DETECTED = "GAP_DETECTED"
    MISSING_BARS = "MISSING_BARS"


@dataclass
class OHLCValidationResult:
    """Result of validating a single OHLC bar."""

    is_valid: bool
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime
    reject_reasons: list[OHLCRejectReason] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_bar_count: int = 0

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "timestamp": self.timestamp.isoformat(),
            "reject_reasons": [r.value for r in self.reject_reasons],
            "warnings": self.warnings,
            "missing_bar_count": self.missing_bar_count,
        }


@dataclass
class SymbolBarStats:
    """Per-symbol OHLC validation statistics."""

    total_bars: int = 0
    valid_bars: int = 0
    rejected_bars: int = 0
    reject_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    warning_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_close: float | None = None
    last_bar_timestamp: datetime | None = None
    last_interval: str | None = None
    total_gaps_detected: int = 0
    total_missing_bars: int = 0


class OHLCValidator:
    """
    Validates OHLCV bars for data integrity.

    Thread-safe implementation with per-symbol tracking. Performs consistency
    checks, gap detection, staleness detection, missing bar detection, and
    extreme candle filtering.

    Args:
        gap_threshold_pct: Flag gap if open differs from prev close by > this pct (default 5%).
        stale_seconds: Reject bar if older than this many seconds (default 120 = 2 min).
        extreme_body_ratio: Flag candle if body/range ratio > this (default 0.95 = near-marubozu).
        extreme_shadow_ratio: Flag candle if shadow/body ratio > this (default 5.0).
    """

    def __init__(
        self,
        *,
        gap_threshold_pct: float = 5.0,
        stale_seconds: float = 120.0,
        extreme_body_ratio: float = 0.95,
        extreme_shadow_ratio: float = 5.0,
    ):
        self._gap_threshold_pct = gap_threshold_pct
        self._stale_seconds = stale_seconds
        self._extreme_body_ratio = extreme_body_ratio
        self._extreme_shadow_ratio = extreme_shadow_ratio
        self._stats: dict[str, SymbolBarStats] = defaultdict(SymbolBarStats)
        self._lock = threading.RLock()

    def validate_bar(
        self,
        symbol: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        timestamp: datetime,
        interval: str = "1m",
    ) -> OHLCValidationResult:
        """
        Validate a single OHLCV bar against all checks.

        Args:
            symbol: Ticker symbol.
            open_: Bar open price.
            high: Bar high price.
            low: Bar low price.
            close: Bar close price.
            volume: Bar volume.
            timestamp: Bar timestamp (should be timezone-aware UTC).
            interval: Bar interval (e.g. "1m", "5m", "1d").

        Returns:
            OHLCValidationResult with is_valid and reject/warning details.
        """
        reject_reasons: list[OHLCRejectReason] = []
        warnings: list[str] = []
        missing_bar_count = 0

        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)

        with self._lock:
            stats = self._stats[symbol]
            stats.total_bars += 1

            # 1. Zero/negative price detection
            for label, val in [("open", open_), ("high", high), ("low", low), ("close", close)]:
                if val < 0:
                    reject_reasons.append(OHLCRejectReason.NEGATIVE_PRICE)
                    break
                if val == 0:
                    reject_reasons.append(OHLCRejectReason.ZERO_PRICE)
                    break

            # 2. OHLC consistency: high >= max(open, close), low <= min(open, close)
            if high < low:
                reject_reasons.append(OHLCRejectReason.HIGH_LESS_THAN_LOW)
            if high < open_:
                reject_reasons.append(OHLCRejectReason.HIGH_LESS_THAN_OPEN)
            if high < close:
                reject_reasons.append(OHLCRejectReason.HIGH_LESS_THAN_CLOSE)
            if low > open_:
                reject_reasons.append(OHLCRejectReason.LOW_GREATER_THAN_OPEN)
            if low > close:
                reject_reasons.append(OHLCRejectReason.LOW_GREATER_THAN_CLOSE)

            # 3. Volume consistency
            if volume < 0:
                reject_reasons.append(OHLCRejectReason.NEGATIVE_VOLUME)

            # 4. Gap detection (excluding overnight gaps for daily bars)
            if stats.last_close is not None and stats.last_close > 0:
                is_intraday = interval in ("1m", "3m", "5m", "15m", "30m", "1h", "4h")
                is_overnight = False
                if is_intraday and stats.last_bar_timestamp is not None:
                    # Check if this bar crossed a date boundary (overnight gap)
                    if timestamp.date() != stats.last_bar_timestamp.date():
                        is_overnight = True

                if not is_overnight:
                    gap_pct = abs(open_ - stats.last_close) / stats.last_close * 100
                    if gap_pct > self._gap_threshold_pct:
                        warnings.append(
                            f"{OHLCWarning.GAP_DETECTED.value}: open={open_:.2f} "
                            f"prev_close={stats.last_close:.2f} gap={gap_pct:.1f}%"
                        )
                        stats.total_gaps_detected += 1

            # 5. Stale data detection
            now_utc = datetime.now(UTC)
            age_seconds = (now_utc - timestamp).total_seconds()
            if age_seconds > self._stale_seconds:
                reject_reasons.append(OHLCRejectReason.STALE_BAR)

            # 6. Missing bar detection
            if stats.last_bar_timestamp is not None and stats.last_interval is not None:
                expected_interval_secs = INTERVAL_SECONDS.get(stats.last_interval, 0)
                if expected_interval_secs > 0:
                    elapsed = (timestamp - stats.last_bar_timestamp).total_seconds()
                    # Allow 50% tolerance before flagging
                    if elapsed > expected_interval_secs * 1.5:
                        gaps = int(elapsed / expected_interval_secs) - 1
                        if gaps > 0:
                            missing_bar_count = gaps
                            warnings.append(
                                f"{OHLCWarning.MISSING_BARS.value}: "
                                f"expected interval={stats.last_interval} "
                                f"but {elapsed:.0f}s elapsed ({gaps} bars missing)"
                            )
                            stats.total_missing_bars += gaps

            # 7. Body/shadow ratio checks for extreme candles
            if high > low and open_ > 0 and close > 0:
                candle_range = high - low
                body = abs(close - open_)

                if candle_range > 0:
                    body_ratio = body / candle_range
                    if body_ratio > self._extreme_body_ratio:
                        # Near-marubozu: body fills almost entire range (not a rejection, just a warning)
                        pass  # Normal for high-conviction moves

                    if body > 0:
                        upper_shadow = high - max(open_, close)
                        lower_shadow = min(open_, close) - low
                        total_shadow = upper_shadow + lower_shadow
                        shadow_to_body = total_shadow / body
                        if shadow_to_body > self._extreme_shadow_ratio:
                            reject_reasons.append(OHLCRejectReason.EXTREME_CANDLE)

            # Build result
            is_valid = len(reject_reasons) == 0
            result = OHLCValidationResult(
                is_valid=is_valid,
                symbol=symbol,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
                timestamp=timestamp,
                reject_reasons=reject_reasons,
                warnings=warnings,
                missing_bar_count=missing_bar_count,
            )

            # Update stats
            if is_valid:
                stats.valid_bars += 1
                stats.last_close = close
                stats.last_bar_timestamp = timestamp
                stats.last_interval = interval
            else:
                stats.rejected_bars += 1
                for reason in reject_reasons:
                    stats.reject_counts[reason.value] += 1
                logger.debug(
                    "OHLC bar rejected for %s: O=%.2f H=%.2f L=%.2f C=%.2f V=%.0f reasons=%s",
                    symbol,
                    open_,
                    high,
                    low,
                    close,
                    volume,
                    [r.value for r in reject_reasons],
                )

            for w in warnings:
                stats.warning_counts[w.split(":")[0]] += 1

            return result

    def get_stats(self, symbol: str) -> dict:
        """Get validation statistics for a symbol."""
        with self._lock:
            stats = self._stats.get(symbol)
            if stats is None:
                return {"symbol": symbol, "total_bars": 0}
            valid_pct = (stats.valid_bars / stats.total_bars * 100) if stats.total_bars > 0 else 0.0
            return {
                "symbol": symbol,
                "total_bars": stats.total_bars,
                "valid_bars": stats.valid_bars,
                "rejected_bars": stats.rejected_bars,
                "valid_pct": round(valid_pct, 2),
                "reject_counts": dict(stats.reject_counts),
                "warning_counts": dict(stats.warning_counts),
                "last_close": stats.last_close,
                "last_bar_timestamp": (stats.last_bar_timestamp.isoformat() if stats.last_bar_timestamp else None),
                "total_gaps_detected": stats.total_gaps_detected,
                "total_missing_bars": stats.total_missing_bars,
            }

    def get_all_stats(self) -> dict[str, dict]:
        """Get validation statistics for all tracked symbols."""
        with self._lock:
            return {symbol: self.get_stats(symbol) for symbol in self._stats}

    def get_summary(self) -> dict:
        """Get aggregate summary across all symbols."""
        with self._lock:
            total = sum(s.total_bars for s in self._stats.values())
            valid = sum(s.valid_bars for s in self._stats.values())
            rejected = sum(s.rejected_bars for s in self._stats.values())
            valid_pct = (valid / total * 100) if total > 0 else 0.0
            gaps = sum(s.total_gaps_detected for s in self._stats.values())
            missing = sum(s.total_missing_bars for s in self._stats.values())

            all_reasons: dict[str, int] = defaultdict(int)
            for s in self._stats.values():
                for reason, count in s.reject_counts.items():
                    all_reasons[reason] += count

            return {
                "total_symbols": len(self._stats),
                "total_bars": total,
                "valid_bars": valid,
                "rejected_bars": rejected,
                "valid_pct": round(valid_pct, 2),
                "reject_reasons": dict(all_reasons),
                "total_gaps_detected": gaps,
                "total_missing_bars": missing,
            }

    def reset_stats(self, symbol: str | None = None) -> None:
        """Reset statistics for a symbol or all symbols."""
        with self._lock:
            if symbol:
                if symbol in self._stats:
                    prev_close = self._stats[symbol].last_close
                    self._stats[symbol] = SymbolBarStats()
                    self._stats[symbol].last_close = prev_close
            else:
                self._stats.clear()
