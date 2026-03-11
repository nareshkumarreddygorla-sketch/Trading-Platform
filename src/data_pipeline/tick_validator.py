"""
Tick-level data validation for institutional-grade data integrity.

Validates individual ticks against price bounds, volume sanity, timestamp
freshness, NSE circuit limits, zero-price detection, and duplicate detection.
Thread-safe with per-symbol statistics tracking.
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class TickRejectReason(str, Enum):
    """Reasons a tick can be rejected."""

    VALID = "VALID"
    NEGATIVE_PRICE = "NEGATIVE_PRICE"
    ZERO_PRICE = "ZERO_PRICE"
    PRICE_EXCEEDS_BOUND = "PRICE_EXCEEDS_BOUND"
    NEGATIVE_VOLUME = "NEGATIVE_VOLUME"
    VOLUME_EXCEEDS_BOUND = "VOLUME_EXCEEDS_BOUND"
    FUTURE_TIMESTAMP = "FUTURE_TIMESTAMP"
    STALE_TIMESTAMP = "STALE_TIMESTAMP"
    CIRCUIT_LIMIT_BREACH = "CIRCUIT_LIMIT_BREACH"
    DUPLICATE_TICK = "DUPLICATE_TICK"


@dataclass
class TickValidationResult:
    """Result of validating a single tick."""

    is_valid: bool
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    reject_reasons: list[TickRejectReason] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "symbol": self.symbol,
            "price": self.price,
            "volume": self.volume,
            "timestamp": self.timestamp.isoformat(),
            "reject_reasons": [r.value for r in self.reject_reasons],
            "warnings": self.warnings,
        }


@dataclass
class SymbolTickStats:
    """Per-symbol tick validation statistics."""

    total_ticks: int = 0
    valid_ticks: int = 0
    rejected_ticks: int = 0
    reject_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_valid_price: float | None = None
    last_valid_volume: float | None = None
    last_valid_timestamp: datetime | None = None
    previous_close: float | None = None
    average_daily_volume: float | None = None
    # For duplicate detection
    last_tick_key: tuple[float, float] | None = None  # (timestamp_epoch, price)


class TickValidator:
    """
    Validates individual market ticks for data integrity.

    Thread-safe implementation with per-symbol tracking. Configurable
    thresholds for price bounds, volume limits, staleness, and circuit limits.

    Args:
        price_bound_multiplier: Reject if price > multiplier * previous_close (default 10x).
        volume_bound_multiplier: Reject if volume > multiplier * avg_daily_volume (default 100x).
        stale_seconds: Reject if tick is older than this many seconds (default 300 = 5 min).
        circuit_limit_pct: Reject if price beyond +/- this pct of previous close (default 20%).
    """

    def __init__(
        self,
        *,
        price_bound_multiplier: float = 10.0,
        volume_bound_multiplier: float = 100.0,
        stale_seconds: float = 300.0,
        circuit_limit_pct: float = 20.0,
    ):
        self._price_bound_mult = price_bound_multiplier
        self._volume_bound_mult = volume_bound_multiplier
        self._stale_seconds = stale_seconds
        self._circuit_limit_pct = circuit_limit_pct
        self._stats: dict[str, SymbolTickStats] = defaultdict(SymbolTickStats)
        self._lock = threading.RLock()

    def set_previous_close(self, symbol: str, price: float) -> None:
        """Set the previous close price for a symbol (used for bounds and circuit checks)."""
        with self._lock:
            self._stats[symbol].previous_close = price

    def set_average_daily_volume(self, symbol: str, volume: float) -> None:
        """Set the average daily volume for a symbol (used for volume sanity)."""
        with self._lock:
            self._stats[symbol].average_daily_volume = volume

    def validate_tick(
        self,
        symbol: str,
        price: float,
        volume: float,
        timestamp: datetime,
    ) -> TickValidationResult:
        """
        Validate a single tick against all checks.

        Args:
            symbol: Ticker symbol (e.g. "RELIANCE").
            price: Tick price.
            volume: Tick volume (trade size).
            timestamp: Tick timestamp (should be timezone-aware UTC).

        Returns:
            TickValidationResult with is_valid=True/False and reject reasons.
        """
        reject_reasons: list[TickRejectReason] = []
        warnings: list[str] = []

        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)

        with self._lock:
            stats = self._stats[symbol]
            stats.total_ticks += 1

            # 1. Zero-price detection
            if price == 0:
                reject_reasons.append(TickRejectReason.ZERO_PRICE)

            # 2. Negative price
            if price < 0:
                reject_reasons.append(TickRejectReason.NEGATIVE_PRICE)

            # 3. Price bounds check (only if previous close is known)
            if stats.previous_close is not None and stats.previous_close > 0:
                upper_bound = stats.previous_close * self._price_bound_mult
                if price > upper_bound:
                    reject_reasons.append(TickRejectReason.PRICE_EXCEEDS_BOUND)

            # 4. Volume sanity
            if volume < 0:
                reject_reasons.append(TickRejectReason.NEGATIVE_VOLUME)

            if (
                stats.average_daily_volume is not None
                and stats.average_daily_volume > 0
                and volume > stats.average_daily_volume * self._volume_bound_mult
            ):
                reject_reasons.append(TickRejectReason.VOLUME_EXCEEDS_BOUND)

            # 5. Timestamp validation
            now_utc = datetime.now(UTC)

            # Future timestamp (allow 2 seconds clock skew)
            if timestamp > now_utc + timedelta(seconds=2):
                reject_reasons.append(TickRejectReason.FUTURE_TIMESTAMP)

            # Stale timestamp
            age_seconds = (now_utc - timestamp).total_seconds()
            if age_seconds > self._stale_seconds:
                reject_reasons.append(TickRejectReason.STALE_TIMESTAMP)

            # 6. Circuit limit check (NSE-style: +/- 20% of previous close)
            if stats.previous_close is not None and stats.previous_close > 0 and price > 0:
                pct_change = abs(price - stats.previous_close) / stats.previous_close * 100
                if pct_change > self._circuit_limit_pct:
                    reject_reasons.append(TickRejectReason.CIRCUIT_LIMIT_BREACH)

            # 7. Duplicate tick detection (same timestamp + price)
            tick_key = (timestamp.timestamp(), price)
            if stats.last_tick_key == tick_key:
                reject_reasons.append(TickRejectReason.DUPLICATE_TICK)
            stats.last_tick_key = tick_key

            # Build result
            is_valid = len(reject_reasons) == 0
            result = TickValidationResult(
                is_valid=is_valid,
                symbol=symbol,
                price=price,
                volume=volume,
                timestamp=timestamp,
                reject_reasons=reject_reasons,
                warnings=warnings,
            )

            # Update stats
            if is_valid:
                stats.valid_ticks += 1
                stats.last_valid_price = price
                stats.last_valid_volume = volume
                stats.last_valid_timestamp = timestamp
            else:
                stats.rejected_ticks += 1
                for reason in reject_reasons:
                    stats.reject_counts[reason.value] += 1
                logger.debug(
                    "Tick rejected for %s: price=%.2f vol=%.0f reasons=%s",
                    symbol,
                    price,
                    volume,
                    [r.value for r in reject_reasons],
                )

            return result

    def get_stats(self, symbol: str) -> dict:
        """Get validation statistics for a symbol."""
        with self._lock:
            stats = self._stats.get(symbol)
            if stats is None:
                return {"symbol": symbol, "total_ticks": 0}
            valid_pct = (stats.valid_ticks / stats.total_ticks * 100) if stats.total_ticks > 0 else 0.0
            return {
                "symbol": symbol,
                "total_ticks": stats.total_ticks,
                "valid_ticks": stats.valid_ticks,
                "rejected_ticks": stats.rejected_ticks,
                "valid_pct": round(valid_pct, 2),
                "reject_counts": dict(stats.reject_counts),
                "last_valid_price": stats.last_valid_price,
                "last_valid_timestamp": (
                    stats.last_valid_timestamp.isoformat() if stats.last_valid_timestamp else None
                ),
                "previous_close": stats.previous_close,
                "average_daily_volume": stats.average_daily_volume,
            }

    def get_all_stats(self) -> dict[str, dict]:
        """Get validation statistics for all tracked symbols."""
        with self._lock:
            return {symbol: self.get_stats(symbol) for symbol in self._stats}

    def get_summary(self) -> dict:
        """Get aggregate summary across all symbols."""
        with self._lock:
            total = sum(s.total_ticks for s in self._stats.values())
            valid = sum(s.valid_ticks for s in self._stats.values())
            rejected = sum(s.rejected_ticks for s in self._stats.values())
            valid_pct = (valid / total * 100) if total > 0 else 0.0

            # Aggregate reject reasons
            all_reasons: dict[str, int] = defaultdict(int)
            for s in self._stats.values():
                for reason, count in s.reject_counts.items():
                    all_reasons[reason] += count

            return {
                "total_symbols": len(self._stats),
                "total_ticks": total,
                "valid_ticks": valid,
                "rejected_ticks": rejected,
                "valid_pct": round(valid_pct, 2),
                "reject_reasons": dict(all_reasons),
            }

    def reset_stats(self, symbol: str | None = None) -> None:
        """Reset statistics for a symbol or all symbols."""
        with self._lock:
            if symbol:
                if symbol in self._stats:
                    prev_close = self._stats[symbol].previous_close
                    adv = self._stats[symbol].average_daily_volume
                    self._stats[symbol] = SymbolTickStats()
                    self._stats[symbol].previous_close = prev_close
                    self._stats[symbol].average_daily_volume = adv
            else:
                self._stats.clear()
