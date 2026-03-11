"""
Real-time data quality monitoring for institutional-grade trading.

Aggregates tick and OHLC validator results into per-symbol quality scores,
detects stale feeds, tracks data freshness, and provides integration hooks
to halt trading on degraded data quality.
"""

import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

from .ohlc_validator import OHLCValidationResult, OHLCValidator
from .tick_validator import TickValidationResult, TickValidator

logger = logging.getLogger(__name__)


class DataQualityLevel(str, Enum):
    """Overall data quality classification."""

    EXCELLENT = "EXCELLENT"  # 90-100
    GOOD = "GOOD"  # 80-89
    DEGRADED = "DEGRADED"  # 60-79
    POOR = "POOR"  # 40-59
    CRITICAL = "CRITICAL"  # 0-39


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class DataQualityAlert:
    """A data quality alert event."""

    severity: AlertSeverity
    symbol: str
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    quality_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "severity": self.severity.value,
            "symbol": self.symbol,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "quality_score": self.quality_score,
        }


@dataclass
class SymbolQualityState:
    """Per-symbol data quality tracking state."""

    # Tick metrics
    tick_total: int = 0
    tick_valid: int = 0
    tick_rejected: int = 0

    # Bar metrics
    bar_total: int = 0
    bar_valid: int = 0
    bar_rejected: int = 0
    bar_missing_count: int = 0
    bar_gap_count: int = 0

    # Freshness
    last_tick_time: datetime | None = None
    last_bar_time: datetime | None = None
    last_update_time: datetime | None = None

    # Scoring
    quality_score: float = 100.0
    quality_level: DataQualityLevel = DataQualityLevel.EXCELLENT

    # Alert state
    alert_fired: bool = False


class DataQualityMonitor:
    """
    Aggregates tick and OHLC validation results into real-time data quality
    scores per symbol. Detects stale feeds, tracks freshness, and provides
    hooks to halt trading on bad data.

    Args:
        quality_threshold: Minimum acceptable quality score (default 80).
        stale_feed_seconds: Seconds of no updates before feed is stale (default 120).
        on_quality_degraded: Optional callback when quality drops below threshold.
            Receives (symbol: str, score: float, level: DataQualityLevel).
        on_halt_trading: Optional callback to halt trading.
            Receives (symbol: str, reason: str).
        halt_quality_threshold: Quality score below which trading halts (default 40).
        max_alerts: Maximum alerts to retain in memory (default 1000).
    """

    def __init__(
        self,
        *,
        quality_threshold: float = 80.0,
        stale_feed_seconds: float = 120.0,
        on_quality_degraded: Callable | None = None,
        on_halt_trading: Callable | None = None,
        halt_quality_threshold: float = 40.0,
        max_alerts: int = 1000,
    ):
        self._quality_threshold = quality_threshold
        self._stale_feed_seconds = stale_feed_seconds
        self._on_quality_degraded = on_quality_degraded
        self._on_halt_trading = on_halt_trading
        self._halt_quality_threshold = halt_quality_threshold
        self._max_alerts = max_alerts

        self._symbol_states: dict[str, SymbolQualityState] = defaultdict(SymbolQualityState)
        self._alerts: list[DataQualityAlert] = []
        self._halted_symbols: set[str] = set()
        self._source_freshness: dict[str, datetime] = {}
        self._lock = threading.RLock()

        # Validators (owned by monitor for centralized access)
        self.tick_validator = TickValidator()
        self.ohlc_validator = OHLCValidator()

    def record_tick_validation(self, result: TickValidationResult) -> None:
        """Record the result of a tick validation."""
        with self._lock:
            state = self._symbol_states[result.symbol]
            state.tick_total += 1
            now = datetime.now(UTC)
            state.last_update_time = now

            if result.is_valid:
                state.tick_valid += 1
                state.last_tick_time = now
            else:
                state.tick_rejected += 1

            self._recalculate_quality(result.symbol, state)

    def record_bar_validation(self, result: OHLCValidationResult) -> None:
        """Record the result of an OHLC bar validation."""
        with self._lock:
            state = self._symbol_states[result.symbol]
            state.bar_total += 1
            now = datetime.now(UTC)
            state.last_update_time = now

            if result.is_valid:
                state.bar_valid += 1
                state.last_bar_time = now
            else:
                state.bar_rejected += 1

            state.bar_missing_count += result.missing_bar_count
            if any("GAP_DETECTED" in w for w in result.warnings):
                state.bar_gap_count += 1

            self._recalculate_quality(result.symbol, state)

    def record_source_update(self, source_name: str) -> None:
        """Record a data update from a named source for freshness tracking."""
        with self._lock:
            self._source_freshness[source_name] = datetime.now(UTC)

    def _recalculate_quality(self, symbol: str, state: SymbolQualityState) -> None:
        """
        Recalculate quality score for a symbol.

        Score components (100 points total):
        - Tick acceptance rate: 40 points
        - Bar acceptance rate: 30 points
        - Data freshness: 20 points
        - Gap/missing penalty: 10 points
        """
        score = 100.0

        # Tick acceptance rate (40 points)
        if state.tick_total > 0:
            tick_rate = state.tick_valid / state.tick_total
            score -= (1.0 - tick_rate) * 40.0

        # Bar acceptance rate (30 points)
        if state.bar_total > 0:
            bar_rate = state.bar_valid / state.bar_total
            score -= (1.0 - bar_rate) * 30.0

        # Data freshness (20 points)
        now = datetime.now(UTC)
        latest_data = state.last_tick_time or state.last_bar_time
        if latest_data is not None:
            staleness_secs = (now - latest_data).total_seconds()
            if staleness_secs > self._stale_feed_seconds:
                # Linear decay: lose all 20 points over 2x the stale threshold
                decay = min(1.0, staleness_secs / (self._stale_feed_seconds * 2))
                score -= decay * 20.0
        elif state.tick_total > 0 or state.bar_total > 0:
            # Had data but no valid timestamps tracked
            score -= 10.0

        # Gap/missing bar penalty (10 points)
        total_data = state.bar_total + state.tick_total
        if total_data > 0:
            gap_ratio = (state.bar_missing_count + state.bar_gap_count) / max(total_data, 1)
            score -= min(gap_ratio * 50, 10.0)  # Cap at 10 points

        score = max(0.0, min(100.0, score))
        state.quality_score = round(score, 1)

        # Determine quality level
        if score >= 90:
            state.quality_level = DataQualityLevel.EXCELLENT
        elif score >= 80:
            state.quality_level = DataQualityLevel.GOOD
        elif score >= 60:
            state.quality_level = DataQualityLevel.DEGRADED
        elif score >= 40:
            state.quality_level = DataQualityLevel.POOR
        else:
            state.quality_level = DataQualityLevel.CRITICAL

        # Fire alerts on quality degradation
        if score < self._quality_threshold and not state.alert_fired:
            state.alert_fired = True
            alert = DataQualityAlert(
                severity=AlertSeverity.CRITICAL if score < self._halt_quality_threshold else AlertSeverity.WARNING,
                symbol=symbol,
                message=f"Data quality degraded: score={score:.1f} level={state.quality_level.value}",
                quality_score=score,
            )
            self._add_alert(alert)
            logger.warning(
                "Data quality alert for %s: score=%.1f level=%s",
                symbol,
                score,
                state.quality_level.value,
            )
            if self._on_quality_degraded:
                try:
                    self._on_quality_degraded(symbol, score, state.quality_level)
                except Exception as e:
                    logger.exception("on_quality_degraded callback failed: %s", e)

        # Halt trading if below critical threshold
        if score < self._halt_quality_threshold and symbol not in self._halted_symbols:
            self._halted_symbols.add(symbol)
            logger.critical(
                "HALTING TRADING for %s: data quality score=%.1f below threshold=%.1f",
                symbol,
                score,
                self._halt_quality_threshold,
            )
            if self._on_halt_trading:
                try:
                    self._on_halt_trading(symbol, f"Data quality critical: {score:.1f}")
                except Exception as e:
                    logger.exception("on_halt_trading callback failed: %s", e)

        # Re-enable if quality recovers above threshold
        if score >= self._quality_threshold:
            state.alert_fired = False
            self._halted_symbols.discard(symbol)

    def _add_alert(self, alert: DataQualityAlert) -> None:
        """Add an alert, enforcing max capacity."""
        self._alerts.append(alert)
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts :]

    def get_quality_score(self, symbol: str) -> float:
        """Get current quality score for a symbol (0-100)."""
        with self._lock:
            state = self._symbol_states.get(symbol)
            return state.quality_score if state else 100.0

    def get_quality_level(self, symbol: str) -> DataQualityLevel:
        """Get current quality level for a symbol."""
        with self._lock:
            state = self._symbol_states.get(symbol)
            return state.quality_level if state else DataQualityLevel.EXCELLENT

    def is_trading_halted(self, symbol: str) -> bool:
        """Check if trading is halted for a symbol due to bad data."""
        with self._lock:
            return symbol in self._halted_symbols

    def get_symbol_quality(self, symbol: str) -> dict:
        """Get detailed quality information for a symbol."""
        with self._lock:
            state = self._symbol_states.get(symbol)
            if state is None:
                return {
                    "symbol": symbol,
                    "quality_score": 100.0,
                    "quality_level": DataQualityLevel.EXCELLENT.value,
                    "has_data": False,
                }
            return {
                "symbol": symbol,
                "quality_score": state.quality_score,
                "quality_level": state.quality_level.value,
                "has_data": True,
                "tick_total": state.tick_total,
                "tick_valid": state.tick_valid,
                "tick_rejected": state.tick_rejected,
                "tick_acceptance_rate": round(state.tick_valid / state.tick_total * 100, 2)
                if state.tick_total > 0
                else 100.0,
                "bar_total": state.bar_total,
                "bar_valid": state.bar_valid,
                "bar_rejected": state.bar_rejected,
                "bar_acceptance_rate": round(state.bar_valid / state.bar_total * 100, 2)
                if state.bar_total > 0
                else 100.0,
                "bar_missing_count": state.bar_missing_count,
                "bar_gap_count": state.bar_gap_count,
                "last_tick_time": state.last_tick_time.isoformat() if state.last_tick_time else None,
                "last_bar_time": state.last_bar_time.isoformat() if state.last_bar_time else None,
                "last_update_time": state.last_update_time.isoformat() if state.last_update_time else None,
                "trading_halted": symbol in self._halted_symbols,
            }

    def get_all_quality_scores(self) -> dict[str, dict]:
        """Get quality scores for all tracked symbols."""
        with self._lock:
            return {
                symbol: {
                    "quality_score": state.quality_score,
                    "quality_level": state.quality_level.value,
                    "trading_halted": symbol in self._halted_symbols,
                }
                for symbol, state in self._symbol_states.items()
            }

    def get_staleness_report(self) -> dict:
        """Get data freshness/staleness report for all symbols and sources."""
        with self._lock:
            now = datetime.now(UTC)
            symbol_staleness = {}
            stale_symbols = []

            for symbol, state in self._symbol_states.items():
                latest = state.last_tick_time or state.last_bar_time
                if latest is None:
                    staleness_secs = None
                    is_stale = state.tick_total > 0 or state.bar_total > 0
                else:
                    staleness_secs = round((now - latest).total_seconds(), 1)
                    is_stale = staleness_secs > self._stale_feed_seconds

                symbol_staleness[symbol] = {
                    "last_data_time": latest.isoformat() if latest else None,
                    "staleness_seconds": staleness_secs,
                    "is_stale": is_stale,
                }
                if is_stale:
                    stale_symbols.append(symbol)

            source_staleness = {}
            for source, ts in self._source_freshness.items():
                age = round((now - ts).total_seconds(), 1)
                source_staleness[source] = {
                    "last_update": ts.isoformat(),
                    "age_seconds": age,
                    "is_stale": age > self._stale_feed_seconds,
                }

            return {
                "timestamp": now.isoformat(),
                "stale_feed_threshold_seconds": self._stale_feed_seconds,
                "total_symbols": len(self._symbol_states),
                "stale_symbols": stale_symbols,
                "stale_count": len(stale_symbols),
                "symbols": symbol_staleness,
                "sources": source_staleness,
            }

    def get_alerts(self, limit: int = 50) -> list[dict]:
        """Get recent alerts."""
        with self._lock:
            return [a.to_dict() for a in self._alerts[-limit:]]

    def get_summary(self) -> dict:
        """Get overall data quality summary."""
        with self._lock:
            scores = [s.quality_score for s in self._symbol_states.values()]
            avg_score = sum(scores) / len(scores) if scores else 100.0
            level_counts: dict[str, int] = defaultdict(int)
            for s in self._symbol_states.values():
                level_counts[s.quality_level.value] += 1

            return {
                "total_symbols": len(self._symbol_states),
                "average_quality_score": round(avg_score, 1),
                "halted_symbols": list(self._halted_symbols),
                "halted_count": len(self._halted_symbols),
                "quality_levels": dict(level_counts),
                "total_alerts": len(self._alerts),
                "quality_threshold": self._quality_threshold,
                "halt_threshold": self._halt_quality_threshold,
                "tick_stats": self.tick_validator.get_summary(),
                "bar_stats": self.ohlc_validator.get_summary(),
            }

    def get_validation_stats(self) -> dict:
        """Get combined validation statistics from tick and OHLC validators."""
        with self._lock:
            return {
                "tick_validation": self.tick_validator.get_summary(),
                "ohlc_validation": self.ohlc_validator.get_summary(),
                "symbols": {
                    symbol: {
                        "tick": self.tick_validator.get_stats(symbol),
                        "ohlc": self.ohlc_validator.get_stats(symbol),
                    }
                    for symbol in set(list(self.tick_validator._stats.keys()) + list(self.ohlc_validator._stats.keys()))
                },
            }

    def validate_and_record_tick(
        self, symbol: str, price: float, volume: float, timestamp: datetime
    ) -> TickValidationResult:
        """Convenience: validate a tick and record the result in one call."""
        result = self.tick_validator.validate_tick(symbol, price, volume, timestamp)
        self.record_tick_validation(result)
        return result

    def validate_and_record_bar(
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
        """Convenience: validate a bar and record the result in one call."""
        result = self.ohlc_validator.validate_bar(symbol, open_, high, low, close, volume, timestamp, interval)
        self.record_bar_validation(result)
        return result

    def reset(self) -> None:
        """Reset all monitoring state."""
        with self._lock:
            self._symbol_states.clear()
            self._alerts.clear()
            self._halted_symbols.clear()
            self._source_freshness.clear()
            self.tick_validator.reset_stats()
            self.ohlc_validator.reset_stats()
