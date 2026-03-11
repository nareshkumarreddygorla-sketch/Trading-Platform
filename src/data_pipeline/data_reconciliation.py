"""
Multi-source data reconciliation for institutional-grade data integrity.

Compares data from multiple feeds/sources to detect price and volume
discrepancies, scores source reliability, and selects fallback sources
when the primary fails.
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

logger = logging.getLogger(__name__)


class DiscrepancyType(str, Enum):
    """Types of discrepancies between data sources."""

    PRICE_DIVERGENCE = "PRICE_DIVERGENCE"
    VOLUME_DIVERGENCE = "VOLUME_DIVERGENCE"
    MISSING_FROM_SOURCE = "MISSING_FROM_SOURCE"
    STALE_SOURCE = "STALE_SOURCE"


@dataclass
class SourceDataPoint:
    """A single data point from a named source."""

    source: str
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    received_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ReconciliationResult:
    """Result of reconciling data across sources."""

    symbol: str
    is_consistent: bool
    primary_source: str
    consensus_price: float | None = None
    consensus_volume: float | None = None
    price_spread: float = 0.0
    volume_spread_pct: float = 0.0
    discrepancies: list[dict] = field(default_factory=list)
    sources_reporting: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "is_consistent": self.is_consistent,
            "primary_source": self.primary_source,
            "consensus_price": self.consensus_price,
            "consensus_volume": self.consensus_volume,
            "price_spread": round(self.price_spread, 4),
            "volume_spread_pct": round(self.volume_spread_pct, 2),
            "discrepancies": self.discrepancies,
            "sources_reporting": self.sources_reporting,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SourceReliabilityStats:
    """Reliability statistics for a data source."""

    total_points: int = 0
    consistent_points: int = 0
    discrepant_points: int = 0
    stale_count: int = 0
    missing_count: int = 0
    last_update: datetime | None = None
    avg_latency_ms: float = 0.0
    _latency_sum: float = 0.0
    _latency_count: int = 0

    @property
    def reliability_score(self) -> float:
        """Reliability score 0-100 based on consistency and availability."""
        if self.total_points == 0:
            return 50.0  # No data = neutral score
        consistency = self.consistent_points / self.total_points
        # Penalize for stale/missing data
        availability_penalty = min(
            (self.stale_count + self.missing_count) / max(self.total_points, 1) * 0.5,
            0.5,
        )
        score = (consistency - availability_penalty) * 100.0
        return max(0.0, min(100.0, round(score, 1)))


class DataReconciliator:
    """
    Compares and reconciles data from multiple sources. Detects discrepancies
    in price and volume, scores source reliability, and provides fallback
    source selection.

    Args:
        price_tolerance_pct: Max allowed price difference in percent (default 0.5%).
        volume_tolerance_pct: Max allowed volume difference in percent (default 10%).
        stale_source_seconds: Seconds before a source is considered stale (default 120).
        primary_source: Name of the primary/preferred data source.
    """

    def __init__(
        self,
        *,
        price_tolerance_pct: float = 0.5,
        volume_tolerance_pct: float = 10.0,
        stale_source_seconds: float = 120.0,
        primary_source: str = "primary",
    ):
        self._price_tolerance_pct = price_tolerance_pct
        self._volume_tolerance_pct = volume_tolerance_pct
        self._stale_source_seconds = stale_source_seconds
        self._primary_source = primary_source

        # Latest data point from each source per symbol
        self._latest: dict[str, dict[str, SourceDataPoint]] = defaultdict(dict)
        # Source reliability stats
        self._source_stats: dict[str, SourceReliabilityStats] = defaultdict(SourceReliabilityStats)
        # Reconciliation history
        self._reconciliation_log: list[ReconciliationResult] = []
        self._max_log_size = 500

        self._lock = threading.RLock()

    def ingest(self, source: str, symbol: str, price: float, volume: float, timestamp: datetime) -> None:
        """
        Ingest a data point from a named source.

        Args:
            source: Name of the data source (e.g. "angel_one", "yahoo", "zerodha").
            symbol: Ticker symbol.
            price: Price from this source.
            volume: Volume from this source.
            timestamp: Data timestamp.
        """
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)

        now = datetime.now(UTC)
        data_point = SourceDataPoint(
            source=source,
            symbol=symbol,
            price=price,
            volume=volume,
            timestamp=timestamp,
            received_at=now,
        )

        with self._lock:
            self._latest[symbol][source] = data_point

            stats = self._source_stats[source]
            stats.total_points += 1
            stats.last_update = now

            # Track latency (time from data timestamp to receipt)
            latency_ms = (now - timestamp).total_seconds() * 1000
            if latency_ms >= 0:
                stats._latency_sum += latency_ms
                stats._latency_count += 1
                stats.avg_latency_ms = stats._latency_sum / stats._latency_count

    def reconcile(self, symbol: str) -> ReconciliationResult:
        """
        Reconcile data for a symbol across all sources that have reported.

        Returns:
            ReconciliationResult with consistency status and any discrepancies.
        """
        with self._lock:
            sources = self._latest.get(symbol, {})
            if not sources:
                return ReconciliationResult(
                    symbol=symbol,
                    is_consistent=True,
                    primary_source=self._primary_source,
                    sources_reporting=[],
                )

            now = datetime.now(UTC)
            active_sources: dict[str, SourceDataPoint] = {}
            discrepancies: list[dict] = []

            # Filter out stale sources
            for source_name, dp in sources.items():
                age = (now - dp.received_at).total_seconds()
                if age > self._stale_source_seconds:
                    discrepancies.append(
                        {
                            "type": DiscrepancyType.STALE_SOURCE.value,
                            "source": source_name,
                            "age_seconds": round(age, 1),
                            "message": f"Source {source_name} is stale ({age:.0f}s old)",
                        }
                    )
                    self._source_stats[source_name].stale_count += 1
                else:
                    active_sources[source_name] = dp

            if len(active_sources) == 0:
                return ReconciliationResult(
                    symbol=symbol,
                    is_consistent=False,
                    primary_source=self._primary_source,
                    discrepancies=discrepancies,
                    sources_reporting=list(sources.keys()),
                )

            if len(active_sources) == 1:
                # Only one active source: consistent by default
                src_name, dp = next(iter(active_sources.items()))
                self._source_stats[src_name].consistent_points += 1
                result = ReconciliationResult(
                    symbol=symbol,
                    is_consistent=True,
                    primary_source=src_name,
                    consensus_price=dp.price,
                    consensus_volume=dp.volume,
                    sources_reporting=[src_name],
                    discrepancies=discrepancies,
                )
                self._log_result(result)
                return result

            # Multiple sources: compare prices and volumes
            prices = {name: dp.price for name, dp in active_sources.items()}
            volumes = {name: dp.volume for name, dp in active_sources.items()}

            # Consensus price = median
            sorted_prices = sorted(prices.values())
            n = len(sorted_prices)
            if n % 2 == 0:
                consensus_price = (sorted_prices[n // 2 - 1] + sorted_prices[n // 2]) / 2
            else:
                consensus_price = sorted_prices[n // 2]

            # Consensus volume = median
            sorted_volumes = sorted(volumes.values())
            if n % 2 == 0:
                consensus_volume = (sorted_volumes[n // 2 - 1] + sorted_volumes[n // 2]) / 2
            else:
                consensus_volume = sorted_volumes[n // 2]

            # Price spread
            price_spread = max(sorted_prices) - min(sorted_prices)

            # Volume spread percentage
            if consensus_volume > 0:
                vol_spread_pct = (max(sorted_volumes) - min(sorted_volumes)) / consensus_volume * 100
            else:
                vol_spread_pct = 0.0

            is_consistent = True

            # Check price discrepancies
            for src_name, price in prices.items():
                if consensus_price > 0:
                    pct_diff = abs(price - consensus_price) / consensus_price * 100
                    if pct_diff > self._price_tolerance_pct:
                        is_consistent = False
                        discrepancies.append(
                            {
                                "type": DiscrepancyType.PRICE_DIVERGENCE.value,
                                "source": src_name,
                                "source_price": price,
                                "consensus_price": consensus_price,
                                "divergence_pct": round(pct_diff, 4),
                                "message": (
                                    f"Price divergence: {src_name} reports {price:.2f} "
                                    f"vs consensus {consensus_price:.2f} ({pct_diff:.2f}%)"
                                ),
                            }
                        )
                        self._source_stats[src_name].discrepant_points += 1
                    else:
                        self._source_stats[src_name].consistent_points += 1

            # Check volume discrepancies
            for src_name, vol in volumes.items():
                if consensus_volume > 0:
                    pct_diff = abs(vol - consensus_volume) / consensus_volume * 100
                    if pct_diff > self._volume_tolerance_pct:
                        discrepancies.append(
                            {
                                "type": DiscrepancyType.VOLUME_DIVERGENCE.value,
                                "source": src_name,
                                "source_volume": vol,
                                "consensus_volume": consensus_volume,
                                "divergence_pct": round(pct_diff, 2),
                                "message": (
                                    f"Volume divergence: {src_name} reports {vol:.0f} "
                                    f"vs consensus {consensus_volume:.0f} ({pct_diff:.1f}%)"
                                ),
                            }
                        )

            # Check for sources that are missing data for this symbol
            all_known_sources = set(self._source_stats.keys())
            reporting_sources = set(active_sources.keys())
            for missing_src in (
                all_known_sources - reporting_sources - set(s for s in sources if s not in active_sources)
            ):
                self._source_stats[missing_src].missing_count += 1

            result = ReconciliationResult(
                symbol=symbol,
                is_consistent=is_consistent,
                primary_source=self._select_best_source(active_sources),
                consensus_price=round(consensus_price, 4),
                consensus_volume=round(consensus_volume, 2),
                price_spread=price_spread,
                volume_spread_pct=vol_spread_pct,
                discrepancies=discrepancies,
                sources_reporting=list(active_sources.keys()),
            )

            self._log_result(result)
            return result

    def _select_best_source(self, active_sources: dict[str, SourceDataPoint]) -> str:
        """Select the best source based on reliability and preference."""
        # Prefer primary source if it is active and reliable
        if self._primary_source in active_sources:
            primary_stats = self._source_stats.get(self._primary_source)
            if primary_stats and primary_stats.reliability_score >= 70:
                return self._primary_source

        # Otherwise, choose the source with the highest reliability score
        best_source = self._primary_source
        best_score = -1.0
        for src_name in active_sources:
            stats = self._source_stats.get(src_name)
            score = stats.reliability_score if stats else 50.0
            if score > best_score:
                best_score = score
                best_source = src_name

        return best_source

    def get_best_source(self, symbol: str) -> str:
        """
        Get the best (most reliable) source for a symbol.

        Falls back to primary if no data available.
        """
        with self._lock:
            sources = self._latest.get(symbol, {})
            if not sources:
                return self._primary_source

            now = datetime.now(UTC)
            active = {
                name: dp
                for name, dp in sources.items()
                if (now - dp.received_at).total_seconds() <= self._stale_source_seconds
            }
            if not active:
                return self._primary_source

            return self._select_best_source(active)

    def get_source_reliability(self) -> dict[str, dict]:
        """Get reliability scores and stats for all known sources."""
        with self._lock:
            result = {}
            for source, stats in self._source_stats.items():
                result[source] = {
                    "reliability_score": stats.reliability_score,
                    "total_points": stats.total_points,
                    "consistent_points": stats.consistent_points,
                    "discrepant_points": stats.discrepant_points,
                    "stale_count": stats.stale_count,
                    "missing_count": stats.missing_count,
                    "avg_latency_ms": round(stats.avg_latency_ms, 1),
                    "last_update": stats.last_update.isoformat() if stats.last_update else None,
                }
            return result

    def get_reconciliation_summary(self) -> dict:
        """Get overall reconciliation summary."""
        with self._lock:
            total = len(self._reconciliation_log)
            consistent = sum(1 for r in self._reconciliation_log if r.is_consistent)
            inconsistent = total - consistent
            consistency_rate = (consistent / total * 100) if total > 0 else 100.0

            return {
                "total_reconciliations": total,
                "consistent": consistent,
                "inconsistent": inconsistent,
                "consistency_rate": round(consistency_rate, 2),
                "sources": self.get_source_reliability(),
                "primary_source": self._primary_source,
                "symbols_tracked": list(self._latest.keys()),
            }

    def _log_result(self, result: ReconciliationResult) -> None:
        """Log a reconciliation result."""
        self._reconciliation_log.append(result)
        if len(self._reconciliation_log) > self._max_log_size:
            self._reconciliation_log = self._reconciliation_log[-self._max_log_size :]

    def reset(self) -> None:
        """Reset all reconciliation state."""
        with self._lock:
            self._latest.clear()
            self._source_stats.clear()
            self._reconciliation_log.clear()
