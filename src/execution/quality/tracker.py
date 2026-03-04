"""
Phase 8: Execution quality tracker.
Record fills (expected vs realized slippage), rejections, partial fills;
slippage_ratio, rejection_rate, partial_fill_rate; recommend_size_multiplier, recommend_disable.
"""
from dataclasses import dataclass, field
from typing import Deque, Optional

import collections


@dataclass
class QualityMetrics:
    slippage_ratio: float  # realized / expected (e.g. > 1.5 -> bad)
    rejection_rate: float
    partial_fill_rate: float
    mean_latency_ms: float
    n_fills: int
    n_rejections: int
    n_orders: int


class ExecutionQualityTracker:
    """
    Rolling window: expected_slippage_bps, realized_slippage_bps, partial_fill (bool), rejection (bool), latency_ms.
    Aggregate: slippage_ratio, rejection_rate, partial_fill_rate.
    recommend_size_multiplier(): < 1 if slippage/rejections bad; recommend_disable(): True if over threshold.
    """

    def __init__(self, window: int = 100):
        self.window = window
        self._expected_slip: Deque[float] = collections.deque(maxlen=window)
        self._realized_slip: Deque[float] = collections.deque(maxlen=window)
        self._partial_fill: Deque[bool] = collections.deque(maxlen=window)
        self._rejections: int = 0
        self._orders: int = 0
        self._latency_ms: Deque[float] = collections.deque(maxlen=window)
        self._slippage_ratio_threshold: float = 1.5
        self._rejection_rate_threshold: float = 0.2
        self._partial_fill_rate_threshold: float = 0.3
        self._size_multiplier_on_degrade: float = 0.7

    def record_fill(
        self,
        expected_slippage_bps: float,
        realized_slippage_bps: float,
        partial_fill: bool = False,
        latency_ms: Optional[float] = None,
    ) -> None:
        self._orders += 1
        self._expected_slip.append(expected_slippage_bps)
        self._realized_slip.append(realized_slippage_bps)
        self._partial_fill.append(partial_fill)
        if latency_ms is not None:
            self._latency_ms.append(latency_ms)

    def record_rejection(self) -> None:
        self._rejections += 1
        self._orders += 1

    def get_slippage_ratio(self) -> float:
        if not self._expected_slip or sum(self._expected_slip) <= 0:
            return 1.0
        return sum(self._realized_slip) / sum(self._expected_slip)

    def get_rejection_rate(self) -> float:
        if self._orders <= 0:
            return 0.0
        return self._rejections / self._orders

    def get_partial_fill_rate(self) -> float:
        if not self._partial_fill:
            return 0.0
        return sum(self._partial_fill) / len(self._partial_fill)

    def get_metrics(self) -> QualityMetrics:
        sr = self.get_slippage_ratio()
        rr = self.get_rejection_rate()
        pfr = self.get_partial_fill_rate()
        lat = sum(self._latency_ms) / len(self._latency_ms) if self._latency_ms else 0.0
        return QualityMetrics(
            slippage_ratio=sr,
            rejection_rate=rr,
            partial_fill_rate=pfr,
            mean_latency_ms=lat,
            n_fills=len(self._expected_slip),
            n_rejections=self._rejections,
            n_orders=self._orders,
        )

    def recommend_size_multiplier(self) -> float:
        """Return multiplier for position size (e.g. 0.7 if execution degraded)."""
        sr = self.get_slippage_ratio()
        rr = self.get_rejection_rate()
        if sr >= self._slippage_ratio_threshold or rr >= self._rejection_rate_threshold:
            return self._size_multiplier_on_degrade
        return 1.0

    def recommend_disable(self) -> bool:
        """True if rejection or partial fill rate over threshold (strategy/symbol disable)."""
        if self.get_rejection_rate() >= self._rejection_rate_threshold:
            return True
        if self.get_partial_fill_rate() >= self._partial_fill_rate_threshold:
            return True
        return False
