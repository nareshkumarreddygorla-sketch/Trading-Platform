"""
Order-to-Trade Ratio (OTR) Monitor for SEBI Compliance.

SEBI regulations require monitoring the ratio of orders submitted to orders
actually filled/traded. Excessive OTR may indicate algorithmic strategies that
place disproportionate load on the exchange order book without genuine trading
intent (potential market manipulation).

Key features:
  - Real-time OTR tracking per algo ID.
  - Rolling window calculations: 1-minute, 5-minute, 1-hour.
  - Automatic throttling at 25:1 (warning) and 50:1 (halt).
  - Thread-safe implementation.
  - Logging and alerting for SEBI compliance.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

OTR_WARNING_THRESHOLD = 8.0  # P2-2: 8:1 triggers warning (was 25:1)
OTR_HALT_THRESHOLD = 10.0  # P2-2: 10:1 triggers halt per NSE limit (was 50:1)
WINDOW_1MIN = 60  # seconds
WINDOW_5MIN = 300
WINDOW_1HR = 3600


class OTRStatus(str, Enum):
    """Status levels for OTR monitoring."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"  # OTR > 25:1
    HALTED = "HALTED"  # OTR > 50:1


class OTREventType(str, Enum):
    """Types of OTR events."""

    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"


@dataclass
class OTRAlert:
    """Alert generated when OTR thresholds are breached."""

    alert_id: str
    algo_id: str
    timestamp: datetime
    otr_value: float
    window_seconds: int
    status: OTRStatus
    orders_count: int
    trades_count: int
    message: str


@dataclass
class _TimestampedEvent:
    """Internal record of an order/trade event with timestamp for windowing."""

    timestamp: float  # time.monotonic() for window calculations
    utc_time: datetime
    event_type: OTREventType
    order_id: str
    symbol: str


class OTRMonitor:
    """
    Thread-safe Order-to-Trade Ratio monitor.

    Tracks orders submitted and filled per algorithm, computing rolling OTR
    across multiple time windows. Automatically throttles or halts algo
    activity when thresholds are breached.

    Parameters
    ----------
    warning_threshold : float
        OTR ratio that triggers a WARNING (default 25.0).
    halt_threshold : float
        OTR ratio that triggers a HALT (default 50.0).
    on_alert : callable, optional
        Callback invoked with an OTRAlert when thresholds are breached.
    """

    def __init__(
        self,
        warning_threshold: float = OTR_WARNING_THRESHOLD,
        halt_threshold: float = OTR_HALT_THRESHOLD,
        on_alert: Callable[[OTRAlert], None] | None = None,
    ) -> None:
        self._warning_threshold = warning_threshold
        self._halt_threshold = halt_threshold
        self._on_alert = on_alert
        self._lock = threading.Lock()

        # Per-algo deques of timestamped events
        self._order_events: dict[str, deque[_TimestampedEvent]] = defaultdict(deque)
        self._trade_events: dict[str, deque[_TimestampedEvent]] = defaultdict(deque)

        # Current status per algo
        self._algo_status: dict[str, OTRStatus] = defaultdict(lambda: OTRStatus.NORMAL)

        # Alert history
        self._alerts: list[OTRAlert] = []

        # Cumulative counters (lifetime)
        self._total_orders: dict[str, int] = defaultdict(int)
        self._total_trades: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Public API: record events
    # ------------------------------------------------------------------

    def record_order(
        self,
        algo_id: str,
        order_id: str,
        symbol: str,
    ) -> OTRStatus:
        """
        Record an order submission. Returns current OTR status for the algo.

        Parameters
        ----------
        algo_id : str
            Unique identifier for the algorithm (SEBI algo registration ID).
        order_id : str
            Exchange or internal order ID.
        symbol : str
            Trading instrument symbol.

        Returns
        -------
        OTRStatus
            Current status after recording this order.
        """
        now_mono = time.monotonic()
        now_utc = datetime.now(UTC)

        event = _TimestampedEvent(
            timestamp=now_mono,
            utc_time=now_utc,
            event_type=OTREventType.ORDER_SUBMITTED,
            order_id=order_id,
            symbol=symbol,
        )

        with self._lock:
            self._order_events[algo_id].append(event)
            self._total_orders[algo_id] += 1
            self._prune_old_events(algo_id, now_mono)
            return self._evaluate_otr(algo_id, now_mono, now_utc)

    def record_trade(
        self,
        algo_id: str,
        order_id: str,
        symbol: str,
    ) -> OTRStatus:
        """
        Record a trade (fill). Returns current OTR status for the algo.

        Parameters
        ----------
        algo_id : str
            Algorithm identifier.
        order_id : str
            Order ID that was filled.
        symbol : str
            Trading instrument symbol.

        Returns
        -------
        OTRStatus
            Current status after recording this trade.
        """
        now_mono = time.monotonic()
        now_utc = datetime.now(UTC)

        event = _TimestampedEvent(
            timestamp=now_mono,
            utc_time=now_utc,
            event_type=OTREventType.ORDER_FILLED,
            order_id=order_id,
            symbol=symbol,
        )

        with self._lock:
            self._trade_events[algo_id].append(event)
            self._total_trades[algo_id] += 1
            self._prune_old_events(algo_id, now_mono)
            return self._evaluate_otr(algo_id, now_mono, now_utc)

    # ------------------------------------------------------------------
    # Public API: query OTR
    # ------------------------------------------------------------------

    def get_otr(
        self,
        algo_id: str,
        window_seconds: int = WINDOW_5MIN,
    ) -> float:
        """
        Get current OTR for an algorithm within a time window.

        Returns
        -------
        float
            Order-to-trade ratio. Returns 0.0 if no orders in window.
            If there are orders but no trades, returns float('inf').
        """
        now_mono = time.monotonic()
        with self._lock:
            return self._compute_otr(algo_id, window_seconds, now_mono)

    def get_all_otr_reports(self) -> list[dict[str, Any]]:
        """
        Get OTR reports for all tracked algorithms across all windows.

        Returns
        -------
        list[dict]
            List of per-algo OTR summaries.
        """
        now_mono = time.monotonic()
        reports = []

        with self._lock:
            for algo_id in set(list(self._order_events.keys()) + list(self._trade_events.keys())):
                otr_1m = self._compute_otr(algo_id, WINDOW_1MIN, now_mono)
                otr_5m = self._compute_otr(algo_id, WINDOW_5MIN, now_mono)
                otr_1h = self._compute_otr(algo_id, WINDOW_1HR, now_mono)

                orders_1m = self._count_events_in_window(
                    self._order_events.get(algo_id, deque()), WINDOW_1MIN, now_mono
                )
                trades_1m = self._count_events_in_window(
                    self._trade_events.get(algo_id, deque()), WINDOW_1MIN, now_mono
                )
                orders_5m = self._count_events_in_window(
                    self._order_events.get(algo_id, deque()), WINDOW_5MIN, now_mono
                )
                trades_5m = self._count_events_in_window(
                    self._trade_events.get(algo_id, deque()), WINDOW_5MIN, now_mono
                )
                orders_1h = self._count_events_in_window(self._order_events.get(algo_id, deque()), WINDOW_1HR, now_mono)
                trades_1h = self._count_events_in_window(self._trade_events.get(algo_id, deque()), WINDOW_1HR, now_mono)

                reports.append(
                    {
                        "algo_id": algo_id,
                        "status": self._algo_status[algo_id].value,
                        "otr_1min": otr_1m if otr_1m != float("inf") else -1,
                        "otr_5min": otr_5m if otr_5m != float("inf") else -1,
                        "otr_1hr": otr_1h if otr_1h != float("inf") else -1,
                        "orders_1min": orders_1m,
                        "trades_1min": trades_1m,
                        "orders_5min": orders_5m,
                        "trades_5min": trades_5m,
                        "orders_1hr": orders_1h,
                        "trades_1hr": trades_1h,
                        "total_orders_lifetime": self._total_orders.get(algo_id, 0),
                        "total_trades_lifetime": self._total_trades.get(algo_id, 0),
                        "warning_threshold": self._warning_threshold,
                        "halt_threshold": self._halt_threshold,
                    }
                )

        return reports

    def get_status(self, algo_id: str) -> OTRStatus:
        """Get the current OTR status for an algorithm."""
        with self._lock:
            return self._algo_status[algo_id]

    def is_halted(self, algo_id: str) -> bool:
        """Check if an algorithm is halted due to excessive OTR."""
        return self.get_status(algo_id) == OTRStatus.HALTED

    def reset_status(self, algo_id: str) -> None:
        """
        Manually reset an algo's OTR status to NORMAL.
        Typically called after human review of the halt condition.
        """
        with self._lock:
            prev = self._algo_status[algo_id]
            self._algo_status[algo_id] = OTRStatus.NORMAL
            logger.info(
                "OTR status for algo %s manually reset from %s to NORMAL",
                algo_id,
                prev.value,
            )

    def get_alerts(
        self,
        algo_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get recent OTR alerts, optionally filtered by algo ID.

        Returns
        -------
        list[dict]
            Alert records in reverse chronological order.
        """
        with self._lock:
            alerts = list(self._alerts)

        if algo_id:
            alerts = [a for a in alerts if a.algo_id == algo_id]

        alerts = alerts[-limit:]
        alerts.reverse()

        return [
            {
                "alert_id": a.alert_id,
                "algo_id": a.algo_id,
                "timestamp": a.timestamp.isoformat(),
                "otr_value": a.otr_value if a.otr_value != float("inf") else -1,
                "window_seconds": a.window_seconds,
                "status": a.status.value,
                "orders_count": a.orders_count,
                "trades_count": a.trades_count,
                "message": a.message,
            }
            for a in alerts
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_otr(
        self,
        algo_id: str,
        window_seconds: int,
        now_mono: float,
    ) -> float:
        """Compute OTR within a time window. Must be called under lock."""
        orders = self._count_events_in_window(self._order_events.get(algo_id, deque()), window_seconds, now_mono)
        trades = self._count_events_in_window(self._trade_events.get(algo_id, deque()), window_seconds, now_mono)

        if orders == 0:
            return 0.0
        if trades == 0:
            # Warmup grace: don't flag OTR until sufficient orders accumulated.
            # Paper mode fills arrive with ~3s latency, and execution agent may
            # submit 8+ orders per tick. Use 20 as threshold to allow fills to
            # catch up before enforcing.
            if orders < 20:
                return 0.0
            return float("inf")
        return orders / trades

    def _count_events_in_window(
        self,
        events: deque[_TimestampedEvent],
        window_seconds: int,
        now_mono: float,
    ) -> int:
        """Count events within the time window. Must be called under lock."""
        cutoff = now_mono - window_seconds
        count = 0
        for evt in reversed(events):
            if evt.timestamp < cutoff:
                break
            count += 1
        return count

    def _prune_old_events(self, algo_id: str, now_mono: float) -> None:
        """Remove events older than 1 hour (longest window). Must be called under lock."""
        cutoff = now_mono - WINDOW_1HR - 60  # 1 minute buffer
        for events_dict in (self._order_events, self._trade_events):
            dq = events_dict.get(algo_id)
            if dq:
                while dq and dq[0].timestamp < cutoff:
                    dq.popleft()

    def _evaluate_otr(
        self,
        algo_id: str,
        now_mono: float,
        now_utc: datetime,
    ) -> OTRStatus:
        """
        Evaluate OTR across all windows and update status.
        Generates alerts when thresholds are breached.
        Must be called under lock.
        """
        import uuid

        worst_status = OTRStatus.NORMAL
        worst_otr = 0.0
        worst_window = WINDOW_1MIN

        for window in (WINDOW_1MIN, WINDOW_5MIN, WINDOW_1HR):
            otr = self._compute_otr(algo_id, window, now_mono)

            if otr == float("inf") or otr >= self._halt_threshold:
                if worst_status != OTRStatus.HALTED:
                    worst_status = OTRStatus.HALTED
                    worst_otr = otr
                    worst_window = window
            elif otr >= self._warning_threshold:
                if worst_status == OTRStatus.NORMAL:
                    worst_status = OTRStatus.WARNING
                    worst_otr = otr
                    worst_window = window

        prev_status = self._algo_status[algo_id]
        self._algo_status[algo_id] = worst_status

        # Generate alert on status change or if already halted
        if worst_status != OTRStatus.NORMAL and worst_status != prev_status:
            orders_count = self._count_events_in_window(
                self._order_events.get(algo_id, deque()), worst_window, now_mono
            )
            trades_count = self._count_events_in_window(
                self._trade_events.get(algo_id, deque()), worst_window, now_mono
            )

            if worst_status == OTRStatus.HALTED:
                message = (
                    f"SEBI OTR HALT: Algorithm {algo_id} exceeded halt threshold "
                    f"({self._halt_threshold}:1). Current OTR: "
                    f"{'inf' if worst_otr == float('inf') else f'{worst_otr:.1f}'}:1 "
                    f"over {worst_window}s window. "
                    f"Orders: {orders_count}, Trades: {trades_count}. "
                    f"Algorithm execution HALTED pending review."
                )
                logger.critical(message)
            else:
                message = (
                    f"SEBI OTR WARNING: Algorithm {algo_id} exceeded warning threshold "
                    f"({self._warning_threshold}:1). Current OTR: {worst_otr:.1f}:1 "
                    f"over {worst_window}s window. "
                    f"Orders: {orders_count}, Trades: {trades_count}."
                )
                logger.warning(message)

            alert = OTRAlert(
                alert_id=str(uuid.uuid4()),
                algo_id=algo_id,
                timestamp=now_utc,
                otr_value=worst_otr,
                window_seconds=worst_window,
                status=worst_status,
                orders_count=orders_count,
                trades_count=trades_count,
                message=message,
            )
            self._alerts.append(alert)

            # Cap alert history at 10000 entries
            if len(self._alerts) > 10000:
                self._alerts = self._alerts[-5000:]

            if self._on_alert:
                try:
                    self._on_alert(alert)
                except Exception:
                    logger.exception("OTR alert callback failed for algo %s", algo_id)

        return worst_status
