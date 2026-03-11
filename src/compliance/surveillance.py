"""
Post-Trade Surveillance Module for SEBI Compliance.

Detects potential market manipulation patterns in trading activity:
  - Layering: Multiple orders at different price levels rapidly cancelled
    to create an artificial impression of supply/demand.
  - Spoofing: Large orders placed and cancelled before execution to
    mislead other market participants.
  - Wash Trades: Self-matching trades where the same entity is on both
    sides (buy and sell) to inflate volume.

Each detection generates an alert with a full evidence trail suitable
for regulatory review.

Key features:
  - Configurable detection thresholds per pattern type.
  - Alert generation with evidence trail.
  - Integration hooks for the execution engine.
  - Thread-safe implementation.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------


class ManipulationPattern(str, Enum):
    """Types of market manipulation patterns detected."""

    LAYERING = "LAYERING"
    SPOOFING = "SPOOFING"
    WASH_TRADE = "WASH_TRADE"


class AlertSeverity(str, Enum):
    """Severity levels for surveillance alerts."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class SurveillanceAlert:
    """
    Alert generated when a potential manipulation pattern is detected.

    Fields
    ------
    alert_id : str
        Unique identifier for the alert.
    pattern : ManipulationPattern
        Type of manipulation pattern detected.
    severity : AlertSeverity
        Alert severity level.
    timestamp : datetime
        UTC timestamp when the alert was generated.
    algo_id : str
        Algorithm or trader ID involved.
    symbol : str
        Instrument symbol involved.
    description : str
        Human-readable description of the detected pattern.
    evidence : dict
        Detailed evidence supporting the alert (order IDs, timestamps,
        prices, quantities, etc.).
    acknowledged : bool
        Whether a compliance officer has reviewed this alert.
    """

    alert_id: str
    pattern: ManipulationPattern
    severity: AlertSeverity
    timestamp: datetime
    algo_id: str
    symbol: str
    description: str
    evidence: dict[str, Any]
    acknowledged: bool = False


@dataclass
class _OrderRecord:
    """Internal record of an order for surveillance analysis."""

    order_id: str
    algo_id: str
    symbol: str
    side: str  # BUY or SELL
    price: float
    quantity: float
    timestamp: float  # monotonic
    utc_time: datetime
    status: str  # SUBMITTED, FILLED, CANCELLED
    cancel_time: float | None = None  # monotonic time of cancellation


@dataclass
class _TradeRecord:
    """Internal record of a trade for surveillance analysis."""

    trade_id: str
    order_id: str
    algo_id: str
    symbol: str
    side: str
    price: float
    quantity: float
    timestamp: float  # monotonic
    utc_time: datetime
    counterparty_id: str | None = None


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------


@dataclass
class SurveillanceThresholds:
    """
    Configurable thresholds for manipulation pattern detection.

    Layering thresholds:
        layering_min_levels: Minimum number of price levels with orders
            to consider as potential layering.
        layering_cancel_ratio: Minimum fraction of orders cancelled
            within the time window.
        layering_time_window_sec: Time window for evaluating layering.

    Spoofing thresholds:
        spoofing_size_multiple: Minimum ratio of order size to average
            order size to flag as potentially spoofing.
        spoofing_cancel_time_sec: Maximum seconds between placement
            and cancellation to flag.
        spoofing_min_quantity: Minimum order quantity to consider.

    Wash trade thresholds:
        wash_price_tolerance: Maximum price difference (as fraction)
            between buy and sell to consider self-match.
        wash_time_window_sec: Time window for matching buy/sell pairs.
        wash_quantity_tolerance: Maximum quantity difference (as fraction)
            between buy and sell.
    """

    # Layering
    layering_min_levels: int = 3
    layering_cancel_ratio: float = 0.7
    layering_time_window_sec: float = 30.0

    # Spoofing
    spoofing_size_multiple: float = 5.0
    spoofing_cancel_time_sec: float = 5.0
    spoofing_min_quantity: float = 100.0

    # Wash trades
    wash_price_tolerance: float = 0.001  # 0.1%
    wash_time_window_sec: float = 60.0
    wash_quantity_tolerance: float = 0.05  # 5%


# ---------------------------------------------------------------------------
# Main surveillance engine
# ---------------------------------------------------------------------------


class SurveillanceEngine:
    """
    Post-trade surveillance engine for detecting market manipulation patterns.

    Thread-safe. Maintains a rolling window of orders and trades for each
    symbol, and runs detection algorithms on each new event.

    Parameters
    ----------
    thresholds : SurveillanceThresholds, optional
        Detection thresholds (uses defaults if not provided).
    on_alert : callable, optional
        Callback invoked with a SurveillanceAlert when a pattern is detected.
    max_history : int
        Maximum number of orders/trades to retain per symbol (default 10000).
    """

    def __init__(
        self,
        thresholds: SurveillanceThresholds | None = None,
        on_alert: Callable[[SurveillanceAlert], None] | None = None,
        max_history: int = 10000,
    ) -> None:
        self._thresholds = thresholds or SurveillanceThresholds()
        self._on_alert = on_alert
        self._max_history = max_history
        self._lock = threading.Lock()

        # Per-symbol order history
        self._orders: dict[str, deque[_OrderRecord]] = defaultdict(deque)
        # Per-symbol trade history
        self._trades: dict[str, deque[_TradeRecord]] = defaultdict(deque)
        # Order ID -> order record lookup
        self._order_index: dict[str, _OrderRecord] = {}

        # Alert storage
        self._alerts: list[SurveillanceAlert] = []

        # Track algo IDs for wash trade detection
        self._algo_trades: dict[str, deque[_TradeRecord]] = defaultdict(deque)

    # ------------------------------------------------------------------
    # Public API: record events
    # ------------------------------------------------------------------

    def record_order(
        self,
        order_id: str,
        algo_id: str,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
    ) -> list[SurveillanceAlert]:
        """
        Record a new order submission. Runs layering detection.

        Returns
        -------
        list[SurveillanceAlert]
            Any alerts generated by this event.
        """
        now_mono = time.monotonic()
        now_utc = datetime.now(UTC)

        record = _OrderRecord(
            order_id=order_id,
            algo_id=algo_id,
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            timestamp=now_mono,
            utc_time=now_utc,
            status="SUBMITTED",
        )

        alerts = []
        with self._lock:
            self._orders[symbol].append(record)
            self._order_index[order_id] = record
            self._prune_orders(symbol)

            # Run layering detection after new order
            layering_alert = self._detect_layering(symbol, algo_id, now_mono, now_utc)
            if layering_alert:
                alerts.append(layering_alert)

        return alerts

    def record_cancellation(
        self,
        order_id: str,
    ) -> list[SurveillanceAlert]:
        """
        Record an order cancellation. Runs spoofing detection.

        Returns
        -------
        list[SurveillanceAlert]
            Any alerts generated by this event.
        """
        now_mono = time.monotonic()
        now_utc = datetime.now(UTC)

        alerts = []
        with self._lock:
            record = self._order_index.get(order_id)
            if record:
                record.status = "CANCELLED"
                record.cancel_time = now_mono

                # Run spoofing detection after cancellation
                spoofing_alert = self._detect_spoofing(record, now_mono, now_utc)
                if spoofing_alert:
                    alerts.append(spoofing_alert)

        return alerts

    def record_trade(
        self,
        trade_id: str,
        order_id: str,
        algo_id: str,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        counterparty_id: str | None = None,
    ) -> list[SurveillanceAlert]:
        """
        Record a trade execution. Runs wash trade detection.

        Returns
        -------
        list[SurveillanceAlert]
            Any alerts generated by this event.
        """
        now_mono = time.monotonic()
        now_utc = datetime.now(UTC)

        record = _TradeRecord(
            trade_id=trade_id,
            order_id=order_id,
            algo_id=algo_id,
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            timestamp=now_mono,
            utc_time=now_utc,
            counterparty_id=counterparty_id,
        )

        alerts = []
        with self._lock:
            self._trades[symbol].append(record)
            self._algo_trades[algo_id].append(record)
            self._prune_trades(symbol)

            # Mark order as filled
            order = self._order_index.get(order_id)
            if order:
                order.status = "FILLED"

            # Run wash trade detection
            wash_alert = self._detect_wash_trade(record, now_mono, now_utc)
            if wash_alert:
                alerts.append(wash_alert)

        return alerts

    # ------------------------------------------------------------------
    # Public API: query alerts
    # ------------------------------------------------------------------

    def get_alerts(
        self,
        pattern: ManipulationPattern | None = None,
        severity: AlertSeverity | None = None,
        symbol: str | None = None,
        limit: int = 100,
        unacknowledged_only: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Get recent surveillance alerts with optional filters.

        Returns
        -------
        list[dict]
            Alert records in reverse chronological order.
        """
        with self._lock:
            alerts = list(self._alerts)

        if pattern:
            alerts = [a for a in alerts if a.pattern == pattern]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        alerts = alerts[-limit:]
        alerts.reverse()

        return [
            {
                "alert_id": a.alert_id,
                "pattern": a.pattern.value,
                "severity": a.severity.value,
                "timestamp": a.timestamp.isoformat(),
                "algo_id": a.algo_id,
                "symbol": a.symbol,
                "description": a.description,
                "evidence": a.evidence,
                "acknowledged": a.acknowledged,
            }
            for a in alerts
        ]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Mark an alert as acknowledged by compliance officer.

        Returns True if the alert was found and acknowledged.
        """
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    logger.info("Surveillance alert %s acknowledged", alert_id)
                    return True
        return False

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of surveillance activity.

        Returns
        -------
        dict
            Summary with counts by pattern and severity.
        """
        with self._lock:
            alerts = list(self._alerts)

        total = len(alerts)
        unack = sum(1 for a in alerts if not a.acknowledged)
        by_pattern = defaultdict(int)
        by_severity = defaultdict(int)
        for a in alerts:
            by_pattern[a.pattern.value] += 1
            by_severity[a.severity.value] += 1

        return {
            "total_alerts": total,
            "unacknowledged": unack,
            "by_pattern": dict(by_pattern),
            "by_severity": dict(by_severity),
            "thresholds": {
                "layering_min_levels": self._thresholds.layering_min_levels,
                "layering_cancel_ratio": self._thresholds.layering_cancel_ratio,
                "spoofing_size_multiple": self._thresholds.spoofing_size_multiple,
                "spoofing_cancel_time_sec": self._thresholds.spoofing_cancel_time_sec,
                "wash_price_tolerance": self._thresholds.wash_price_tolerance,
                "wash_time_window_sec": self._thresholds.wash_time_window_sec,
            },
        }

    # ------------------------------------------------------------------
    # Detection algorithms
    # ------------------------------------------------------------------

    def _detect_layering(
        self,
        symbol: str,
        algo_id: str,
        now_mono: float,
        now_utc: datetime,
    ) -> SurveillanceAlert | None:
        """
        Detect layering: multiple orders at different price levels by the same
        algo on the same side, with a high cancellation rate.

        Must be called under lock.
        """
        th = self._thresholds
        cutoff = now_mono - th.layering_time_window_sec

        # Get recent orders from this algo on this symbol
        recent_orders = [o for o in self._orders[symbol] if o.algo_id == algo_id and o.timestamp >= cutoff]

        if len(recent_orders) < th.layering_min_levels:
            return None

        # Check each side separately
        for side in ("BUY", "SELL"):
            side_orders = [o for o in recent_orders if o.side == side]
            if len(side_orders) < th.layering_min_levels:
                continue

            # Check for multiple distinct price levels
            price_levels = set(o.price for o in side_orders)
            if len(price_levels) < th.layering_min_levels:
                continue

            # Check cancellation ratio
            cancelled = sum(1 for o in side_orders if o.status == "CANCELLED")
            cancel_ratio = cancelled / len(side_orders) if side_orders else 0

            if cancel_ratio >= th.layering_cancel_ratio:
                evidence = {
                    "side": side,
                    "total_orders": len(side_orders),
                    "price_levels": sorted(list(price_levels)),
                    "cancelled_count": cancelled,
                    "cancel_ratio": round(cancel_ratio, 3),
                    "time_window_sec": th.layering_time_window_sec,
                    "order_ids": [o.order_id for o in side_orders],
                    "order_details": [
                        {
                            "order_id": o.order_id,
                            "price": o.price,
                            "quantity": o.quantity,
                            "status": o.status,
                            "timestamp": o.utc_time.isoformat(),
                        }
                        for o in side_orders
                    ],
                }

                severity = (
                    AlertSeverity.CRITICAL
                    if cancel_ratio > 0.9
                    else AlertSeverity.HIGH
                    if cancel_ratio > 0.8
                    else AlertSeverity.MEDIUM
                )

                alert = SurveillanceAlert(
                    alert_id=str(uuid.uuid4()),
                    pattern=ManipulationPattern.LAYERING,
                    severity=severity,
                    timestamp=now_utc,
                    algo_id=algo_id,
                    symbol=symbol,
                    description=(
                        f"Potential LAYERING detected: {len(side_orders)} {side} orders "
                        f"across {len(price_levels)} price levels with "
                        f"{cancel_ratio:.0%} cancellation rate within "
                        f"{th.layering_time_window_sec}s window."
                    ),
                    evidence=evidence,
                )
                self._emit_alert(alert)
                return alert

        return None

    def _detect_spoofing(
        self,
        cancelled_order: _OrderRecord,
        now_mono: float,
        now_utc: datetime,
    ) -> SurveillanceAlert | None:
        """
        Detect spoofing: large order placed and cancelled quickly before execution.

        Must be called under lock.
        """
        th = self._thresholds

        if cancelled_order.quantity < th.spoofing_min_quantity:
            return None

        cancel_elapsed = (cancelled_order.cancel_time or now_mono) - cancelled_order.timestamp

        if cancel_elapsed > th.spoofing_cancel_time_sec:
            return None

        # Compute average order size for this symbol to check size multiple
        symbol = cancelled_order.symbol
        recent_orders = [
            o
            for o in self._orders[symbol]
            if o.order_id != cancelled_order.order_id and o.timestamp >= (now_mono - 3600)  # last hour
        ]

        if recent_orders:
            avg_qty = sum(o.quantity for o in recent_orders) / len(recent_orders)
            size_ratio = cancelled_order.quantity / avg_qty if avg_qty > 0 else 0
        else:
            # No other orders to compare; use absolute threshold
            size_ratio = th.spoofing_size_multiple  # flag by default

        if size_ratio < th.spoofing_size_multiple:
            return None

        evidence = {
            "order_id": cancelled_order.order_id,
            "side": cancelled_order.side,
            "price": cancelled_order.price,
            "quantity": cancelled_order.quantity,
            "time_to_cancel_sec": round(cancel_elapsed, 3),
            "size_ratio_to_avg": round(size_ratio, 2),
            "avg_order_size": round(avg_qty, 2) if recent_orders else None,
            "order_timestamp": cancelled_order.utc_time.isoformat(),
        }

        severity = (
            AlertSeverity.CRITICAL
            if cancel_elapsed < 1.0 and size_ratio > 10
            else AlertSeverity.HIGH
            if cancel_elapsed < 2.0
            else AlertSeverity.MEDIUM
        )

        alert = SurveillanceAlert(
            alert_id=str(uuid.uuid4()),
            pattern=ManipulationPattern.SPOOFING,
            severity=severity,
            timestamp=now_utc,
            algo_id=cancelled_order.algo_id,
            symbol=cancelled_order.symbol,
            description=(
                f"Potential SPOOFING detected: {cancelled_order.side} order for "
                f"{cancelled_order.quantity} units at {cancelled_order.price} "
                f"cancelled after {cancel_elapsed:.1f}s. "
                f"Size is {size_ratio:.1f}x average order size."
            ),
            evidence=evidence,
        )
        self._emit_alert(alert)
        return alert

    def _detect_wash_trade(
        self,
        new_trade: _TradeRecord,
        now_mono: float,
        now_utc: datetime,
    ) -> SurveillanceAlert | None:
        """
        Detect wash trades: same entity trading on both sides of the market
        at similar prices and quantities (self-matching).

        Must be called under lock.
        """
        th = self._thresholds
        cutoff = now_mono - th.wash_time_window_sec
        opposite_side = "SELL" if new_trade.side == "BUY" else "BUY"

        # Look for matching trades from the same algo on the opposite side
        algo_recent = [
            t
            for t in self._algo_trades.get(new_trade.algo_id, deque())
            if (
                t.trade_id != new_trade.trade_id
                and t.symbol == new_trade.symbol
                and t.side == opposite_side
                and t.timestamp >= cutoff
            )
        ]

        for matching in algo_recent:
            # Check price similarity
            price_diff = abs(new_trade.price - matching.price)
            price_tolerance = new_trade.price * th.wash_price_tolerance
            if price_diff > price_tolerance:
                continue

            # Check quantity similarity
            qty_diff = abs(new_trade.quantity - matching.quantity)
            qty_tolerance = new_trade.quantity * th.wash_quantity_tolerance
            if qty_diff > qty_tolerance:
                continue

            # Potential wash trade found
            time_between = abs(new_trade.timestamp - matching.timestamp)

            evidence = {
                "trade_1": {
                    "trade_id": new_trade.trade_id,
                    "order_id": new_trade.order_id,
                    "side": new_trade.side,
                    "price": new_trade.price,
                    "quantity": new_trade.quantity,
                    "timestamp": new_trade.utc_time.isoformat(),
                },
                "trade_2": {
                    "trade_id": matching.trade_id,
                    "order_id": matching.order_id,
                    "side": matching.side,
                    "price": matching.price,
                    "quantity": matching.quantity,
                    "timestamp": matching.utc_time.isoformat(),
                },
                "price_difference": round(price_diff, 4),
                "quantity_difference": round(qty_diff, 4),
                "time_between_sec": round(time_between, 3),
                "same_algo": True,
            }

            severity = (
                AlertSeverity.CRITICAL
                if time_between < 5
                else AlertSeverity.HIGH
                if time_between < 30
                else AlertSeverity.MEDIUM
            )

            alert = SurveillanceAlert(
                alert_id=str(uuid.uuid4()),
                pattern=ManipulationPattern.WASH_TRADE,
                severity=severity,
                timestamp=now_utc,
                algo_id=new_trade.algo_id,
                symbol=new_trade.symbol,
                description=(
                    f"Potential WASH TRADE detected: Same algo ({new_trade.algo_id}) "
                    f"executed {new_trade.side} and {matching.side} on {new_trade.symbol} "
                    f"within {time_between:.1f}s at similar prices "
                    f"({new_trade.price} vs {matching.price}) and quantities "
                    f"({new_trade.quantity} vs {matching.quantity})."
                ),
                evidence=evidence,
            )
            self._emit_alert(alert)
            return alert

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit_alert(self, alert: SurveillanceAlert) -> None:
        """Store alert and invoke callback. Must be called under lock."""
        self._alerts.append(alert)

        # Cap alert history
        if len(self._alerts) > 10000:
            self._alerts = self._alerts[-5000:]

        logger.warning(
            "Surveillance alert [%s] %s: %s",
            alert.pattern.value,
            alert.severity.value,
            alert.description,
        )

        if self._on_alert:
            try:
                self._on_alert(alert)
            except Exception:
                logger.exception("Surveillance alert callback failed")

    def _prune_orders(self, symbol: str) -> None:
        """Trim order history to max_history. Must be called under lock."""
        dq = self._orders.get(symbol)
        if dq and len(dq) > self._max_history:
            while len(dq) > self._max_history:
                removed = dq.popleft()
                self._order_index.pop(removed.order_id, None)

    def _prune_trades(self, symbol: str) -> None:
        """Trim trade history to max_history. Must be called under lock."""
        dq = self._trades.get(symbol)
        if dq and len(dq) > self._max_history:
            while len(dq) > self._max_history:
                dq.popleft()
