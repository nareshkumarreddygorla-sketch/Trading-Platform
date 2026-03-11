"""
Daily broker position reconciliation.

Compares local position state (RiskManager) against broker's reported positions
(Angel One API) and flags discrepancies for manual review.

SEBI compliance: algorithmic trading systems must reconcile positions daily.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PositionDiscrepancy:
    """A single discrepancy between local and broker position."""

    symbol: str
    exchange: str
    local_qty: float
    broker_qty: float
    local_avg_price: float
    broker_avg_price: float
    discrepancy_type: str  # "missing_local", "missing_broker", "qty_mismatch", "price_mismatch"
    notional_impact: float  # Estimated INR impact
    severity: str  # "low", "medium", "high", "critical"

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "local_qty": self.local_qty,
            "broker_qty": self.broker_qty,
            "local_avg_price": round(self.local_avg_price, 2),
            "broker_avg_price": round(self.broker_avg_price, 2),
            "type": self.discrepancy_type,
            "notional_impact": round(self.notional_impact, 2),
            "severity": self.severity,
        }


@dataclass
class ReconciliationReport:
    """Full reconciliation report."""

    timestamp: str
    local_position_count: int
    broker_position_count: int
    matched: int
    discrepancies: list[PositionDiscrepancy] = field(default_factory=list)
    total_notional_impact: float = 0.0
    status: str = "ok"  # "ok", "warning", "critical"
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "local_positions": self.local_position_count,
            "broker_positions": self.broker_position_count,
            "matched": self.matched,
            "discrepancy_count": len(self.discrepancies),
            "total_notional_impact": round(self.total_notional_impact, 2),
            "status": self.status,
            "error": self.error,
            "discrepancies": [d.as_dict() for d in self.discrepancies],
        }


class BrokerReconciliator:
    """
    Reconciles local positions (from RiskManager) against broker positions.

    Usage:
        reconciliator = BrokerReconciliator(
            get_local_positions=lambda: risk_manager.positions,
            get_broker_positions=angel_one_client.get_positions,
        )
        report = await reconciliator.reconcile()
        if report.status == "critical":
            halt_trading()
    """

    def __init__(
        self,
        get_local_positions=None,
        get_broker_positions=None,
        price_tolerance_pct: float = 1.0,  # 1% price tolerance
        qty_tolerance: float = 0.0,  # Exact qty match required
        alert_threshold_inr: float = 10_000,  # ₹10K notional discrepancy triggers alert
        critical_threshold_inr: float = 100_000,  # ₹1L triggers critical
        on_discrepancy=None,
        halt_trading_fn=None,
    ):
        self._get_local = get_local_positions
        self._get_broker = get_broker_positions
        self._price_tolerance = price_tolerance_pct
        self._qty_tolerance = qty_tolerance
        self._alert_threshold = alert_threshold_inr
        self._critical_threshold = critical_threshold_inr
        self._on_discrepancy = on_discrepancy
        self._halt_trading_fn = halt_trading_fn
        self._periodic_task = None
        self._history: list[ReconciliationReport] = []
        self._history_lock = asyncio.Lock()

    def _normalize_positions(self, positions: list[Any]) -> dict[str, dict[str, Any]]:
        """Normalize positions into a comparable dict keyed by symbol+exchange+side."""
        normalized = {}
        for p in positions:
            symbol = (
                getattr(p, "symbol", None) or p.get("symbol", "") if isinstance(p, dict) else getattr(p, "symbol", "")
            )
            exchange = getattr(p, "exchange", None)
            if hasattr(exchange, "value"):
                exchange = exchange.value
            elif isinstance(p, dict):
                exchange = p.get("exchange", "NSE")
            else:
                exchange = str(exchange or "NSE")

            side = getattr(p, "side", None)
            if hasattr(side, "value"):
                side = side.value
            elif isinstance(p, dict):
                side = p.get("side")
                if not side:
                    logger.warning("Position missing 'side' field: %s — skipping", p)
                    continue
            else:
                side = str(side) if side else None
                if not side:
                    logger.warning("Position missing 'side' field: %s — skipping", p)
                    continue

            qty = float(getattr(p, "quantity", 0) if not isinstance(p, dict) else p.get("quantity", 0))
            avg_price = float(getattr(p, "avg_price", 0) if not isinstance(p, dict) else p.get("avg_price", 0))

            key = f"{symbol}|{exchange}|{side}"
            if key in normalized:
                # Merge: sum quantities, VWAP prices
                existing = normalized[key]
                total_qty = existing["qty"] + qty
                if total_qty > 0:
                    existing["avg_price"] = (existing["qty"] * existing["avg_price"] + qty * avg_price) / total_qty
                existing["qty"] = total_qty
            else:
                normalized[key] = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "side": side,
                    "qty": qty,
                    "avg_price": avg_price,
                }
        return normalized

    async def reconcile(self) -> ReconciliationReport:
        """Run full position reconciliation between local and broker."""
        timestamp = datetime.now(UTC).isoformat()

        # Get positions from both sources
        try:
            local_raw = self._get_local() if self._get_local else []
            if hasattr(local_raw, "__await__"):
                local_raw = await asyncio.wait_for(local_raw, timeout=30.0)
            if not isinstance(local_raw, list):
                local_raw = list(local_raw)
        except Exception as e:
            return ReconciliationReport(
                timestamp=timestamp,
                local_position_count=0,
                broker_position_count=0,
                matched=0,
                status="critical",
                error=f"Failed to get local positions: {e}",
            )

        try:
            broker_raw = self._get_broker() if self._get_broker else []
            if hasattr(broker_raw, "__await__"):
                try:
                    broker_raw = await asyncio.wait_for(broker_raw, timeout=30.0)
                except TimeoutError:
                    logger.error("Broker position fetch timed out (30s)")
                    return ReconciliationReport(
                        timestamp=timestamp,
                        local_position_count=len(local_raw),
                        broker_position_count=0,
                        matched=0,
                        status="critical",
                        error="Broker position fetch timed out (30s)",
                    )
            if not isinstance(broker_raw, list):
                broker_raw = list(broker_raw)
        except TimeoutError:
            raise  # already handled above
        except Exception as e:
            return ReconciliationReport(
                timestamp=timestamp,
                local_position_count=len(local_raw),
                broker_position_count=0,
                matched=0,
                status="critical",
                error=f"Failed to get broker positions: {e}",
            )

        local_positions = self._normalize_positions(local_raw)
        broker_positions = self._normalize_positions(broker_raw)

        discrepancies: list[PositionDiscrepancy] = []
        matched = 0

        # Check all local positions against broker
        all_keys = set(list(local_positions.keys()) + list(broker_positions.keys()))

        for key in all_keys:
            local = local_positions.get(key)
            broker = broker_positions.get(key)

            if local and not broker:
                # Position exists locally but not at broker
                notional = local["qty"] * local["avg_price"]
                discrepancies.append(
                    PositionDiscrepancy(
                        symbol=local["symbol"],
                        exchange=local["exchange"],
                        local_qty=local["qty"],
                        broker_qty=0,
                        local_avg_price=local["avg_price"],
                        broker_avg_price=0,
                        discrepancy_type="missing_broker",
                        notional_impact=notional,
                        severity="critical" if notional > self._critical_threshold else "high",
                    )
                )
            elif broker and not local:
                # Position exists at broker but not locally (phantom position)
                notional = broker["qty"] * broker["avg_price"]
                discrepancies.append(
                    PositionDiscrepancy(
                        symbol=broker["symbol"],
                        exchange=broker["exchange"],
                        local_qty=0,
                        broker_qty=broker["qty"],
                        local_avg_price=0,
                        broker_avg_price=broker["avg_price"],
                        discrepancy_type="missing_local",
                        notional_impact=notional,
                        severity="critical" if notional > self._critical_threshold else "high",
                    )
                )
            else:
                # Both exist — check for qty/price mismatch
                qty_diff = abs(local["qty"] - broker["qty"])
                price_diff_pct = abs(local["avg_price"] - broker["avg_price"]) / max(local["avg_price"], 0.01) * 100

                if qty_diff > self._qty_tolerance:
                    notional = qty_diff * max(local["avg_price"], broker["avg_price"])
                    severity = (
                        "critical"
                        if notional > self._critical_threshold
                        else ("high" if notional > self._alert_threshold else "medium")
                    )
                    discrepancies.append(
                        PositionDiscrepancy(
                            symbol=local["symbol"],
                            exchange=local["exchange"],
                            local_qty=local["qty"],
                            broker_qty=broker["qty"],
                            local_avg_price=local["avg_price"],
                            broker_avg_price=broker["avg_price"],
                            discrepancy_type="qty_mismatch",
                            notional_impact=notional,
                            severity=severity,
                        )
                    )
                elif price_diff_pct > self._price_tolerance:
                    notional = local["qty"] * abs(local["avg_price"] - broker["avg_price"])
                    discrepancies.append(
                        PositionDiscrepancy(
                            symbol=local["symbol"],
                            exchange=local["exchange"],
                            local_qty=local["qty"],
                            broker_qty=broker["qty"],
                            local_avg_price=local["avg_price"],
                            broker_avg_price=broker["avg_price"],
                            discrepancy_type="price_mismatch",
                            notional_impact=notional,
                            severity="low",
                        )
                    )
                else:
                    matched += 1

        total_impact = sum(d.notional_impact for d in discrepancies)
        has_critical = any(d.severity == "critical" for d in discrepancies)
        has_high = any(d.severity == "high" for d in discrepancies)

        status = "ok"
        if has_critical or total_impact > self._critical_threshold:
            status = "critical"
        elif has_high or total_impact > self._alert_threshold:
            status = "warning"

        report = ReconciliationReport(
            timestamp=timestamp,
            local_position_count=len(local_positions),
            broker_position_count=len(broker_positions),
            matched=matched,
            discrepancies=discrepancies,
            total_notional_impact=total_impact,
            status=status,
        )

        async with self._history_lock:
            self._history.append(report)
            self._history = self._history[-30:]

        # Log and alert
        if status == "critical":
            logger.critical(
                "RECONCILIATION CRITICAL: %d discrepancies, total impact ₹%.0f — HALT TRADING",
                len(discrepancies),
                total_impact,
            )
            if report.status == "critical" and self._halt_trading_fn:
                try:
                    self._halt_trading_fn("RECONCILIATION_CRITICAL")
                    logger.critical("Trading halted due to critical reconciliation failure")
                except Exception as e:
                    logger.error("Failed to halt trading after critical reconciliation: %s", e)
        elif status == "warning":
            logger.warning(
                "Reconciliation warning: %d discrepancies, total impact ₹%.0f",
                len(discrepancies),
                total_impact,
            )
        else:
            logger.info(
                "Reconciliation OK: %d matched, %d discrepancies",
                matched,
                len(discrepancies),
            )

        if discrepancies and self._on_discrepancy:
            try:
                result = self._on_discrepancy(report)
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                logger.error("Reconciliation alert callback failed: %s", e)

        return report

    async def get_history(self) -> list[dict]:
        """Return reconciliation history."""
        async with self._history_lock:
            return [r.as_dict() for r in self._history]

    async def start_periodic(self, interval_hours: float = 4.0):
        """Start periodic reconciliation task."""
        import asyncio

        self._periodic_task = asyncio.create_task(self._periodic_loop(interval_hours))
        logger.info("Periodic reconciliation started (every %.1f hours)", interval_hours)

    async def stop_periodic(self):
        """Stop periodic reconciliation task."""
        if hasattr(self, "_periodic_task") and self._periodic_task and not self._periodic_task.done():
            self._periodic_task.cancel()
            try:
                await self._periodic_task
            except asyncio.CancelledError:
                pass
            self._periodic_task = None
            logger.info("Periodic reconciliation stopped")

    async def _periodic_loop(self, interval_hours: float):
        import asyncio

        while True:
            try:
                report = await self.reconcile()
                logger.info(
                    "Periodic reconciliation completed: status=%s, discrepancies=%d",
                    report.status,
                    len(report.discrepancies),
                )
                await asyncio.sleep(interval_hours * 3600)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Periodic reconciliation failed: %s", e)
                await asyncio.sleep(60)
