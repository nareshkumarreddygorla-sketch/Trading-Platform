"""
Gap risk protection module.

Pre-market gap detection and overnight exposure management:
  - Fetches pre-market indicators (SGX Nifty / Nifty futures)
  - Reduces positions if gap > 3%
  - Closes all positions if gap > 5%
  - Enforces overnight exposure limits
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time, timezone, timedelta
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)

IST_OFFSET = timedelta(hours=5, minutes=30)


class GapSeverity(str, Enum):
    NORMAL = "normal"         # gap < 1%
    MODERATE = "moderate"     # 1% <= gap < 3%
    HIGH = "high"             # 3% <= gap < 5%
    SEVERE = "severe"         # gap >= 5%


@dataclass
class GapAssessment:
    """Pre-market gap assessment result."""
    expected_gap_pct: float = 0.0
    severity: GapSeverity = GapSeverity.NORMAL
    action: str = "none"         # "none", "reduce_50", "close_all"
    reduce_to_pct: float = 100.0  # target exposure as % of current
    source: str = "unknown"


@dataclass
class OvernightExposureCheck:
    """Overnight exposure limit check result."""
    current_exposure_pct: float = 0.0
    max_exposure_pct: float = 50.0
    allowed: bool = True
    excess_notional: float = 0.0


class GapRiskManager:
    """
    Manages gap risk by monitoring pre-market indicators and
    enforcing overnight position limits.
    """

    def __init__(
        self,
        moderate_gap_pct: float = 3.0,
        severe_gap_pct: float = 5.0,
        max_overnight_exposure_pct: float = 50.0,
        pre_market_start: time = time(8, 45),   # 8:45 IST
        pre_market_end: time = time(9, 15),      # 9:15 IST
    ):
        self.moderate_gap_pct = moderate_gap_pct
        self.severe_gap_pct = severe_gap_pct
        self.max_overnight_exposure_pct = max_overnight_exposure_pct
        self.pre_market_start = pre_market_start
        self.pre_market_end = pre_market_end
        self._last_close_nifty: Optional[float] = None

    def set_previous_close(self, nifty_close: float) -> None:
        """Set previous session Nifty close for gap calculation."""
        self._last_close_nifty = nifty_close

    def assess_gap(self, pre_market_nifty: float) -> GapAssessment:
        """
        Assess gap severity based on pre-market Nifty indication.

        Args:
            pre_market_nifty: SGX Nifty or pre-open Nifty value

        Returns:
            GapAssessment with severity and recommended action
        """
        if self._last_close_nifty is None or self._last_close_nifty <= 0:
            return GapAssessment(source="no_reference")

        gap_pct = ((pre_market_nifty - self._last_close_nifty) / self._last_close_nifty) * 100

        if abs(gap_pct) >= self.severe_gap_pct:
            return GapAssessment(
                expected_gap_pct=gap_pct,
                severity=GapSeverity.SEVERE,
                action="close_all",
                reduce_to_pct=0.0,
                source="sgx_nifty",
            )
        elif abs(gap_pct) >= self.moderate_gap_pct:
            return GapAssessment(
                expected_gap_pct=gap_pct,
                severity=GapSeverity.HIGH,
                action="reduce_50",
                reduce_to_pct=50.0,
                source="sgx_nifty",
            )
        elif abs(gap_pct) >= 1.0:
            return GapAssessment(
                expected_gap_pct=gap_pct,
                severity=GapSeverity.MODERATE,
                action="none",
                reduce_to_pct=100.0,
                source="sgx_nifty",
            )
        else:
            return GapAssessment(
                expected_gap_pct=gap_pct,
                severity=GapSeverity.NORMAL,
                action="none",
                reduce_to_pct=100.0,
                source="sgx_nifty",
            )

    def check_overnight_exposure(
        self,
        positions_notional: float,
        equity: float,
    ) -> OvernightExposureCheck:
        """
        Check if overnight exposure is within limits.

        Args:
            positions_notional: total notional of positions held overnight
            equity: current portfolio equity

        Returns:
            OvernightExposureCheck with limit compliance
        """
        if equity <= 0:
            return OvernightExposureCheck(allowed=False, excess_notional=positions_notional)

        exposure_pct = (positions_notional / equity) * 100
        excess = max(0, positions_notional - equity * self.max_overnight_exposure_pct / 100)

        return OvernightExposureCheck(
            current_exposure_pct=exposure_pct,
            max_exposure_pct=self.max_overnight_exposure_pct,
            allowed=exposure_pct <= self.max_overnight_exposure_pct,
            excess_notional=excess,
        )

    def is_pre_market_window(self) -> bool:
        """Check if current time is in the pre-market analysis window (IST)."""
        now_utc = datetime.now(timezone.utc)
        now_ist = now_utc + IST_OFFSET
        current_time = now_ist.time()
        return self.pre_market_start <= current_time <= self.pre_market_end

    def positions_to_reduce(
        self,
        positions: List[dict],
        target_pct: float,
    ) -> List[dict]:
        """
        Calculate orders needed to reduce positions to target percentage.

        Args:
            positions: list of {symbol, side, quantity, price}
            target_pct: target as % of current (e.g., 50 = reduce to half)

        Returns:
            List of {symbol, side, quantity} representing reduction orders
        """
        if target_pct >= 100 or not positions:
            return []

        reduce_fraction = 1.0 - (target_pct / 100.0)
        reduction_orders = []

        for pos in positions:
            reduce_qty = int(pos.get("quantity", 0) * reduce_fraction)
            if reduce_qty > 0:
                # Close side is opposite of position side
                close_side = "SELL" if pos.get("side", "BUY").upper() == "BUY" else "BUY"
                reduction_orders.append({
                    "symbol": pos["symbol"],
                    "side": close_side,
                    "quantity": reduce_qty,
                    "reason": f"gap_protection_reduce_{int(target_pct)}pct",
                })

        return reduction_orders

    async def execute_gap_action(
        self,
        submit_order_fn,
        positions: List[dict],
        assessment: Optional[GapAssessment] = None,
        pre_market_nifty: Optional[float] = None,
    ) -> int:
        """
        Execute gap risk action: submit reduce/close orders based on assessment.

        Args:
            submit_order_fn: async callable to submit orders (OrderEntryService.submit_order)
            positions: list of {symbol, side, quantity, price, exchange?, strategy_id?}
            assessment: pre-computed GapAssessment, or None to compute from pre_market_nifty
            pre_market_nifty: pre-market Nifty value (used if assessment not provided)

        Returns:
            Number of orders submitted
        """
        if assessment is None and pre_market_nifty is not None:
            assessment = self.assess_gap(pre_market_nifty)

        if assessment is None or assessment.action == "none":
            return 0

        if assessment.severity not in (GapSeverity.HIGH, GapSeverity.SEVERE):
            return 0

        reduction_orders = self.positions_to_reduce(positions, assessment.reduce_to_pct)
        if not reduction_orders:
            logger.info("Gap risk: no positions to reduce (action=%s)", assessment.action)
            return 0

        logger.warning(
            "Gap risk executing: gap=%.1f%% severity=%s action=%s reducing %d positions",
            assessment.expected_gap_pct, assessment.severity.value, assessment.action, len(reduction_orders),
        )

        submitted = 0
        for order_spec in reduction_orders:
            try:
                # Import here to avoid circular imports
                from src.core.events import Signal, SignalSide, Exchange, OrderType
                from src.execution.order_entry.request import OrderEntryRequest

                side = SignalSide.SELL if order_spec["side"] == "SELL" else SignalSide.BUY
                signal = Signal(
                    strategy_id=order_spec.get("strategy_id", "gap_risk_protection"),
                    symbol=order_spec["symbol"],
                    exchange=Exchange(order_spec.get("exchange", "NSE")),
                    side=side,
                    score=1.0,
                    portfolio_weight=0.0,
                    risk_level="EMERGENCY",
                    reason=order_spec.get("reason", f"gap_risk_{assessment.severity.value}"),
                    price=order_spec.get("price", 0),
                    ts=datetime.now(timezone.utc),
                )
                req = OrderEntryRequest(
                    signal=signal,
                    quantity=order_spec["quantity"],
                    order_type=OrderType.LIMIT if order_spec.get("price", 0) > 0 else OrderType.MARKET,
                    limit_price=order_spec.get("price"),
                    source="gap_risk_protection",
                    force_reduce=True,
                )
                result = await submit_order_fn(req)
                if result.success:
                    submitted += 1
                    logger.info("Gap risk order submitted: %s %s %s qty=%d",
                                order_spec["side"], order_spec["symbol"], result.order_id, order_spec["quantity"])
                else:
                    logger.warning("Gap risk order rejected: %s %s: %s",
                                   order_spec["symbol"], result.reject_reason, result.reject_detail)
            except Exception as e:
                logger.exception("Gap risk order failed for %s: %s", order_spec.get("symbol"), e)

        logger.info("Gap risk execution complete: %d/%d orders submitted", submitted, len(reduction_orders))
        return submitted
