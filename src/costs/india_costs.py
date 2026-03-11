"""
India market transaction cost calculator.

Covers all NSE/BSE charges for equity trading:
  - Securities Transaction Tax (STT)
  - Brokerage (Angel One flat-fee model)
  - GST (18% on brokerage + exchange charges + SEBI fees)
  - Exchange transaction charges (NSE/BSE equity)
  - Stamp duty (state-based, default Maharashtra)
  - SEBI turnover fee

References:
  - NSE circular on charges (updated annually)
  - SEBI (Fees and Other Charges) Regulations
  - Angel One schedule of charges
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ProductType(str, Enum):
    INTRADAY = "INTRADAY"
    DELIVERY = "DELIVERY"  # CNC


@dataclass
class CostBreakdown:
    """Itemised cost breakdown for a single fill."""

    stt: float = 0.0
    brokerage: float = 0.0
    exchange_charges: float = 0.0
    sebi_fee: float = 0.0
    gst: float = 0.0
    stamp_duty: float = 0.0
    total: float = 0.0

    def as_dict(self) -> dict:
        return {
            "stt": round(self.stt, 4),
            "brokerage": round(self.brokerage, 4),
            "exchange_charges": round(self.exchange_charges, 4),
            "sebi_fee": round(self.sebi_fee, 4),
            "gst": round(self.gst, 4),
            "stamp_duty": round(self.stamp_duty, 4),
            "total": round(self.total, 4),
        }


@dataclass
class IndiaCostConfig:
    # STT rates
    stt_delivery_buy_pct: float = 0.1  # 0.1% on buy side (delivery)
    stt_delivery_sell_pct: float = 0.1  # 0.1% on sell side (delivery)
    stt_intraday_sell_pct: float = 0.025  # 0.025% on sell side only (intraday)

    # Brokerage (Angel One flat fee model)
    brokerage_flat_per_order: float = 20.0  # ₹20 flat per executed order
    brokerage_pct_cap: float = 0.03  # 0.03% cap (whichever is lower)

    # Exchange transaction charges (NSE equity)
    nse_txn_charge_pct: float = 0.00345  # 0.00345% of turnover
    bse_txn_charge_pct: float = 0.00375  # 0.00375% of turnover

    # SEBI turnover fee
    sebi_fee_pct: float = 0.0001  # 0.0001% (₹10 per crore)

    # GST rate on (brokerage + exchange charges + SEBI fee)
    gst_pct: float = 18.0  # 18%

    # Stamp duty (state of Maharashtra, buyer side only)
    stamp_duty_buy_pct: float = 0.015  # 0.015% on buy side
    stamp_duty_sell_pct: float = 0.0  # 0% on sell side


class IndiaCostCalculator:
    """
    Calculate realistic India equity transaction costs.

    Usage:
        calc = IndiaCostCalculator()
        costs = calc.calculate("BUY", notional=250000, product_type="INTRADAY", exchange="NSE")
        print(costs.total)  # total cost in INR
    """

    def __init__(self, config: IndiaCostConfig | None = None):
        self.config = config or IndiaCostConfig()

    def calculate(
        self,
        side: str,
        notional: float,
        product_type: str = "INTRADAY",
        exchange: str = "NSE",
    ) -> CostBreakdown:
        """
        Calculate full cost breakdown for a single fill.

        Args:
            side: "BUY" or "SELL"
            notional: qty * price in INR
            product_type: "INTRADAY" or "DELIVERY"
            exchange: "NSE" or "BSE"

        Returns:
            CostBreakdown with all components
        """
        if notional <= 0:
            return CostBreakdown()

        cfg = self.config
        is_buy = side.upper() == "BUY"
        is_delivery = product_type.upper() in ("DELIVERY", "CNC")
        is_nse = exchange.upper() == "NSE"

        # 1. STT
        stt = 0.0
        if is_delivery:
            if is_buy:
                stt = notional * cfg.stt_delivery_buy_pct / 100
            else:
                stt = notional * cfg.stt_delivery_sell_pct / 100
        else:
            # Intraday: STT only on sell side
            if not is_buy:
                stt = notional * cfg.stt_intraday_sell_pct / 100

        # 2. Brokerage: min(flat fee, pct cap)
        brokerage_pct = notional * cfg.brokerage_pct_cap / 100
        brokerage = min(cfg.brokerage_flat_per_order, brokerage_pct)

        # 3. Exchange transaction charges
        txn_rate = cfg.nse_txn_charge_pct if is_nse else cfg.bse_txn_charge_pct
        exchange_charges = notional * txn_rate / 100

        # 4. SEBI turnover fee
        sebi_fee = notional * cfg.sebi_fee_pct / 100

        # 5. GST: 18% on (brokerage + exchange charges + SEBI fee)
        gst_base = brokerage + exchange_charges + sebi_fee
        gst = gst_base * cfg.gst_pct / 100

        # 6. Stamp duty (buy side only)
        stamp_duty = 0.0
        if is_buy:
            stamp_duty = notional * cfg.stamp_duty_buy_pct / 100

        total = stt + brokerage + exchange_charges + sebi_fee + gst + stamp_duty

        return CostBreakdown(
            stt=stt,
            brokerage=brokerage,
            exchange_charges=exchange_charges,
            sebi_fee=sebi_fee,
            gst=gst,
            stamp_duty=stamp_duty,
            total=total,
        )

    def round_trip_cost(
        self,
        buy_notional: float,
        sell_notional: float,
        product_type: str = "INTRADAY",
        exchange: str = "NSE",
    ) -> CostBreakdown:
        """Calculate total costs for a round-trip (buy + sell)."""
        buy_costs = self.calculate("BUY", buy_notional, product_type, exchange)
        sell_costs = self.calculate("SELL", sell_notional, product_type, exchange)
        return CostBreakdown(
            stt=buy_costs.stt + sell_costs.stt,
            brokerage=buy_costs.brokerage + sell_costs.brokerage,
            exchange_charges=buy_costs.exchange_charges + sell_costs.exchange_charges,
            sebi_fee=buy_costs.sebi_fee + sell_costs.sebi_fee,
            gst=buy_costs.gst + sell_costs.gst,
            stamp_duty=buy_costs.stamp_duty + sell_costs.stamp_duty,
            total=buy_costs.total + sell_costs.total,
        )

    def cost_pct(self, side: str, notional: float, product_type: str = "INTRADAY", exchange: str = "NSE") -> float:
        """Return total cost as a percentage of notional."""
        if notional <= 0:
            return 0.0
        costs = self.calculate(side, notional, product_type, exchange)
        return (costs.total / notional) * 100
