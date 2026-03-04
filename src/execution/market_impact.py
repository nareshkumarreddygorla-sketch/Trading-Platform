"""
Market impact model for execution quality.

Implements Almgren-Chriss framework:
  - Temporary impact: immediate price impact that decays
  - Permanent impact: lasting price impact from information
  - Transaction cost budget: only trade when expected alpha > expected cost
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ImpactEstimate:
    """Estimated market impact for an order."""
    temporary_impact_bps: float = 0.0
    permanent_impact_bps: float = 0.0
    total_impact_bps: float = 0.0
    transaction_cost_bps: float = 0.0  # from IndiaCostCalculator
    total_cost_bps: float = 0.0        # impact + transaction costs
    participation_rate: float = 0.0
    recommended_qty: int = 0            # qty where impact < alpha
    alpha_sufficient: bool = True       # True if alpha > total cost

    def as_dict(self) -> dict:
        return {
            "temporary_impact_bps": round(self.temporary_impact_bps, 2),
            "permanent_impact_bps": round(self.permanent_impact_bps, 2),
            "total_impact_bps": round(self.total_impact_bps, 2),
            "transaction_cost_bps": round(self.transaction_cost_bps, 2),
            "total_cost_bps": round(self.total_cost_bps, 2),
            "participation_rate": round(self.participation_rate, 4),
            "recommended_qty": self.recommended_qty,
            "alpha_sufficient": self.alpha_sufficient,
        }


@dataclass
class NSEImpactConfig:
    """NSE-specific market impact parameters (calibrated from empirical NSE data)."""
    gamma: float = 0.25           # permanent impact coefficient (NSE empirical, higher than US 0.1)
    min_adv: float = 10_000       # minimum ADV floor
    default_sigma: float = 0.02   # default daily vol for NSE equities
    transaction_cost_bps: float = 12.0  # realistic India round-trip cost (STT+brokerage+GST+stamp)


class MarketImpactModel:
    """
    Almgren-Chriss market impact estimator, calibrated for NSE/BSE.

    Temp impact:  sigma * sqrt(Q / ADV)  (in return units)
    Perm impact:  gamma * (Q / ADV)       (in return units)

    where:
      sigma = daily volatility
      Q     = order quantity
      ADV   = average daily volume (shares)
      gamma = permanent impact coefficient (NSE: 0.25, US: 0.1)
    """

    def __init__(
        self,
        gamma: float = 0.25,       # NSE calibrated (was 0.1 for US)
        min_adv: float = 10_000,
        config: Optional[NSEImpactConfig] = None,
    ):
        if config:
            self.gamma = config.gamma
            self.min_adv = config.min_adv
        else:
            self.gamma = gamma
            self.min_adv = min_adv

    def estimate(
        self,
        quantity: int,
        price: float,
        adv: float,
        sigma: float = 0.02,
        transaction_cost_bps: float = 5.0,
    ) -> ImpactEstimate:
        """
        Estimate market impact for an order.

        Args:
            quantity: order quantity (shares)
            price: order price
            adv: average daily volume (shares)
            sigma: daily volatility (e.g., 0.02 = 2%)
            transaction_cost_bps: India transaction costs in basis points

        Returns:
            ImpactEstimate with detailed breakdown
        """
        if quantity <= 0 or price <= 0:
            return ImpactEstimate()

        adv = max(adv, self.min_adv)
        participation = quantity / adv

        # Temporary impact: sigma * sqrt(participation) * 10000 (in bps)
        temp_impact_bps = sigma * math.sqrt(min(participation, 1.0)) * 10_000

        # Permanent impact: gamma * participation * 10000 (in bps)
        perm_impact_bps = self.gamma * min(participation, 1.0) * 10_000

        total_impact_bps = temp_impact_bps + perm_impact_bps
        total_cost_bps = total_impact_bps + transaction_cost_bps

        return ImpactEstimate(
            temporary_impact_bps=temp_impact_bps,
            permanent_impact_bps=perm_impact_bps,
            total_impact_bps=total_impact_bps,
            transaction_cost_bps=transaction_cost_bps,
            total_cost_bps=total_cost_bps,
            participation_rate=participation,
            recommended_qty=quantity,
            alpha_sufficient=True,  # caller checks this
        )

    def check_alpha_sufficient(
        self,
        quantity: int,
        price: float,
        adv: float,
        expected_alpha_bps: float,
        sigma: float = 0.02,
        transaction_cost_bps: float = 5.0,
    ) -> ImpactEstimate:
        """
        Check if expected alpha exceeds total expected cost (impact + txn costs).
        If not, find the maximum quantity where alpha > cost.

        Args:
            quantity: desired order quantity
            price: order price
            adv: average daily volume
            expected_alpha_bps: expected signal alpha in basis points
            sigma: daily volatility
            transaction_cost_bps: India transaction costs in bps

        Returns:
            ImpactEstimate with recommended_qty and alpha_sufficient flag
        """
        estimate = self.estimate(quantity, price, adv, sigma, transaction_cost_bps)

        if estimate.total_cost_bps <= expected_alpha_bps:
            estimate.alpha_sufficient = True
            estimate.recommended_qty = quantity
            return estimate

        # Alpha insufficient — find max qty where cost < alpha
        # Binary search for optimal quantity
        estimate.alpha_sufficient = False
        low, high = 0, quantity
        best_qty = 0

        for _ in range(20):  # max iterations
            if low >= high:
                break
            mid = (low + high) // 2
            if mid <= 0:
                break
            mid_est = self.estimate(mid, price, adv, sigma, transaction_cost_bps)
            if mid_est.total_cost_bps <= expected_alpha_bps:
                best_qty = mid
                low = mid + 1
            else:
                high = mid

        estimate.recommended_qty = best_qty
        return estimate

    def should_trade(
        self,
        quantity: int,
        price: float,
        adv: float,
        expected_alpha_bps: float,
        sigma: float = 0.02,
        transaction_cost_bps: float = 5.0,
    ) -> bool:
        """Simple check: should this trade be executed?"""
        result = self.check_alpha_sufficient(
            quantity, price, adv, expected_alpha_bps, sigma, transaction_cost_bps
        )
        return result.alpha_sufficient
