"""
Volatility-targeted portfolio construction.

Targets constant portfolio volatility by scaling exposure inversely with realized vol.
When vol is high → reduce position sizes. When vol is low → increase.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VolTargetState:
    """Current vol-targeting state."""

    target_vol_annual: float = 12.0  # target annualised vol (%)
    realized_vol_annual: float = 12.0  # current realised vol (%)
    realized_vol_daily: float = 0.0
    scale_factor: float = 1.0  # position size multiplier
    min_scale: float = 0.25
    max_scale: float = 2.0
    n_observations: int = 0

    def as_dict(self) -> dict:
        return {
            "target_vol_annual": round(self.target_vol_annual, 2),
            "realized_vol_annual": round(self.realized_vol_annual, 2),
            "realized_vol_daily": round(self.realized_vol_daily, 6),
            "scale_factor": round(self.scale_factor, 4),
            "n_observations": self.n_observations,
        }


class VolatilityTargeter:
    """
    Targets constant portfolio volatility.

    Usage:
        vt = VolatilityTargeter(target_vol_annual=12.0)
        vt.record_daily_return(0.005)
        scale = vt.get_scale_factor()
        position_size *= scale
    """

    def __init__(
        self,
        target_vol_annual: float = 12.0,
        ewma_lambda: float = 0.94,
        lookback: int = 20,
        min_scale: float = 0.25,
        max_scale: float = 2.0,
        min_observations: int = 5,
    ):
        self.target_vol_annual = target_vol_annual
        self.target_vol_daily = target_vol_annual / math.sqrt(252) / 100  # convert to daily decimal
        self.ewma_lambda = ewma_lambda
        self.lookback = lookback
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_observations = min_observations

        self._daily_returns: deque[float] = deque(maxlen=lookback * 2)
        self._ewma_var: float | None = None
        self._state = VolTargetState(
            target_vol_annual=target_vol_annual,
            min_scale=min_scale,
            max_scale=max_scale,
        )

    @property
    def state(self) -> VolTargetState:
        return self._state

    def record_daily_return(self, daily_return: float) -> None:
        """Record a portfolio daily return observation."""
        self._daily_returns.append(daily_return)
        # Update EWMA variance
        if self._ewma_var is None:
            self._ewma_var = daily_return**2
        else:
            self._ewma_var = self.ewma_lambda * self._ewma_var + (1 - self.ewma_lambda) * daily_return**2
        self._update_state()

    def _update_state(self) -> None:
        """Recalculate scale factor."""
        n = len(self._daily_returns)
        self._state.n_observations = n

        if n < self.min_observations:
            self._state.scale_factor = 1.0
            return

        # Use EWMA vol
        if self._ewma_var is not None and self._ewma_var > 0:
            daily_vol = math.sqrt(self._ewma_var)
        else:
            daily_vol = float(np.std(list(self._daily_returns)))

        self._state.realized_vol_daily = daily_vol
        self._state.realized_vol_annual = daily_vol * math.sqrt(252) * 100

        if daily_vol > 0:
            raw_scale = self.target_vol_daily / daily_vol
            self._state.scale_factor = max(self.min_scale, min(self.max_scale, raw_scale))
        else:
            self._state.scale_factor = self.max_scale

    def get_scale_factor(self) -> float:
        """Get current vol-target scaling factor for position sizes."""
        return self._state.scale_factor

    def apply_to_quantity(self, base_quantity: int) -> int:
        """Apply vol-targeting scale to a position quantity."""
        return max(1, int(base_quantity * self._state.scale_factor))

    def reset(self) -> None:
        """Reset all state."""
        self._daily_returns.clear()
        self._ewma_var = None
        self._state = VolTargetState(
            target_vol_annual=self.target_vol_annual,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
        )
