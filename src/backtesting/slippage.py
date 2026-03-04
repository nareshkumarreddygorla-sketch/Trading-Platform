"""Slippage model for backtest and execution simulation."""
from typing import Optional

import numpy as np


class SlippageModel:
    """Apply basis-point slippage (e.g. 5 bps = 0.05%)."""

    def __init__(self, bps: float = 5.0):
        self.bps = bps

    def apply(self, price: float, side: str) -> float:
        """Return execution price after slippage (buy: higher, sell: lower)."""
        pct = self.bps / 10_000.0
        if side.upper() == "BUY":
            return price * (1 + pct)
        return price * (1 - pct)

    def apply_random(self, price: float, side: str, seed: Optional[int] = None) -> float:
        """Random slippage within 0..2*bps."""
        if seed is not None:
            np.random.seed(seed)
        pct = np.random.uniform(0, 2 * self.bps / 10_000.0)
        if side.upper() == "BUY":
            return price * (1 + pct)
        return price * (1 - pct)
