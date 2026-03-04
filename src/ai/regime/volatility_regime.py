"""Volatility-based regime: high / low vs rolling history."""
from typing import List

import numpy as np


class VolatilityRegimeDetector:
    """Classify current vol as high/low relative to rolling percentile."""

    def __init__(self, window: int = 60, high_pct: float = 80.0, low_pct: float = 20.0):
        self.window = window
        self.high_pct = high_pct
        self.low_pct = low_pct
        self._vols: List[float] = []

    def add(self, vol: float) -> None:
        self._vols.append(vol)
        if len(self._vols) > self.window * 2:
            self._vols = self._vols[-self.window * 2 :]

    def regime(self) -> str:
        """Returns 'high' | 'low' | 'normal'."""
        if len(self._vols) < self.window:
            return "normal"
        recent = self._vols[-self.window :]
        p_high = np.percentile(recent, self.high_pct)
        p_low = np.percentile(recent, self.low_pct)
        current = recent[-1]
        if current >= p_high:
            return "high"
        if current <= p_low:
            return "low"
        return "normal"
