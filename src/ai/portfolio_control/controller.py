"""
Phase 7: Portfolio heat & drawdown control.
ExposureScale = max(0.2, 1 - current_drawdown_pct / dd_limit_pct);
heat limit; trade pause threshold; vol spike detection.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class HeatConfig:
    dd_limit_pct: float = 10.0
    trade_pause_dd_pct: float = 3.0
    heat_limit: float = 1.0  # sum |position_value| / equity
    vol_spike_mult: float = 2.0  # current_vol > mult * median_vol -> spike
    vol_lookback: int = 20
    min_exposure_scale: float = 0.2


class PortfolioHeatController:
    """
    Rolling max drawdown; vol spike detection; portfolio heat; trade pause;
    exposure scale for position sizing.
    """

    def __init__(self, config: Optional[HeatConfig] = None):
        self.config = config or HeatConfig()
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
        self._position_values: List[float] = []
        self._vol_history: List[float] = []
        self._paused: bool = False

    def update_equity(self, equity: float) -> None:
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

    def update_positions(self, position_values: List[float]) -> None:
        """position_values: list of absolute position notional (long or short)."""
        self._position_values = list(position_values)

    def update_vol(self, vol: float) -> None:
        self._vol_history.append(vol)
        if len(self._vol_history) > self.config.vol_lookback:
            self._vol_history = self._vol_history[-self.config.vol_lookback :]

    def get_drawdown_pct(self) -> float:
        if self._peak_equity <= 0:
            return 0.0
        dd = (self._peak_equity - self._current_equity) / self._peak_equity * 100.0
        return max(0.0, dd)

    def get_heat(self) -> float:
        """Portfolio heat = sum |position_value| / equity."""
        if self._current_equity <= 0:
            return 0.0
        total = sum(abs(v) for v in self._position_values)
        return total / self._current_equity

    def get_exposure_scale(self) -> float:
        """ExposureScale = max(0.2, 1 - current_dd_pct / dd_limit_pct)."""
        cfg = self.config
        dd = self.get_drawdown_pct()
        scale = 1.0 - (dd / (cfg.dd_limit_pct + 1e-8))
        return max(cfg.min_exposure_scale, min(1.0, scale))

    def should_pause_new_trades(self) -> bool:
        """True if drawdown >= trade_pause_dd_pct or manual pause."""
        if self._paused:
            return True
        return self.get_drawdown_pct() >= self.config.trade_pause_dd_pct

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def vol_spike_detected(self) -> bool:
        """True if current vol > vol_spike_mult * median(vol_history)."""
        if len(self._vol_history) < 5:
            return False
        median_vol = float(np.median(self._vol_history))
        if median_vol <= 0:
            return False
        return self._vol_history[-1] > self.config.vol_spike_mult * median_vol
