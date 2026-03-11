"""
Phase G: Decay monitoring.
Rolling IC, rolling Sharpe, half-life estimate; drift metrics;
gradual weight reduction on decay; abrupt disable only if catastrophic.
"""
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import collections
import math

import numpy as np

from ..validation.ic import ic_rank


@dataclass
class DecayConfig:
    rolling_window: int = 20
    decay_threshold: float = 0.8  # IC(t) < IC(t-L) * threshold -> decay
    half_life_min_days: float = 20.0
    weight_decay_step: float = 0.1  # reduce weight by 10% per step
    weight_floor: float = 0.2
    catastrophic_ic_threshold: float = -0.05
    catastrophic_sharpe_threshold: float = -0.5
    catastrophic_lookback_bars: int = 78 * 10  # ~10 days 1m


class DecayMonitor:
    """
    For each deployed alpha: track rolling IC, rolling Sharpe;
    estimate half-life (exponential decay); detect decay (IC/Sharpe drop);
    recommend weight multiplier (gradual) or disable (catastrophic).
    """

    def __init__(self, config: Optional[DecayConfig] = None):
        self.config = config or DecayConfig()
        self._rolling_ic: Dict[str, Deque[float]] = {}
        self._rolling_sharpe: Dict[str, Deque[float]] = {}
        self._weight_mult: Dict[str, float] = {}
        self._signal_buffer: Dict[str, List[float]] = {}
        self._return_buffer: Dict[str, List[float]] = {}

    def register(self, signal_id: str) -> None:
        self._rolling_ic.setdefault(signal_id, collections.deque(maxlen=self.config.rolling_window))
        self._rolling_sharpe.setdefault(signal_id, collections.deque(maxlen=self.config.rolling_window))
        self._weight_mult[signal_id] = 1.0
        self._signal_buffer.setdefault(signal_id, [])
        self._return_buffer.setdefault(signal_id, [])

    def update(
        self,
        signal_id: str,
        signal: np.ndarray,
        forward_return: np.ndarray,
        sharpe_rolling: Optional[float] = None,
    ) -> None:
        """Append latest window IC (and optional Sharpe); detect decay."""
        self.register(signal_id)
        ic = ic_rank(signal, forward_return)
        self._rolling_ic[signal_id].append(ic)
        if sharpe_rolling is not None:
            self._rolling_sharpe[signal_id].append(sharpe_rolling)

    def rolling_ic(self, signal_id: str) -> float:
        """Current rolling mean IC."""
        q = self._rolling_ic.get(signal_id)
        if not q:
            return 0.0
        return sum(q) / len(q) if q else 0.0

    def rolling_sharpe(self, signal_id: str) -> float:
        """Current rolling mean Sharpe."""
        q = self._rolling_sharpe.get(signal_id)
        if not q:
            return 0.0
        return sum(q) / len(q) if q else 0.0

    def half_life_estimate(self, signal_id: str) -> Optional[float]:
        """Exponential decay: IC(t) = IC_0 * exp(-t/tau). Fit tau; half_life = tau * ln(2)."""
        q = list(self._rolling_ic.get(signal_id, []))
        if len(q) < 5:
            return None
        try:
            # simple: regress log(|IC|+eps) on t
            y = np.log(np.abs(np.array(q)) + 1e-8)
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            if slope >= 0:
                return None
            tau = -1.0 / slope
            half_life = tau * math.log(2)
            return half_life
        except Exception:
            return None

    def decay_detected(self, signal_id: str) -> bool:
        """True if IC or Sharpe dropped by more than decay_threshold vs earlier."""
        self.register(signal_id)
        q = list(self._rolling_ic.get(signal_id, []))
        if len(q) < 2:
            return False
        L = min(len(q) // 2, 5)
        recent = np.mean(q[-L:]) if L else q[-1]
        older = np.mean(q[:-L]) if len(q) > L else q[0]
        if abs(older) < 1e-8:
            return False
        ratio = recent / older if older != 0 else 1.0
        if ratio < self.config.decay_threshold:
            return True
        sq = list(self._rolling_sharpe.get(signal_id, []))
        if len(sq) >= 2 and L:
            s_recent = np.mean(sq[-L:])
            s_older = np.mean(sq[:-L])
            if s_older != 0 and s_recent / s_older < self.config.decay_threshold:
                return True
        return False

    def catastrophic_detected(self, signal_id: str) -> bool:
        """True if IC < catastrophic_ic or Sharpe < catastrophic_sharpe over lookback."""
        q = list(self._rolling_ic.get(signal_id, []))
        if len(q) >= 5:
            mean_ic = np.mean(q[-self.config.catastrophic_lookback_bars :])
            if mean_ic < self.config.catastrophic_ic_threshold:
                return True
        sq = list(self._rolling_sharpe.get(signal_id, []))
        if len(sq) >= 5:
            mean_s = np.mean(sq[-self.config.catastrophic_lookback_bars :])
            if mean_s < self.config.catastrophic_sharpe_threshold:
                return True
        return False

    def recommended_weight_multiplier(self, signal_id: str) -> float:
        """
        Gradual: reduce weight by decay_step when decay_detected; floor at weight_floor.
        Catastrophic: return 0 (disable).
        """
        if self.catastrophic_detected(signal_id):
            return 0.0
        mult = self._weight_mult.get(signal_id, 1.0)
        if self.decay_detected(signal_id):
            mult = max(self.config.weight_floor, mult - self.config.weight_decay_step)
            self._weight_mult[signal_id] = mult
        return mult

    def reset_weight(self, signal_id: str, mult: float = 1.0) -> None:
        """Reset weight multiplier (e.g. after recovery)."""
        self._weight_mult[signal_id] = mult
