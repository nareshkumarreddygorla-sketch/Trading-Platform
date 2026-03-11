"""
Triple-barrier labeling: profit target, stop loss, time limit.
Volatility-adjusted barriers; cost-aware variant.
No lookahead: only information at t and future prices for barrier touch (no future in features).
"""

from dataclasses import dataclass

import numpy as np

from .cost_aware import cost_aware_barriers


@dataclass
class TripleBarrierConfig:
    """Parameters for triple-barrier label generation."""

    profit_target_pct: float = 0.01  # b_u: upper barrier as fraction of price
    stop_loss_pct: float = 0.01  # b_d: lower barrier
    max_holding_bars: int = 30  # H: time limit
    round_trip_cost_pct: float = 0.001  # c: slippage + commission both sides
    use_volatility_adjust: bool = True  # scale barriers by ATR/vol
    vol_scale: float = 1.0  # k_u, k_d = vol_scale * sigma
    neutral_threshold: float = 1e-6  # |r_H| < this => label 0


class TripleBarrierLabeler:
    """
    Generate labels from price series using triple-barrier method.
    For each bar t: look forward until first touch of U, L, or H bars; label +1, -1, or 0.
    """

    def __init__(self, config: TripleBarrierConfig | None = None):
        self.config = config or TripleBarrierConfig()

    def _volatility_at(self, prices: np.ndarray, t: int, window: int = 20) -> float:
        if t < window or len(prices) < t + 1:
            return 0.01
        returns = np.diff(prices[t - window : t + 1]) / (prices[t - window : t] + 1e-12)
        return float(np.std(returns))

    def label_at(
        self,
        t: int,
        prices: np.ndarray,
        b_u: float | None = None,
        b_d: float | None = None,
        H: int | None = None,
        cost_pct: float | None = None,
    ) -> int:
        """
        Return label at bar t: +1 (upper first), -1 (lower first), 0 (time out or neutral).
        Uses only prices[t], prices[t+1], ... prices[t+H] for barrier touch (no future in features).
        """
        b_u = b_u if b_u is not None else self.config.profit_target_pct
        b_d = b_d if b_d is not None else self.config.stop_loss_pct
        H = H if H is not None else self.config.max_holding_bars
        cost_pct = cost_pct if cost_pct is not None else self.config.round_trip_cost_pct

        if self.config.use_volatility_adjust:
            vol = self._volatility_at(prices, t)
            b_u = min(b_u, self.config.vol_scale * vol * np.sqrt(H))
            b_d = min(b_d, self.config.vol_scale * vol * np.sqrt(H))

        p_t = float(prices[t])
        U, L = cost_aware_barriers(p_t, b_u, b_d, cost_pct)

        for tau in range(1, min(H + 1, len(prices) - t)):
            p_f = float(prices[t + tau])
            if p_f >= U:
                return 1
            if p_f <= L:
                return -1

        # Time limit: label by sign of return over H
        end_idx = min(t + H, len(prices) - 1)
        r_H = (prices[end_idx] - p_t) / (p_t + 1e-12)
        if abs(r_H) < self.config.neutral_threshold:
            return 0
        return 1 if r_H > 0 else -1

    def label_series(
        self,
        prices: np.ndarray,
        b_u: float | None = None,
        b_d: float | None = None,
        H: int | None = None,
    ) -> np.ndarray:
        """Labels for all valid t (t + H <= len(prices) for time-out; else last bar gets 0)."""
        n = len(prices)
        H = H or self.config.max_holding_bars
        labels = np.zeros(n, dtype=np.int32)
        for t in range(n - 1):
            if t + H < n:
                labels[t] = self.label_at(t, prices, b_u=b_u, b_d=b_d, H=H)
            else:
                labels[t] = 0
        return labels
