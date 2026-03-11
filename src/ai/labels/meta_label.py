"""
Meta-labeling: given primary prediction (e.g. +1/-1), meta-label = 1 if actual trade
would have been profitable after costs, else 0. Trained only on samples where primary predicted a trade.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MetaLabelConfig:
    round_trip_cost_pct: float = 0.001
    min_holding_bars: int = 1


class MetaLabeler:
    """
    Compute meta-labels from primary predictions and realized outcomes.
    Primary gives direction (e.g. +1 or -1); meta = 1 if realized PnL after cost > 0.
    """

    def __init__(self, config: Optional[MetaLabelConfig] = None):
        self.config = config or MetaLabelConfig()

    def meta_label(
        self,
        entry_price: float,
        exit_price: float,
        side: int,  # +1 long, -1 short
    ) -> int:
        """
        Realized return after cost: r = (exit/entry - 1) * side - cost.
        Meta = 1 if r > 0 else 0.
        """
        gross = (exit_price / entry_price - 1.0) * float(side)
        cost = self.config.round_trip_cost_pct
        net = gross - cost
        return 1 if net > 0 else 0

    def meta_labels_from_triple_barrier(
        self,
        prices: np.ndarray,
        primary_labels: np.ndarray,
        entry_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        For each bar t where primary_labels[t] != 0, compute exit (first touch or H)
        and meta_label = 1 if profitable after cost.
        entry_indices: if provided, only those t; else all t where primary_labels[t] != 0.
        """
        n = len(prices)
        if entry_indices is None:
            entry_indices = np.where(primary_labels != 0)[0]
        meta = np.zeros(n, dtype=np.int32)
        for t in entry_indices:
            if t >= n - 1:
                continue
            side = int(primary_labels[t])
            entry_price = float(prices[t])
            # Simplified: exit at t+1 (or use triple-barrier exit in production)
            exit_idx = min(t + self.config.min_holding_bars, n - 1)
            exit_price = float(prices[exit_idx])
            meta[t] = self.meta_label(entry_price, exit_price, side)
        return meta
