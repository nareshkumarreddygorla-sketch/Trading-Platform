"""
Strategy Capital Allocation Engine: per-strategy Sharpe, win rate, drawdown;
decay detection; dynamic allocation via risk parity / Kelly / confidence.
Automatically disables weak strategies.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .decay import DecayDetector
from .weights import compute_risk_parity_weights, compute_kelly_weights

try:
    from ..position_sizing import dynamic_position_fraction
except ImportError:
    dynamic_position_fraction = None

logger = logging.getLogger(__name__)


@dataclass
class StrategyStats:
    strategy_id: str
    sharpe: float
    rolling_win_rate: float
    max_drawdown: float
    n_trades: int
    confidence: float  # from model or recent hit rate


@dataclass
class StrategyAllocation:
    strategy_id: str
    weight: float
    enabled: bool
    reason: str


class MetaAllocator:
    """
    Tracks per-strategy performance; detects decay; outputs capital weights.
    Disables strategy if decay or drawdown threshold breached.
    """

    def __init__(
        self,
        min_sharpe: float = 0.5,
        min_win_rate: float = 0.45,
        max_drawdown_pct: float = 15.0,
        decay_lookback: int = 20,
        allocation_method: str = "kelly",  # kelly | risk_parity | confidence
    ):
        self.min_sharpe = min_sharpe
        self.min_win_rate = min_win_rate
        self.max_drawdown_pct = max_drawdown_pct
        self.decay_lookback = decay_lookback
        self.allocation_method = allocation_method
        self._stats: Dict[str, StrategyStats] = {}
        self._returns: Dict[str, List[float]] = {}
        self._decay_detector = DecayDetector(lookback=decay_lookback)

    def update_returns(self, strategy_id: str, period_return: float) -> None:
        if strategy_id not in self._returns:
            self._returns[strategy_id] = []
        self._returns[strategy_id].append(period_return)
        if len(self._returns[strategy_id]) > 500:
            self._returns[strategy_id] = self._returns[strategy_id][-500:]

    def update_stats(
        self,
        strategy_id: str,
        sharpe: float,
        win_rate: float,
        max_dd: float,
        n_trades: int,
        confidence: float = 0.5,
    ) -> None:
        self._stats[strategy_id] = StrategyStats(
            strategy_id=strategy_id,
            sharpe=sharpe,
            rolling_win_rate=win_rate,
            max_drawdown=max_dd,
            n_trades=n_trades,
            confidence=confidence,
        )

    def _is_decayed(self, strategy_id: str) -> bool:
        returns = self._returns.get(strategy_id, [])
        if len(returns) < self.decay_lookback:
            return False
        return self._decay_detector.detect(np.array(returns))

    def allocate(
        self,
        strategy_ids: List[str],
        equity: float,
        current_drawdown_pct: float = 0.0,
        regime_multiplier: float = 1.0,
        meta_alpha_scale: float = 1.0,
        alpha_decay_multipliers: Optional[Dict[str, float]] = None,
    ) -> List[StrategyAllocation]:
        """
        Compute weights for each strategy. Disable if decayed or below min Sharpe/win rate.
        Optional: scale by dynamic position sizing (confidence × drawdown × regime), meta_alpha,
        and alpha_decay_multipliers (from DecayMonitor: Phase H meta-alpha feedback).
        """
        alpha_decay_multipliers = alpha_decay_multipliers or {}
        allocations: List[StrategyAllocation] = []
        active = []
        raw_returns: Dict[str, List[float]] = {}

        for sid in strategy_ids:
            stats = self._stats.get(sid)
            returns = self._returns.get(sid, [])

            if stats is None:
                allocations.append(StrategyAllocation(sid, 0.0, False, "no_stats"))
                continue

            decayed = self._is_decayed(sid)
            if decayed:
                allocations.append(StrategyAllocation(sid, 0.0, False, "decay_detected"))
                continue
            if stats.sharpe < self.min_sharpe:
                allocations.append(StrategyAllocation(sid, 0.0, False, "low_sharpe"))
                continue
            if stats.rolling_win_rate < self.min_win_rate:
                allocations.append(StrategyAllocation(sid, 0.0, False, "low_win_rate"))
                continue
            if stats.max_drawdown >= self.max_drawdown_pct / 100.0:
                allocations.append(StrategyAllocation(sid, 0.0, False, "max_drawdown"))
                continue

            active.append(sid)
            raw_returns[sid] = returns

        if not active:
            return allocations

        if self.allocation_method == "risk_parity":
            weights = compute_risk_parity_weights(
                {sid: self._returns.get(sid, []) for sid in active}
            )
        elif self.allocation_method == "kelly":
            weights = compute_kelly_weights(
                {sid: (self._stats[sid].sharpe, self._stats[sid].rolling_win_rate) for sid in active}
            )
        else:
            # confidence-weighted: weight by stats.confidence
            confs = [self._stats[sid].confidence for sid in active]
            total = sum(confs) or 1
            weights = {sid: self._stats[sid].confidence / total for sid in active}

        # Optional: scale each weight by dynamic position fraction (confidence × drawdown × regime)
        if dynamic_position_fraction is not None:
            scaled = {}
            for sid in active:
                s = self._stats[sid]
                f = dynamic_position_fraction(
                    p_win=s.rolling_win_rate,
                    win_loss_ratio=1.0,
                    confidence=s.confidence,
                    current_drawdown_pct=current_drawdown_pct,
                    regime_multiplier=regime_multiplier,
                )
                scaled[sid] = weights.get(sid, 0.0) * f
            total_s = sum(scaled.values()) or 1.0
            weights = {sid: scaled[sid] / total_s for sid in active}

        # Meta-alpha: scale all weights when primary model likely wrong
        if meta_alpha_scale < 1.0 and meta_alpha_scale > 0:
            weights = {sid: w * meta_alpha_scale for sid, w in weights.items()}
            total_w = sum(weights.values()) or 1.0
            weights = {sid: weights[sid] / total_w for sid in active}

        # Alpha decay (Phase H): per-signal weight from DecayMonitor
        for sid in active:
            decay_mult = alpha_decay_multipliers.get(sid, 1.0)
            if decay_mult <= 0:
                weights[sid] = 0.0
            else:
                weights[sid] = weights.get(sid, 0.0) * decay_mult
        total_w = sum(weights.values()) or 1.0
        if total_w > 0:
            weights = {sid: weights[sid] / total_w for sid in active}

        for sid in strategy_ids:
            if sid in active:
                allocations.append(StrategyAllocation(sid, weights.get(sid, 0.0), True, "ok"))
            elif not any(a.strategy_id == sid for a in allocations):
                allocations.append(StrategyAllocation(sid, 0.0, False, "skipped"))

        return allocations
