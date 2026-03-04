"""
Portfolio allocator: rank signals, enforce max active, allocate capital proportionally,
strategy-level caps, regime/drawdown adjustment, risk-aware sizing.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.core.events import Position, Signal

logger = logging.getLogger(__name__)


@dataclass
class AllocatorConfig:
    max_active_signals: int = 5
    max_capital_pct_per_signal: float = 10.0
    strategy_cap_pct: Optional[dict] = None  # strategy_id -> max % of equity
    min_confidence: float = 0.5
    drawdown_exposure_scale: float = 1.0  # reduce exposure when drawdown; 0.5 = half
    regime_scale: float = 1.0  # reduce in bad regime
    use_kelly: bool = True  # Enable Kelly criterion position sizing
    kelly_fraction: float = 0.5  # Half-Kelly for safety


class PortfolioAllocator:
    """
    Rank signals, enforce max active, allocate capital proportionally.
    Respect strategy-level capital caps, drawdown and regime scaling.
    """

    def __init__(self, config: Optional[AllocatorConfig] = None):
        self.config = config or AllocatorConfig()

    def allocate(
        self,
        signals: List[Signal],
        equity: float,
        positions: List[Position],
        *,
        exposure_multiplier: float = 1.0,
        drawdown_scale: Optional[float] = None,
        regime_scale: Optional[float] = None,
        max_position_pct: float = 5.0,
    ) -> List[Tuple[Signal, int]]:
        """
        Returns list of (signal, quantity) to submit. Quantity is risk-adjusted.
        Filters by min_confidence, applies max_active_signals, capital allocation.
        """
        if equity <= 0:
            return []
        scale = exposure_multiplier * (drawdown_scale or self.config.drawdown_exposure_scale) * (regime_scale or self.config.regime_scale)
        scale = max(0.0, min(1.0, scale))

        # Filter by confidence (score)
        candidates = [s for s in signals if s.score >= self.config.min_confidence]
        if not candidates:
            return []

        # Already sorted by score in StrategyRunner; take top N
        top = candidates[: self.config.max_active_signals]

        out: List[Tuple[Signal, int]] = []
        for signal in top:
            strategy_cap_pct = (self.config.strategy_cap_pct or {}).get(signal.strategy_id) or self.config.max_capital_pct_per_signal

            # Kelly criterion: f* = (bp - q) / b
            # b = win/loss ratio estimate, p = win probability (signal score), q = 1-p
            kelly_pct = strategy_cap_pct
            if self.config.use_kelly and signal.score > 0.5:
                p = signal.score  # Win probability from model confidence
                q = 1.0 - p
                b = 1.5  # Assumed reward/risk ratio (1.5:1)
                kelly_full = ((b * p) - q) / b if b > 0 else 0.0
                kelly_half = kelly_full * self.config.kelly_fraction
                # Convert to % of equity, bounded by strategy cap
                kelly_pct = min(strategy_cap_pct, max(1.0, kelly_half * 100.0))

            notional_per_signal = equity * (kelly_pct / 100.0) * scale
            price = signal.price or 0.0
            if price <= 0:
                continue
            raw_qty = int(notional_per_signal / price)
            if raw_qty <= 0:
                continue
            # Cap by max position size (% of equity)
            max_notional = equity * (max_position_pct / 100.0)
            max_qty = int(max_notional / price) if price > 0 else 0
            qty = min(raw_qty, max_qty)
            if qty > 0:
                out.append((signal, qty))
        return out
