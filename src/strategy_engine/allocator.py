"""
Portfolio allocator: rank signals, enforce max active, allocate capital proportionally,
strategy-level caps, regime/drawdown adjustment, risk-aware sizing.
"""

import logging
from dataclasses import dataclass

from src.core.events import Position, Signal

logger = logging.getLogger(__name__)


@dataclass
class AllocatorConfig:
    max_active_signals: int = 5
    max_capital_pct_per_signal: float = 10.0
    strategy_cap_pct: dict | None = None  # strategy_id -> max % of equity
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

    def __init__(self, config: AllocatorConfig | None = None):
        self.config = config or AllocatorConfig()

    def allocate(
        self,
        signals: list[Signal],
        equity: float,
        positions: list[Position],
        *,
        exposure_multiplier: float = 1.0,
        drawdown_scale: float | None = None,
        regime_scale: float | None = None,
        max_position_pct: float = 5.0,
    ) -> list[tuple[Signal, int]]:
        """
        Returns list of (signal, quantity) to submit. Quantity is risk-adjusted.
        Filters by min_confidence, applies max_active_signals, capital allocation.
        """
        if equity <= 0:
            return []
        scale = (
            exposure_multiplier
            * (drawdown_scale or self.config.drawdown_exposure_scale)
            * (regime_scale or self.config.regime_scale)
        )
        scale = max(0.0, min(1.0, scale))

        # Filter by confidence (score)
        candidates = [s for s in signals if s.score >= self.config.min_confidence]
        if not candidates:
            return []

        # Already sorted by score in StrategyRunner; take top N
        top = candidates[: self.config.max_active_signals]

        out: list[tuple[Signal, int]] = []
        for signal in top:
            strategy_cap_pct = (self.config.strategy_cap_pct or {}).get(
                signal.strategy_id
            ) or self.config.max_capital_pct_per_signal

            # Kelly criterion: f* = (bp - q) / b
            # b = win/loss ratio estimate, p = win probability (from signal score), q = 1-p
            # Applied to ALL signals (score already >= min_confidence from filter above).
            # Higher score = higher confidence = larger position; lower score = smaller.
            kelly_pct = strategy_cap_pct
            if self.config.use_kelly:
                # Dampen signal score toward 50% to avoid overconfident sizing
                p = 0.5 + (signal.score - 0.5) * 0.5
                q = 1.0 - p
                b = 1.5  # Assumed reward/risk ratio (1.5:1)
                kelly_full = ((b * p) - q) / b if b > 0 else 0.0
                kelly_half = max(0.0, kelly_full) * self.config.kelly_fraction
                # Convert to % of equity, bounded by strategy cap AND max_position_pct
                kelly_pct = min(strategy_cap_pct, max_position_pct, max(1.0, kelly_half * 100.0))

            # Enforce max_position_pct as hard cap on allocation %
            effective_pct = min(kelly_pct, max_position_pct)
            notional_per_signal = equity * (effective_pct / 100.0) * scale
            price = signal.price or 0.0
            if price <= 0:
                continue
            raw_qty = int(notional_per_signal / price)
            if raw_qty <= 0:
                continue
            # Double-cap by max position size (% of equity) as safety net
            max_notional = equity * (max_position_pct / 100.0)
            max_qty = int(max_notional / price) if price > 0 else 0
            qty = min(raw_qty, max_qty)
            if qty > 0:
                out.append((signal, qty))
        return out
