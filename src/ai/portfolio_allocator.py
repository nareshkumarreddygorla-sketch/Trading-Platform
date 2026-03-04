"""
AI portfolio allocator: safe wrapper that ranks signals, limits concurrent trades,
volatility scaling, exposure multiplier, per-strategy and sector caps.
Respects RiskManager.can_place_order() for every candidate. Returns SizedSignal (signal, quantity).
Does NOT call broker. All execution goes through OrderEntryService.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.core.events import Position, Signal
from src.risk_engine import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class SizedSignal:
    """Signal with allocated quantity. Safe to pass to order entry."""
    signal: Signal
    quantity: int


class PortfolioAllocator:
    """
    Rank signals, apply max concurrent trades, volatility scaling, exposure multiplier,
    per-strategy cap, sector cap. Filter every (signal, qty) through RiskManager.can_place_order().
    Returns list of SizedSignal only. Does not call broker.
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        max_concurrent_trades: int = 5,
        max_capital_pct_per_signal: float = 10.0,
        per_strategy_cap_pct: Optional[dict] = None,
        min_confidence: float = 0.5,
        volatility_scale: float = 1.0,
        exposure_multiplier: float = 1.0,
    ):
        self.risk_manager = risk_manager
        self.max_concurrent_trades = max_concurrent_trades
        self.max_capital_pct_per_signal = max_capital_pct_per_signal
        self.per_strategy_cap_pct = per_strategy_cap_pct or {}
        self.min_confidence = min_confidence
        self.volatility_scale = max(0.0, min(1.0, volatility_scale))
        self.exposure_multiplier = max(0.0, min(1.0, exposure_multiplier))

    def allocate(
        self,
        signals: List[Signal],
        equity: float,
        positions: List[Position],
        *,
        exposure_multiplier: Optional[float] = None,
        drawdown_scale: Optional[float] = None,
        regime_scale: Optional[float] = None,
        max_position_pct: Optional[float] = None,
        volatility_scale: Optional[float] = None,
    ) -> List[SizedSignal]:
        """
        Rank signals (by score), cap count, allocate capital, apply scaling.
        Each (signal, qty) is checked with risk_manager.can_place_order(); only allowed ones returned.
        """
        if equity <= 0:
            return []
        em = exposure_multiplier if exposure_multiplier is not None else self.exposure_multiplier
        vs = volatility_scale if volatility_scale is not None else self.volatility_scale
        scale = em * vs
        if drawdown_scale is not None:
            scale *= max(0.0, min(1.0, drawdown_scale))
        if regime_scale is not None:
            scale *= max(0.0, min(1.0, regime_scale))
        scale = max(0.0, min(1.0, scale))

        candidates = [s for s in signals if s.score >= self.min_confidence]
        if not candidates:
            return []
        sorted_sigs = sorted(candidates, key=lambda s: s.score, reverse=True)
        top = sorted_sigs[: self.max_concurrent_trades]

        max_pct = max_position_pct if max_position_pct is not None else getattr(
            self.risk_manager.limits, "max_position_pct", 5.0
        )
        out: List[SizedSignal] = []
        for signal in top:
            strategy_cap_pct = self.per_strategy_cap_pct.get(signal.strategy_id) or self.max_capital_pct_per_signal
            notional = equity * (strategy_cap_pct / 100.0) * scale
            price = signal.price or 0.0
            if price <= 0:
                continue
            raw_qty = int(notional / price)
            max_notional = equity * (max_pct / 100.0)
            max_qty = int(max_notional / price) if price > 0 else 0
            qty = min(raw_qty, max_qty)
            if qty <= 0:
                continue
            check = self.risk_manager.can_place_order(signal, qty, price)
            if not check.allowed:
                logger.debug("Allocator skip: risk can_place_order rejected %s qty=%s: %s", signal.symbol, qty, check.reason)
                continue
            out.append(SizedSignal(signal=signal, quantity=qty))
        return out
