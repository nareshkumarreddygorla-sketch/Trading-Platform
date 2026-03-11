"""
Phase 3: Dynamic slippage model.
Slippage_bps = base + k1 * (order_size / avg_minute_volume)^alpha
             + k2 * (spread_bps / ref_spread) + k3 * vol_regime_mult
Nonlinear scaling with participation; used in backtest and live participation cap.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class DynamicSlippageConfig:
    base_bps: float = 5.0
    k1: float = 10.0  # participation scaling
    alpha: float = 1.5  # nonlinear
    k2: float = 2.0   # spread
    k3: float = 5.0   # vol regime
    ref_spread_bps: float = 10.0
    max_volume_participation_pct: float = 10.0
    max_slippage_bps: float = 100.0


class DynamicSlippageModel:
    """
    Slippage = f(order_size / avg_minute_volume, spread_bps, vol_regime_mult).
    For backtest: fill size capped by volume participation; slippage applied to fill price.
    For live: reduce size if participation would exceed threshold.
    """

    def __init__(self, config: Optional[DynamicSlippageConfig] = None):
        self.config = config or DynamicSlippageConfig()

    def participation_ratio(self, order_size: float, avg_minute_volume: float) -> float:
        """order_size / avg_minute_volume; cap at 1.0 for safety."""
        if avg_minute_volume <= 0:
            return 1.0
        return min(1.0, order_size / avg_minute_volume)

    def slippage_bps(
        self,
        order_size: float,
        avg_minute_volume: float,
        spread_bps: float = 10.0,
        vol_regime_mult: float = 1.0,
    ) -> float:
        """
        Dynamic slippage in bps.
        Slippage_bps = base + k1 * participation^alpha + k2 * (spread/ref) + k3 * vol_regime_mult.
        """
        cfg = self.config
        part = self.participation_ratio(order_size, avg_minute_volume)
        term1 = cfg.k1 * (part ** cfg.alpha)
        term2 = cfg.k2 * (spread_bps / (cfg.ref_spread_bps + 1e-8))
        term3 = cfg.k3 * vol_regime_mult
        bps = cfg.base_bps + term1 + term2 + term3
        return min(cfg.max_slippage_bps, max(0.0, bps))

    def fill_price(self, base_price: float, side: str, slippage_bps: float) -> float:
        """Apply slippage in bps to base price."""
        pct = slippage_bps / 10_000.0
        if side.upper() == "BUY":
            return base_price * (1.0 + pct)
        return base_price * (1.0 - pct)

    def max_order_size_for_participation(self, avg_minute_volume: float) -> float:
        """Max order size such that participation <= max_volume_participation_pct/100."""
        return avg_minute_volume * (self.config.max_volume_participation_pct / 100.0)
