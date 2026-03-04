"""
Unified fill model: same logic for backtest and live simulation.
Latency (N bars delay), slippage (bps), spread, volume participation limit, commission.
No lookahead: fill at bar i+latency_bars using that bar's open/close and volume.
"""
from dataclasses import dataclass
from typing import Optional

from src.core.events import Bar

from .slippage import SlippageModel


@dataclass
class FillModelConfig:
    latency_bars: int = 1  # fill at bar i+latency_bars
    slippage_bps: float = 5.0
    spread_bps: float = 3.0  # half-spread each side
    max_volume_participation_pct: float = 10.0  # max fill qty = bar_volume * pct / 100
    commission_pct: float = 0.05
    use_bar_open_for_fill: bool = False  # if True, fill at open of fill bar; else close


class FillModel:
    """
    Given signal at bar index i, compute fill at bar i+latency_bars.
    execute_at_bar_index returns (fill_bar, fill_price, fill_qty, commission).
    fill_qty may be partial if volume limit. Out of range returns (None, 0, 0, 0).
    """

    def __init__(self, config: Optional[FillModelConfig] = None):
        self.config = config or FillModelConfig()
        self.slippage = SlippageModel(self.config.slippage_bps)

    def fill_price(self, bar: Bar, side: str) -> float:
        """Execution price for this bar: open or close + slippage + spread."""
        if self.config.use_bar_open_for_fill:
            base = bar.open
        else:
            base = bar.close
        p = self.slippage.apply(base, side)
        spread_pct = self.config.spread_bps / 10_000.0
        if side.upper() == "BUY":
            return p * (1 + spread_pct)
        return p * (1 - spread_pct)

    def fill_quantity(self, requested_qty: float, bar: Bar) -> float:
        """Cap by volume participation. Returns filled qty."""
        if bar.volume <= 0:
            return 0.0
        max_qty = bar.volume * (self.config.max_volume_participation_pct / 100.0)
        return min(requested_qty, max_qty)

    def commission(self, notional: float) -> float:
        return notional * (self.config.commission_pct / 100.0)

    def execute_at_bar_index(self, signal_bar_index: int, bars: list, side: str, requested_qty: float, price_hint: float) -> tuple[Optional[Bar], float, float, float]:
        """
        Signal was at bars[signal_bar_index]. Fill at signal_bar_index + latency_bars.
        Returns (fill_bar, fill_price, fill_qty, commission).
        If fill bar is out of range, returns (None, 0, 0, 0).
        """
        fill_idx = signal_bar_index + self.config.latency_bars
        if fill_idx >= len(bars):
            return None, 0.0, 0.0, 0.0
        fill_bar = bars[fill_idx]
        fill_price = self.fill_price(fill_bar, side)
        fill_qty = self.fill_quantity(requested_qty, fill_bar)
        if fill_qty <= 0:
            return fill_bar, fill_price, 0.0, 0.0
        notional = fill_price * fill_qty
        comm = self.commission(notional)
        return fill_bar, fill_price, fill_qty, comm
