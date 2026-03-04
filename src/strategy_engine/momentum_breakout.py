"""
Momentum Breakout Strategy: volume breakout + ADX trend confirmation.
Best in: TRENDING_UP, TRENDING_DOWN regimes.
"""
from datetime import datetime, timezone
from typing import List

import numpy as np

from src.core.events import Signal, SignalSide
from .base import MarketState, StrategyBase


class MomentumBreakoutStrategy(StrategyBase):
    """
    BUY when price breaks above 20-bar high on volume surge + strong ADX.
    SELL when price breaks below 20-bar low on volume surge + strong ADX.
    """
    strategy_id = "momentum_breakout"
    description = "Volume breakout with ADX trend confirmation"

    def __init__(self, lookback: int = 20, volume_mult: float = 1.5, adx_min: float = 25.0):
        self.lookback = lookback
        self.volume_mult = volume_mult
        self.adx_min = adx_min

    def warm(self, state: MarketState) -> bool:
        return len(state.bars) >= self.lookback + 15

    def generate_signals(self, state: MarketState) -> List[Signal]:
        if not self.warm(state):
            return []

        bars = state.bars
        closes = np.array([b.close for b in bars], dtype=float)
        highs = np.array([b.high for b in bars], dtype=float)
        lows = np.array([b.low for b in bars], dtype=float)
        volumes = np.array([b.volume for b in bars], dtype=float)
        price = state.latest_price or closes[-1]

        # 20-bar channel breakout
        channel_high = np.max(highs[-self.lookback - 1:-1])
        channel_low = np.min(lows[-self.lookback - 1:-1])

        # Volume surge check
        avg_vol = np.mean(volumes[-self.lookback - 1:-1])
        current_vol = volumes[-1]
        vol_surge = current_vol > avg_vol * self.volume_mult

        # ADX for trend strength
        adx = self._calc_adx(highs, lows, closes, 14)

        if not vol_surge or adx < self.adx_min:
            return []

        if price > channel_high:
            score = min(1.0, 0.5 + (price - channel_high) / (channel_high * 0.01 + 1e-12))
            return [Signal(
                strategy_id=self.strategy_id,
                symbol=state.symbol,
                exchange=state.exchange,
                side=SignalSide.BUY,
                score=score,
                portfolio_weight=0.15,
                risk_level="NORMAL",
                reason=f"breakout_high adx={adx:.0f} vol_surge={current_vol/avg_vol:.1f}x",
                price=price,
                stop_loss=round(channel_low, 2),
                target=round(price + 2 * (price - channel_low), 2),
                ts=datetime.now(timezone.utc),
                metadata={"adx": adx, "vol_ratio": current_vol / avg_vol},
            )]

        if price < channel_low:
            score = min(1.0, 0.5 + (channel_low - price) / (channel_low * 0.01 + 1e-12))
            return [Signal(
                strategy_id=self.strategy_id,
                symbol=state.symbol,
                exchange=state.exchange,
                side=SignalSide.SELL,
                score=score,
                portfolio_weight=0.15,
                risk_level="NORMAL",
                reason=f"breakout_low adx={adx:.0f} vol_surge={current_vol/avg_vol:.1f}x",
                price=price,
                ts=datetime.now(timezone.utc),
                metadata={"adx": adx, "vol_ratio": current_vol / avg_vol},
            )]

        return []

    @staticmethod
    def _calc_adx(high, low, close, period=14):
        n = len(close)
        if n < period + 1:
            return 0.0
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                           np.maximum(high[1:] - high[:-1], 0), 0.0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                            np.maximum(low[:-1] - low[1:], 0), 0.0)
        atr_s = np.mean(tr[:period])
        p_s = np.mean(plus_dm[:period])
        m_s = np.mean(minus_dm[:period])
        for i in range(period, len(tr)):
            atr_s = (atr_s * (period - 1) + tr[i]) / period
            p_s = (p_s * (period - 1) + plus_dm[i]) / period
            m_s = (m_s * (period - 1) + minus_dm[i]) / period
        if atr_s < 1e-12:
            return 0.0
        plus_di = 100.0 * p_s / atr_s
        minus_di = 100.0 * m_s / atr_s
        di_sum = plus_di + minus_di
        if di_sum < 1e-12:
            return 0.0
        return float(100.0 * abs(plus_di - minus_di) / di_sum)
