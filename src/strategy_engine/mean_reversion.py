"""
Mean Reversion Strategy: Bollinger Band + RSI extremes.
Best in: SIDEWAYS, LOW_VOLATILITY regimes.
"""

from datetime import UTC, datetime

import numpy as np

from src.core.events import Signal, SignalSide

from .base import MarketState, StrategyBase


class MeanReversionStrategy(StrategyBase):
    """
    BUY when price touches lower Bollinger Band + RSI oversold.
    SELL when price touches upper Bollinger Band + RSI overbought.
    Target: revert to middle band (SMA). Stop: beyond outer band.
    """

    strategy_id = "mean_reversion"
    description = "Bollinger Band + RSI mean reversion"

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def warm(self, state: MarketState) -> bool:
        return len(state.bars) >= self.bb_period + 10

    def generate_signals(self, state: MarketState) -> list[Signal]:
        if not self.warm(state):
            return []

        closes = np.array([b.close for b in state.bars], dtype=float)
        price = state.latest_price or closes[-1]

        # Bollinger Bands
        window = closes[-self.bb_period :]
        sma = np.mean(window)
        std = np.std(window, ddof=1)
        upper = sma + self.bb_std * std
        lower = sma - self.bb_std * std

        # RSI
        rsi = self._calc_rsi(closes, self.rsi_period)

        # Bandwidth: only trade when bands are reasonably wide (avoid dead markets)
        bandwidth = (upper - lower) / (sma + 1e-12)
        if bandwidth < 0.005:  # bands too tight → no opportunity
            return []

        # BUY: price at/below lower band + RSI oversold
        if price <= lower and rsi <= self.rsi_oversold:
            # How far below band = strength of signal
            distance = (lower - price) / (std + 1e-12)
            score = min(1.0, 0.5 + distance * 0.2)
            return [
                Signal(
                    strategy_id=self.strategy_id,
                    symbol=state.symbol,
                    exchange=state.exchange,
                    side=SignalSide.BUY,
                    score=score,
                    portfolio_weight=0.12,
                    risk_level="NORMAL",
                    reason=f"mean_rev_buy rsi={rsi:.0f} bb_dist={distance:.2f}",
                    price=price,
                    stop_loss=round(lower - std, 2),  # stop beyond lower band
                    target=round(sma, 2),  # target: middle band
                    ts=datetime.now(UTC),
                    metadata={"rsi": rsi, "bollinger_lower": lower, "sma": sma},
                )
            ]

        # SELL: price at/above upper band + RSI overbought
        if price >= upper and rsi >= self.rsi_overbought:
            distance = (price - upper) / (std + 1e-12)
            score = min(1.0, 0.5 + distance * 0.2)
            return [
                Signal(
                    strategy_id=self.strategy_id,
                    symbol=state.symbol,
                    exchange=state.exchange,
                    side=SignalSide.SELL,
                    score=score,
                    portfolio_weight=0.12,
                    risk_level="NORMAL",
                    reason=f"mean_rev_sell rsi={rsi:.0f} bb_dist={distance:.2f}",
                    price=price,
                    stop_loss=round(upper + std, 2),
                    target=round(sma, 2),
                    ts=datetime.now(UTC),
                    metadata={"rsi": rsi, "bollinger_upper": upper, "sma": sma},
                )
            ]

        return []

    @staticmethod
    def _calc_rsi(closes, period=14):
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-period - 1 :])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss < 1e-12:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / (avg_loss + 1e-12)
        return float(100.0 - (100.0 / (1.0 + rs)))
