"""Classical technical strategies: EMA, MACD, RSI, etc. Plugin-based."""

from datetime import UTC, datetime

import numpy as np

from src.core.events import Signal, SignalSide

from .base import MarketState, StrategyBase


def _sma(series, period: int):
    return series.rolling(period).mean()


def _ema(series, period: int):
    return series.ewm(span=period, adjust=False).mean()


class EMACrossoverStrategy(StrategyBase):
    """EMA fast cross above/below EMA slow -> BUY/SELL. Scored by distance."""

    strategy_id = "ema_crossover"
    description = "EMA fast/slow crossover"

    def __init__(self, fast: int = 9, slow: int = 21):
        self.fast = fast
        self.slow = slow

    def warm(self, state: MarketState) -> bool:
        return len(state.bars) >= self.slow + 5

    def generate_signals(self, state: MarketState) -> list[Signal]:
        if not self.warm(state):
            return []
        closes = [b.close for b in state.bars]
        import pandas as pd

        s = pd.Series(closes)
        ema_f = _ema(s, self.fast).iloc[-1]
        ema_s = _ema(s, self.slow).iloc[-1]
        price = state.latest_price
        if price > ema_f > ema_s:
            score = min(1.0, (price - ema_s) / ema_s * 5) if ema_s else 0.5
            return [
                Signal(
                    strategy_id=self.strategy_id,
                    symbol=state.symbol,
                    exchange=state.exchange,
                    side=SignalSide.BUY,
                    score=max(0.3, score),
                    portfolio_weight=0.2,
                    risk_level="NORMAL",
                    reason="ema_cross_up",
                    price=price,
                    stop_loss=round(price * 0.99, 2),
                    target=round(price * 1.02, 2),
                    ts=datetime.now(UTC),
                )
            ]
        if price < ema_f < ema_s:
            score = min(1.0, (ema_s - price) / ema_s * 5) if ema_s else 0.5
            return [
                Signal(
                    strategy_id=self.strategy_id,
                    symbol=state.symbol,
                    exchange=state.exchange,
                    side=SignalSide.SELL,
                    score=max(0.3, score),
                    portfolio_weight=0.2,
                    risk_level="NORMAL",
                    reason="ema_cross_down",
                    price=price,
                    stop_loss=round(price * 1.01, 2),
                    target=round(price * 0.98, 2),
                    ts=datetime.now(UTC),
                )
            ]
        return []


class MACDStrategy(StrategyBase):
    """MACD line cross signal line -> momentum signal."""

    strategy_id = "macd"
    description = "MACD crossover"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def warm(self, state: MarketState) -> bool:
        return len(state.bars) >= self.slow + self.signal + 5

    def generate_signals(self, state: MarketState) -> list[Signal]:
        if not self.warm(state):
            return []
        closes = [b.close for b in state.bars]
        import pandas as pd

        s = pd.Series(closes)
        ema_f = _ema(s, self.fast)
        ema_s = _ema(s, self.slow)
        macd_line = ema_f - ema_s
        signal_line = _ema(macd_line, self.signal)
        if len(macd_line) < 2 or len(signal_line) < 2:
            return []
        macd = macd_line.iloc[-1]
        sig = signal_line.iloc[-1]
        price = state.latest_price
        if macd > sig and macd_line.iloc[-2] <= signal_line.iloc[-2]:
            return [
                Signal(
                    strategy_id=self.strategy_id,
                    symbol=state.symbol,
                    exchange=state.exchange,
                    side=SignalSide.BUY,
                    score=max(0.3, min(1.0, (macd - sig) / (abs(sig) + 1e-8) * 2)),
                    portfolio_weight=0.15,
                    risk_level="NORMAL",
                    reason="macd_cross_up",
                    price=price,
                    stop_loss=round(price * 0.985, 2),
                    target=round(price * 1.025, 2),
                    ts=datetime.now(UTC),
                )
            ]
        if macd < sig and macd_line.iloc[-2] >= signal_line.iloc[-2]:
            return [
                Signal(
                    strategy_id=self.strategy_id,
                    symbol=state.symbol,
                    exchange=state.exchange,
                    side=SignalSide.SELL,
                    score=max(0.3, min(1.0, (sig - macd) / (abs(sig) + 1e-8) * 2)),
                    portfolio_weight=0.15,
                    risk_level="NORMAL",
                    reason="macd_cross_down",
                    price=price,
                    stop_loss=round(price * 1.015, 2),
                    target=round(price * 0.975, 2),
                    ts=datetime.now(UTC),
                )
            ]
        return []


def _rsi(series, period: int = 14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


class RSIStrategy(StrategyBase):
    """RSI oversold (< 30) BUY, overbought (> 70) SELL."""

    strategy_id = "rsi"
    description = "RSI oversold/overbought"

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def warm(self, state: MarketState) -> bool:
        return len(state.bars) >= self.period + 10

    def generate_signals(self, state: MarketState) -> list[Signal]:
        if not self.warm(state):
            return []
        closes = [b.close for b in state.bars]
        import pandas as pd

        s = pd.Series(closes)
        rsi = _rsi(s, self.period).iloc[-1]
        if np.isnan(rsi):
            return []
        price = state.latest_price
        if rsi < self.oversold:
            score = (self.oversold - rsi) / self.oversold
            return [
                Signal(
                    strategy_id=self.strategy_id,
                    symbol=state.symbol,
                    exchange=state.exchange,
                    side=SignalSide.BUY,
                    score=min(1.0, score),
                    portfolio_weight=0.15,
                    risk_level="NORMAL",
                    reason="rsi_oversold",
                    price=price,
                    stop_loss=round(price * 0.985, 2),
                    target=round(price * 1.03, 2),
                    ts=datetime.now(UTC),
                )
            ]
        if rsi > self.overbought:
            score = (rsi - self.overbought) / (100 - self.overbought)
            return [
                Signal(
                    strategy_id=self.strategy_id,
                    symbol=state.symbol,
                    exchange=state.exchange,
                    side=SignalSide.SELL,
                    score=min(1.0, score),
                    portfolio_weight=0.15,
                    risk_level="NORMAL",
                    reason="rsi_overbought",
                    price=price,
                    stop_loss=round(price * 1.015, 2),
                    target=round(price * 0.97, 2),
                    ts=datetime.now(UTC),
                )
            ]
        return []
