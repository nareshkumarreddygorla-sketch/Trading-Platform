"""
Backtesting engine: historical bars + strategy -> equity curve, metrics.
Simulates slippage, latency, fees (broker, exchange, tax).
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from src.core.events import Bar, Exchange, Signal, SignalSide
from src.strategy_engine.base import MarketState, StrategyBase
from .metrics import BacktestMetrics, compute_backtest_metrics
from .slippage import SlippageModel
from .fill_model import FillModel, FillModelConfig


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_pct: float = 0.05  # broker
    slippage_bps: float = 5.0
    latency_bars: int = 1  # fill at bar i+latency_bars (no same-bar fill)
    spread_bps: float = 3.0
    max_volume_participation_pct: float = 10.0
    tax_rate_pct: float = 0.0  # short-term cap gains if applicable


@dataclass
class BacktestResult:
    equity_curve: List[float] = field(default_factory=list)
    metrics: Optional[BacktestMetrics] = None
    trades: List[dict] = field(default_factory=list)
    config: Optional[BacktestConfig] = None


class BacktestEngine:
    """
    Run strategy over historical bars; record fills with slippage/fees; produce equity curve.
    Walk-forward: slice bars into train/test windows (caller can loop).
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.slippage = SlippageModel(self.config.slippage_bps)
        self.fill_model = FillModel(FillModelConfig(
            latency_bars=self.config.latency_bars,
            slippage_bps=self.config.slippage_bps,
            spread_bps=self.config.spread_bps,
            max_volume_participation_pct=self.config.max_volume_participation_pct,
            commission_pct=self.config.commission_pct,
        ))

    def run(
        self,
        strategy: StrategyBase,
        bars: List[Bar],
        symbol: str,
        exchange: Exchange = Exchange.NSE,
    ) -> BacktestResult:
        if not bars:
            return BacktestResult(config=self.config)
        equity = self.config.initial_capital
        equity_curve = [equity]
        trades: List[dict] = []
        position = 0.0
        entry_price = 0.0
        # Rolling window of bars for strategy
        for i in range(len(bars)):
            window = bars[max(0, i - 100) : i + 1]
            if len(window) < 20:
                equity_curve.append(equity)
                continue
            latest = window[-1]
            state = MarketState(
                symbol=symbol,
                exchange=exchange,
                bars=window,
                latest_price=latest.close,
                volume=latest.volume,
            )
            if not strategy.warm(state):
                equity_curve.append(equity)
                continue
            signals = strategy.generate_signals(state)
            # Execute first signal only; fill at i+latency_bars (no same-bar lookahead)
            for sig in signals[:1]:
                if sig.side == SignalSide.BUY and position <= 0:
                    req_qty = (equity * 0.05) / (sig.price or latest.close)
                    fill_bar, fill_price, fill_qty, commission = self.fill_model.execute_at_bar_index(
                        i, bars, "BUY", req_qty, sig.price or latest.close
                    )
                    if fill_bar is not None and fill_qty > 0:
                        cost = fill_price * fill_qty
                        position = fill_qty
                        entry_price = fill_price
                        equity -= cost + commission
                        trades.append({"ts": fill_bar.ts, "side": "BUY", "price": fill_price, "qty": position, "equity": equity})
                elif sig.side == SignalSide.SELL and position > 0:
                    fill_bar, fill_price, fill_qty, commission = self.fill_model.execute_at_bar_index(
                        i, bars, "SELL", position, latest.close
                    )
                    if fill_bar is not None and fill_qty > 0:
                        sale_proceeds = fill_price * fill_qty
                        equity += sale_proceeds - commission
                        pnl = (fill_price - entry_price) * fill_qty
                        trades.append({"ts": fill_bar.ts, "side": "SELL", "price": fill_price, "pnl": pnl, "equity": equity})
                        position = position - fill_qty
                        if position <= 0:
                            entry_price = 0.0
            equity_curve.append(equity + (position * latest.close if position > 0 else 0))
        metrics = compute_backtest_metrics(equity_curve, self.config.initial_capital)
        return BacktestResult(
            equity_curve=equity_curve,
            metrics=metrics,
            trades=trades,
            config=self.config,
        )
