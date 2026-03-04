"""Backtest performance metrics: CAGR, Sharpe, max DD, win rate, etc."""
from dataclasses import dataclass
from typing import List

import numpy as np

from src.risk_engine.metrics import RiskMetrics, compute_risk_metrics


@dataclass
class BacktestMetrics:
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    risk_metrics: RiskMetrics


def compute_backtest_metrics(equity_curve: List[float], initial_capital: float) -> BacktestMetrics:
    if not equity_curve or initial_capital <= 0:
        return BacktestMetrics(
            total_return_pct=0.0, cagr_pct=0.0, sharpe=0.0, max_drawdown_pct=0.0,
            win_rate_pct=0.0, num_trades=0,
            risk_metrics=RiskMetrics(0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0),
        )
    arr = np.array(equity_curve)
    total_return_pct = (arr[-1] - initial_capital) / initial_capital * 100
    n = len(arr)
    # Assume daily bars for CAGR
    years = n / 252.0 if n >= 252 else n / 365.0
    cagr = (arr[-1] / initial_capital) ** (1 / max(years, 0.01)) - 1
    cagr_pct = cagr * 100
    returns = np.diff(arr) / (arr[:-1] + 1e-12)
    rm = compute_risk_metrics(returns)
    # Win rate from returns
    wins = np.sum(returns > 0)
    win_rate_pct = wins / len(returns) * 100 if len(returns) else 0
    return BacktestMetrics(
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        sharpe=rm.sharpe,
        max_drawdown_pct=abs(rm.max_drawdown) * 100,
        win_rate_pct=win_rate_pct,
        num_trades=0,  # set by engine from trades list
        risk_metrics=rm,
    )
