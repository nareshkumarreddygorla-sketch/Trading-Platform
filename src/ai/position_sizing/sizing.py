"""
Dynamic position sizing:
  f = clip(Kelly * confidence * regime_mult * drawdown_scale, 0, f_max)
Volatility targeting: notional = capital * sigma_target / sigma_forecast.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class SizingConfig:
    f_max: float = 0.10
    drawdown_scale_alpha: float = 0.5
    max_dd_pct: float = 10.0
    sigma_target_annual: float = 0.15


def kelly_binary(p_win: float, win_loss_ratio: float) -> float:
    """
    Kelly for binary outcome: f* = (p*b - q) / b, where b = win/loss ratio.
    f* = (p_win * win_loss_ratio - (1 - p_win)) / win_loss_ratio.
    """
    if win_loss_ratio <= 0:
        return 0.0
    q = 1.0 - p_win
    f = (p_win * win_loss_ratio - q) / win_loss_ratio
    return max(0.0, min(1.0, f))


def dynamic_position_fraction(
    p_win: float,
    win_loss_ratio: float,
    confidence: float,
    current_drawdown_pct: float,
    regime_multiplier: float = 1.0,
    config: Optional[SizingConfig] = None,
) -> float:
    """
    Position size as fraction of capital:
      f* = Kelly(p_win, win_loss_ratio)
      f  = clip(f* * confidence * regime_mult * drawdown_scale, 0, f_max)
    drawdown_scale = max(0, 1 - alpha * (current_dd_pct / max_dd_pct))
    """
    cfg = config or SizingConfig()
    f_star = kelly_binary(p_win, win_loss_ratio)
    drawdown_scale = max(0.0, 1.0 - cfg.drawdown_scale_alpha * (current_drawdown_pct / max(cfg.max_dd_pct, 0.01)))
    f = f_star * confidence * regime_multiplier * drawdown_scale
    return max(0.0, min(cfg.f_max, f))


def volatility_target_notional(
    capital: float,
    sigma_forecast: float,
    sigma_target_annual: float = 0.15,
    periods_per_year: int = 252,
) -> float:
    """
    Notional such that expected vol of portfolio = sigma_target.
    For single position: notional = capital * (sigma_target / sigma_forecast).
    sigma_forecast should be same horizon as target (e.g. daily).
    """
    if sigma_forecast <= 0:
        return 0.0
    sigma_target_daily = sigma_target_annual / (periods_per_year ** 0.5)
    return capital * (sigma_target_daily / (sigma_forecast + 1e-12))
