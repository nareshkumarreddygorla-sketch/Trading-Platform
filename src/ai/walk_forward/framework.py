"""
Walk-forward validation framework.
Rolling window: train [t-T, t), test [t, t+O).
Stability score: S = (mean_Sharpe / (std_Sharpe + ε)) * (1 - mean_dd/dd_limit) * I(frac_positive_Sharpe >= 0.7)
Replacement rule: replace only if Sharpe improves, drawdown not worse, S above threshold, N consecutive positive windows.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class WalkForwardConfig:
    train_bars: int = 60 * 78  # e.g. 60 days of 78 1m bars per day
    test_bars: int = 10 * 78
    step_bars: int = 10 * 78
    expanding: bool = False
    dd_limit_pct: float = 10.0
    min_frac_positive_sharpe: float = 0.7
    min_consecutive_positive_windows: int = 3
    stability_min: float = 0.3


@dataclass
class WalkForwardResult:
    window_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    sharpe: float
    max_drawdown_pct: float
    mean_return: float
    n_trades: int


def stability_score(
    sharpes: list[float],
    max_drawdowns_pct: list[float],
    dd_limit_pct: float,
    min_frac_positive: float,
) -> float:
    """
    S = (mean_Sharpe / (std_Sharpe + ε)) * (1 - mean_dd/dd_limit) * I(frac_positive_Sharpe >= min_frac_positive).
    Returns 0 if frac_positive < min_frac_positive.
    """
    if not sharpes:
        return 0.0
    arr = np.array(sharpes)
    frac_pos = np.mean(arr > 0)
    if frac_pos < min_frac_positive:
        return 0.0
    mean_s = np.mean(arr)
    std_s = np.std(arr)
    if std_s < 1e-12:
        std_s = 1e-12
    term1 = mean_s / std_s
    if not max_drawdowns_pct:
        term2 = 1.0
    else:
        mean_dd = np.mean(max_drawdowns_pct)
        term2 = max(0.0, 1.0 - mean_dd / (dd_limit_pct + 1e-12))
    return float(term1 * term2)


def replacement_rule(
    current_sharpe: float,
    current_dd_pct: float,
    candidate_sharpe: float,
    candidate_dd_pct: float,
    candidate_stability: float,
    candidate_consecutive_positive: int,
    config: WalkForwardConfig | None = None,
) -> bool:
    """
    Replace current model with candidate only if:
    - candidate_sharpe > current_sharpe
    - candidate_dd_pct <= current_dd_pct (or not worse by more than small delta)
    - candidate_stability >= stability_min
    - candidate_consecutive_positive >= min_consecutive_positive_windows
    """
    cfg = config or WalkForwardConfig()
    if candidate_sharpe <= current_sharpe:
        return False
    if candidate_dd_pct > current_dd_pct * 1.05:
        return False
    if candidate_stability < cfg.stability_min:
        return False
    if candidate_consecutive_positive < cfg.min_consecutive_positive_windows:
        return False
    return True
