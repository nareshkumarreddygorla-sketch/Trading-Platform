"""
Objective: J = E[R] - λ1*DD - λ2*Turnover - λ3*σ.
Approximation for training: reward per sample = r - λ1*dd_contrib - λ2*turnover_contrib - λ3*vol_contrib.
Sharpe-like: mean_ret / (std_ret + ε).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ObjectiveConfig:
    lambda_drawdown: float = 0.1
    lambda_turnover: float = 0.05
    lambda_vol: float = 0.1
    eps: float = 1e-8


def risk_adjusted_reward(
    returns: np.ndarray,
    drawdown_contrib: np.ndarray | None = None,
    turnover: np.ndarray | None = None,
    config: ObjectiveConfig | None = None,
) -> float:
    """
    J = mean(returns) - λ1 * mean(drawdown_contrib) - λ2 * mean(turnover) - λ3 * std(returns).
    """
    cfg = config or ObjectiveConfig()
    j = float(np.mean(returns))
    if drawdown_contrib is not None and len(drawdown_contrib):
        j -= cfg.lambda_drawdown * float(np.mean(drawdown_contrib))
    if turnover is not None and len(turnover):
        j -= cfg.lambda_turnover * float(np.mean(turnover))
    j -= cfg.lambda_vol * (float(np.std(returns)) + cfg.eps)
    return j


def sharpe_like_score(returns: np.ndarray, eps: float = 1e-8) -> float:
    """mean(returns) / (std(returns) + eps). Use for validation or hyperparameter search."""
    r = np.asarray(returns)
    if len(r) < 2:
        return 0.0
    return float(np.mean(r) / (np.std(r) + eps))
