"""
Risk-adjusted objective: E[R] - λ1*DD - λ2*Turnover - λ3*σ.
Approximations for training (custom loss, Sharpe-like).
"""
from .risk_adjusted import risk_adjusted_reward, sharpe_like_score

__all__ = ["risk_adjusted_reward", "sharpe_like_score"]
