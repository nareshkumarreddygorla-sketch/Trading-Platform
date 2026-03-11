"""
Risk metrics: VaR, CVaR, max drawdown, Sharpe, Kelly, Optimal f.
Used for portfolio risk and position sizing.
"""
from dataclasses import dataclass
from typing import List

import numpy as np

try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# Standard normal z-values for common confidence levels (fallback when scipy unavailable)
_Z_TABLE = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}
_PHI_TABLE = {0.90: 0.1755, 0.95: 0.1031, 0.99: 0.0267}  # pdf at z


@dataclass
class RiskMetrics:
    """Computed risk metrics for a series of returns (e.g. strategy or portfolio)."""
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR (expected shortfall) 95%
    max_drawdown: float
    max_drawdown_duration: int  # bars
    sharpe: float  # annualized if period is daily
    kelly_fraction: float  # optimal fraction of capital
    optimal_f: float  # optional


def var_parametric(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Parametric (normal) VaR."""
    if len(returns) < 2:
        return 0.0
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    if sigma < 1e-12:
        return 0.0
    if _HAS_SCIPY:
        z = _scipy_stats.norm.ppf(1 - confidence)
    else:
        z = -_Z_TABLE.get(confidence, 1.6449)  # negative because ppf(0.05) < 0
    return -(mu + z * sigma)


def cvar_parametric(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Conditional VaR (expected shortfall) — parametric (normal) closed form."""
    if len(returns) < 2:
        return 0.0
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    if sigma < 1e-12:
        return 0.0
    if _HAS_SCIPY:
        z = _scipy_stats.norm.ppf(1 - confidence)
        phi = _scipy_stats.norm.pdf(z)
    else:
        z = -_Z_TABLE.get(confidence, 1.6449)
        phi = _PHI_TABLE.get(confidence, 0.1031)
    # Closed-form CVaR for normal: ES = -mu + sigma * phi(z) / (1 - confidence)
    return float(-mu + sigma * phi / (1 - confidence))


def max_drawdown(prices: np.ndarray) -> tuple[float, int]:
    """Return (max_dd_pct, duration_in_bars). Duration = longest consecutive underwater stretch."""
    if len(prices) < 2:
        return 0.0, 0
    cummax = np.maximum.accumulate(prices)
    dd = (prices - cummax) / (cummax + 1e-12)
    md = float(np.min(dd))
    # Duration: longest consecutive run of underwater bars (prices < cummax)
    underwater = prices < cummax
    max_run = 0
    current_run = 0
    for uw in underwater:
        if uw:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0
    return md, max_run


def sharpe_annual(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized Sharpe (risk-free = 0)."""
    if len(returns) < 2 or np.std(returns, ddof=1) < 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * np.mean(returns) / np.std(returns, ddof=1))


def kelly_fraction(returns: np.ndarray, fraction: float = 0.25) -> float:
    """Kelly criterion: f* = (p*b - q)/b. Simplified: use mean/sigma^2 capped."""
    if len(returns) < 2:
        return 0.0
    mu = np.mean(returns)
    var = np.var(returns)
    if var < 1e-12:
        return 0.0
    f = mu / (var + 1e-12)
    return float(np.clip(f * fraction, 0.0, 1.0))  # half-Kelly or less


def compute_risk_metrics(returns: np.ndarray, confidence: float = 0.95) -> RiskMetrics:
    """Compute full risk metrics from return series."""
    if len(returns) < 2:
        return RiskMetrics(
            var_95=0.0, cvar_95=0.0, max_drawdown=0.0, max_drawdown_duration=0,
            sharpe=0.0, kelly_fraction=0.0, optimal_f=0.0,
        )
    # Filter NaN values before cumprod to prevent propagation
    clean_returns = returns[~np.isnan(returns)]
    if len(clean_returns) < 2:
        return RiskMetrics(
            var_95=0.0, cvar_95=0.0, max_drawdown=0.0, max_drawdown_duration=0,
            sharpe=0.0, kelly_fraction=0.0, optimal_f=0.0,
        )
    prices = np.cumprod(1 + clean_returns)
    md, duration = max_drawdown(prices)
    return RiskMetrics(
        var_95=var_parametric(clean_returns, confidence),
        cvar_95=cvar_parametric(clean_returns, confidence),
        max_drawdown=md,
        max_drawdown_duration=duration,
        sharpe=sharpe_annual(clean_returns),
        kelly_fraction=kelly_fraction(clean_returns),
        optimal_f=kelly_fraction(clean_returns, 1.0),
    )
