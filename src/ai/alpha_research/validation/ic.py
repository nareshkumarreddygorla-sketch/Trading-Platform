"""
Phase B: Information Coefficient and variants.
IC (rank), IC stability across time, IC stability across regimes, turnover-adjusted IC.
"""
from typing import List, Optional

import numpy as np


def _rank_corr_numpy(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation using numpy (fallback when scipy not available)."""
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 0 and np.std(ry) > 0 else 0.0


def ic_rank(signal: np.ndarray, forward_return: np.ndarray) -> float:
    """Spearman correlation between signal and forward return (rank IC)."""
    if len(signal) != len(forward_return) or len(signal) < 30:
        return 0.0
    mask = np.isfinite(signal) & np.isfinite(forward_return)
    if np.sum(mask) < 30:
        return 0.0
    x, y = signal[mask], forward_return[mask]
    try:
        from scipy.stats import spearmanr
        r, _ = spearmanr(x, y)
        return float(r) if np.isfinite(r) else 0.0
    except Exception:
        return _rank_corr_numpy(x, y)


def ic_stability_time(
    signal: np.ndarray,
    forward_return: np.ndarray,
    window: int = 20,
    step: int = 10,
) -> tuple[float, float]:
    """Rolling IC over time; return (mean_IC, std_IC). std_IC used for stability = 1/(1+std_IC)."""
    n = len(signal)
    if n < window or len(forward_return) != n:
        return 0.0, 1.0
    ics: List[float] = []
    for start in range(0, n - window, step):
        end = start + window
        ic = ic_rank(signal[start:end], forward_return[start:end])
        ics.append(ic)
    if not ics:
        return 0.0, 1.0
    arr = np.array(ics)
    return float(np.mean(arr)), float(np.std(arr)) if len(arr) > 1 else 0.0


def ic_stability_regime(
    signal: np.ndarray,
    forward_return: np.ndarray,
    regime_labels: np.ndarray,
) -> tuple[float, float]:
    """IC per regime; return (min_IC_over_regimes, mean_IC). Regime_robustness = min/mean."""
    if len(signal) != len(forward_return) or len(signal) != len(regime_labels):
        return 0.0, 0.0
    regimes = np.unique(regime_labels)
    ics = []
    for r in regimes:
        mask = regime_labels == r
        if np.sum(mask) < 30:
            continue
        ic = ic_rank(signal[mask], forward_return[mask])
        ics.append(ic)
    if not ics:
        return 0.0, 0.0
    return float(min(ics)), float(np.mean(ics))


def turnover_adjusted_ic(
    ic_raw: float,
    turnover: float,
    lambda_penalty: float = 0.1,
) -> float:
    """IC_adj = IC_raw * (1 - lambda * turnover). turnover in [0,1] per period."""
    return ic_raw * max(0.0, 1.0 - lambda_penalty * min(1.0, turnover))
