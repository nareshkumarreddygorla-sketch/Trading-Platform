"""
Regime-related features: volatility clustering, Hurst exponent, trend strength,
market correlation, sector dispersion.
"""
from typing import Dict, List, Optional

import numpy as np

from src.core.events import Bar


def _to_close(bars: List[Bar]) -> np.ndarray:
    return np.array([b.close for b in bars], dtype=float)


def compute_vol_cluster(returns: np.ndarray, window: int = 20) -> float:
    """Autocorrelation of squared returns (volatility clustering)."""
    if len(returns) < window + 5:
        return 0.0
    sq = returns[-window:] ** 2
    if np.std(sq) < 1e-12:
        return 0.0
    return float(np.corrcoef(sq[:-1], sq[1:])[0, 1]) if len(sq) > 1 else 0.0


def compute_hurst(close: np.ndarray, max_lag: int = 20) -> float:
    """
    Simplified Hurst exponent (R/S). >0.5 trend, <0.5 mean-reversion.
    """
    if len(close) < max_lag * 2:
        return 0.5
    returns = np.diff(close) / (close[:-1] + 1e-12)
    n = len(returns)
    rs_vals = []
    for lag in range(2, min(max_lag, n // 2)):
        rs = []
        for start in range(0, n - lag):
            y = np.cumsum(returns[start : start + lag] - np.mean(returns[start : start + lag]))
            R = np.max(y) - np.min(y)
            S = np.std(returns[start : start + lag])
            if S > 1e-12:
                rs.append(R / S)
        if rs:
            rs_vals.append((lag, np.log(np.mean(rs))))
    if len(rs_vals) < 2:
        return 0.5
    lags = np.log([x[0] for x in rs_vals])
    rs_log = np.array([x[1] for x in rs_vals])
    slope = np.polyfit(lags, rs_log, 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


def compute_trend_strength_index(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """ADX-like: average of +DM and -DM contribution (simplified)."""
    if len(close) < period + 1:
        return 0.0
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    up = high[1:] - high[:-1]
    down = low[:-1] - low[1:]
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    atr = np.mean(tr[-period:])
    if atr < 1e-12:
        return 0.0
    di_plus = np.mean(plus_dm[-period:]) / atr
    di_minus = np.mean(minus_dm[-period:]) / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-12)
    return float(np.clip(dx, 0, 100))


def compute_regime_features(
    bars: List[Bar],
    index_returns: Optional[np.ndarray] = None,
    sector_returns: Optional[List[np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Compute regime-related features. index_returns and sector_returns optional
    (same length as bar series for correlation).
    """
    features: Dict[str, float] = {}
    if not bars or len(bars) < 30:
        return features

    close = _to_close(bars)
    high = np.array([b.high for b in bars], dtype=float)
    low = np.array([b.low for b in bars], dtype=float)
    returns = np.diff(close) / (close[:-1] + 1e-12)

    features["vol_cluster_20"] = compute_vol_cluster(returns, 20)
    features["hurst_exponent"] = compute_hurst(close, 20)
    features["trend_strength_index"] = compute_trend_strength_index(high, low, close, 14)

    if index_returns is not None and len(returns) == len(index_returns):
        if np.std(returns) > 1e-12 and np.std(index_returns) > 1e-12:
            features["market_corr_rolling"] = float(np.corrcoef(returns, index_returns)[0, 1])
        else:
            features["market_corr_rolling"] = 0.0
    else:
        features["market_corr_rolling"] = 0.0

    if sector_returns is not None and len(sector_returns) > 1:
        # Dispersion = std of cross-sectional returns
        last_returns = np.array([r[-1] if len(r) else 0 for r in sector_returns])
        features["sector_dispersion"] = float(np.std(last_returns))
    else:
        features["sector_dispersion"] = 0.0

    return features
