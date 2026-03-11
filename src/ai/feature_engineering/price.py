"""
Price-based features: returns, volatility, ATR, Bollinger width, momentum, z-score.
Input: OHLCV series (close, high, low, volume); periods configurable.
"""
from typing import Dict, List, Optional

import numpy as np

from src.core.events import Bar


def _to_series(bars: List[Bar], field: str) -> np.ndarray:
    return np.array([getattr(b, field) for b in bars], dtype=float)


def compute_returns(closes: np.ndarray, periods: List[int]) -> Dict[str, float]:
    out = {}
    n = len(closes)
    for p in periods:
        if n > p and closes[n - 1 - p] != 0:
            out[f"returns_{p}m"] = (closes[-1] - closes[-1 - p]) / (closes[-1 - p] + 1e-12)
        else:
            out[f"returns_{p}m"] = 0.0
    return out


def compute_rolling_volatility(closes: np.ndarray, windows: List[int]) -> Dict[str, float]:
    returns = np.diff(closes) / (closes[:-1] + 1e-12)
    out = {}
    for w in windows:
        if len(returns) >= w:
            out[f"rolling_vol_{w}"] = float(np.std(returns[-w:]))
        else:
            out[f"rolling_vol_{w}"] = 0.0
    return out


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    if len(close) < period + 1:
        return 0.0
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    return float(np.mean(tr[-period:]))


def compute_bb_width(close: np.ndarray, period: int = 20, k: float = 2.0) -> float:
    if len(close) < period:
        return 0.0
    ma = np.mean(close[-period:])
    std = np.std(close[-period:])
    if std < 1e-12:
        return 0.0
    upper = ma + k * std
    lower = ma - k * std
    return float((upper - lower) / (ma + 1e-12))


def compute_momentum(close: np.ndarray, periods: List[int]) -> Dict[str, float]:
    out = {}
    for p in periods:
        if len(close) > p and close[-1 - p] != 0:
            out[f"momentum_{p}"] = (close[-1] - close[-1 - p]) / (close[-1 - p] + 1e-12)
        else:
            out[f"momentum_{p}"] = 0.0
    return out


def compute_zscore(close: np.ndarray, period: int = 20) -> float:
    if len(close) < period:
        return 0.0
    window = close[-period:]
    mean, std = np.mean(window), np.std(window)
    if std < 1e-12:
        return 0.0
    return float((close[-1] - mean) / (std + 1e-12))


def compute_price_features(
    bars: List[Bar],
    return_periods: Optional[List[int]] = None,
    vol_windows: Optional[List[int]] = None,
    momentum_periods: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute all price-based features. Periods are in number of bars (e.g. 1m=1, 5m=5;
    mapping to real time depends on bar interval).
    """
    return_periods = return_periods or [1, 5, 15, 60]
    vol_windows = vol_windows or [20, 60]
    momentum_periods = momentum_periods or [5, 20]

    if not bars:
        return {}
    close = _to_series(bars, "close")
    high = _to_series(bars, "high")
    low = _to_series(bars, "low")

    features: Dict[str, float] = {}
    features.update(compute_returns(close, return_periods))
    features.update(compute_rolling_volatility(close, vol_windows))
    features["atr_14"] = compute_atr(high, low, close, 14)
    features["bb_width_20"] = compute_bb_width(close, 20, 2.0)
    features.update(compute_momentum(close, momentum_periods))
    features["zscore_20"] = compute_zscore(close, 20)
    return features
