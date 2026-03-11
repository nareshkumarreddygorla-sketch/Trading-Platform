"""
Cross-asset features: index correlation, India VIX, USDINR impact, global spillover.
Requires external data (index series, VIX, FX); stubs when not available.
"""

import numpy as np

from src.core.events import Bar


def compute_index_correlation(
    asset_returns: np.ndarray,
    index_returns: np.ndarray,
    window: int = 5,
) -> float:
    if len(asset_returns) < window or len(index_returns) < window:
        return 0.0
    a = asset_returns[-window:]
    b = index_returns[-window:]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_cross_asset_features(
    bars: list[Bar],
    index_returns: np.ndarray | None = None,
    india_vix: float | None = None,
    usdinr_return: float | None = None,
    global_index_return: float | None = None,
) -> dict[str, float]:
    """
    Cross-asset features. Pass index returns (same length as bar count for correlation);
    VIX and FX are point-in-time (e.g. current value).
    """
    features: dict[str, float] = {}
    if not bars:
        return features

    close = np.array([b.close for b in bars], dtype=float)
    returns = np.diff(close) / (close[:-1] + 1e-12)

    if index_returns is not None:
        features["index_correlation_5d"] = compute_index_correlation(returns, index_returns, min(5, len(returns)))
    else:
        features["index_correlation_5d"] = 0.0

    features["india_vix"] = float(india_vix) if india_vix is not None else 0.0
    features["usdinr_impact"] = float(usdinr_return) if usdinr_return is not None else 0.0
    features["global_spillover"] = float(global_index_return) if global_index_return is not None else 0.0
    return features
