"""Risk parity and Kelly-adjusted strategy weights."""
from typing import Dict, List

import numpy as np


def compute_risk_parity_weights(returns_by_strategy: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Inverse volatility weighting (risk parity): weight_i ∝ 1/vol_i.
    """
    vols = {}
    for sid, returns in returns_by_strategy.items():
        if len(returns) < 2:
            vols[sid] = 1.0
        else:
            vols[sid] = max(np.std(returns), 1e-12)
    inv_vol = {sid: 1.0 / v for sid, v in vols.items()}
    total = sum(inv_vol.values())
    if total < 1e-12:
        n = len(inv_vol)
        return {sid: 1.0 / n for sid in inv_vol}
    return {sid: inv_vol[sid] / total for sid in inv_vol}


def compute_kelly_weights(
    strategy_params: Dict[str, tuple],  # strategy_id -> (sharpe, win_rate)
    fraction: float = 0.25,
) -> Dict[str, float]:
    """
    Kelly-adjusted weights from Sharpe and win rate. fraction = half-Kelly etc.
    Simplified: weight_i ∝ max(0, sharpe_i) * fraction, then normalize.
    """
    weights = {}
    for sid, (sharpe, win_rate) in strategy_params.items():
        if sharpe > 0:
            kelly = sharpe * fraction
            weights[sid] = min(kelly, 1.0)
        else:
            weights[sid] = 0.0
    total = sum(weights.values())
    if total < 1e-12:
        n = len(weights)
        return {sid: 1.0 / n for sid in weights}
    return {sid: w / total for sid, w in weights.items()}
