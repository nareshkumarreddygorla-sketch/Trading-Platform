"""
Phase A: Systematic transformation templates.
Price, microstructure, cross-sectional, regime-conditioned, nonlinear.
Each template has a name and params (e.g. windows); generator expands within bounds.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class TransformFamily(str, Enum):
    PRICE = "price"
    MICROSTRUCTURE = "microstructure"
    CROSS_SECTIONAL = "cross_sectional"
    REGIME_CONDITIONED = "regime_conditioned"
    NONLINEAR = "nonlinear"


@dataclass
class TransformTemplate:
    name: str
    family: TransformFamily
    description: str
    param_ranges: Dict[str, List[Any]]  # e.g. {"window": [5, 10, 20], "lag": [1, 2]}
    max_combinations: int = 10  # cap expansions per template


# Pre-filter: univariate IC threshold to avoid expensive backtest on useless candidates
MIN_UNIVARIATE_IC = 0.01

# --- Price-based ---
PRICE_TEMPLATES = [
    TransformTemplate("vol_norm_return", TransformFamily.PRICE, "return / rolling_vol", {"window": [10, 20, 40]}, 3),
    TransformTemplate("momentum_multi_horizon", TransformFamily.PRICE, "price / price[t-h] - 1", {"horizon": [5, 10, 20]}, 3),
    TransformTemplate("mean_reversion_zscore", TransformFamily.PRICE, "zscore(price, window)", {"window": [10, 20, 40]}, 3),
    TransformTemplate("range_compression", TransformFamily.PRICE, "range(high-low) / ATR", {"atr_window": [14, 20]}, 2),
    TransformTemplate("breakout_pressure", TransformFamily.PRICE, "(close - high_N) / ATR", {"N": [10, 20]}, 2),
    TransformTemplate("lag_interaction", TransformFamily.PRICE, "return[t] * return[t-k]", {"lag": [1, 2, 3]}, 3),
]

# --- Microstructure ---
MICRO_TEMPLATES = [
    TransformTemplate("ofi_variant", TransformFamily.MICROSTRUCTURE, "(buy_vol - sell_vol) / (buy_vol + sell_vol)", {"window": [5, 15]}, 2),
    TransformTemplate("spread_accel", TransformFamily.MICROSTRUCTURE, "diff(spread_bps, 2)", {}, 1),
    TransformTemplate("volume_cluster", TransformFamily.MICROSTRUCTURE, "volume / rolling_median(volume)", {"window": [20]}, 1),
    TransformTemplate("quote_imbalance_decay", TransformFamily.MICROSTRUCTURE, "quote_imbalance * exp(-decay*t)", {"decay": [0.1, 0.2]}, 2),
]

# --- Cross-sectional ---
CROSS_TEMPLATES = [
    TransformTemplate("sector_relative_strength", TransformFamily.CROSS_SECTIONAL, "symbol_return - sector_return", {"window": [5, 20]}, 2),
    TransformTemplate("dispersion_divergence", TransformFamily.CROSS_SECTIONAL, "std(cross_sectional_returns)", {"window": [20]}, 1),
    TransformTemplate("breadth_thrust", TransformFamily.CROSS_SECTIONAL, "pct_advancing - 0.5", {"window": [5, 10]}, 2),
]

# --- Regime-conditioned ---
REGIME_TEMPLATES = [
    TransformTemplate("feature_high_vol", TransformFamily.REGIME_CONDITIONED, "feature * I(vol_regime=high)", {"feature_ref": ["momentum", "zscore"]}, 2),
    TransformTemplate("feature_low_liq", TransformFamily.REGIME_CONDITIONED, "feature * I(liquidity=low)", {"feature_ref": ["ofi"]}, 1),
]

# --- Nonlinear ---
NONLINEAR_TEMPLATES = [
    TransformTemplate("rank_transform", TransformFamily.NONLINEAR, "rank(feature) / N", {"window": [20, 60]}, 2),
    TransformTemplate("percentile_transform", TransformFamily.NONLINEAR, "percentile_rank(feature, window)", {"window": [20, 40]}, 2),
    TransformTemplate("cross_product", TransformFamily.NONLINEAR, "feature_a * feature_b", {"pair": ["mom_vol", "ofi_zscore"]}, 2),
]

def transform_templates() -> List[TransformTemplate]:
    return (
        PRICE_TEMPLATES
        + MICRO_TEMPLATES
        + CROSS_TEMPLATES
        + REGIME_TEMPLATES
        + NONLINEAR_TEMPLATES
    )
