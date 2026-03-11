"""
Versioned feature specs for the feature store.
All features are documented and versioned for reproducibility.
"""

from dataclasses import dataclass
from enum import Enum


class FeatureGroup(str, Enum):
    PRICE = "price"
    MICROSTRUCTURE = "microstructure"
    REGIME = "regime"
    CROSS_ASSET = "cross_asset"
    INTRADAY_ALPHA = "intraday_alpha"


@dataclass
class FeatureSpec:
    name: str
    dtype: str
    version: str
    description: str
    group: FeatureGroup


# --- Price-based ---
PRICE_SPECS: list[FeatureSpec] = [
    FeatureSpec("returns_1m", "float", "v1", "1-minute return", FeatureGroup.PRICE),
    FeatureSpec("returns_5m", "float", "v1", "5-minute return", FeatureGroup.PRICE),
    FeatureSpec("returns_15m", "float", "v1", "15-minute return", FeatureGroup.PRICE),
    FeatureSpec("returns_1h", "float", "v1", "1-hour return", FeatureGroup.PRICE),
    FeatureSpec("rolling_vol_20", "float", "v1", "20-period rolling volatility", FeatureGroup.PRICE),
    FeatureSpec("rolling_vol_60", "float", "v1", "60-period rolling volatility", FeatureGroup.PRICE),
    FeatureSpec("atr_14", "float", "v1", "14-period ATR", FeatureGroup.PRICE),
    FeatureSpec("bb_width_20", "float", "v1", "Bollinger band width (20,2)", FeatureGroup.PRICE),
    FeatureSpec("momentum_5", "float", "v1", "5-period momentum score", FeatureGroup.PRICE),
    FeatureSpec("momentum_20", "float", "v1", "20-period momentum score", FeatureGroup.PRICE),
    FeatureSpec("zscore_20", "float", "v1", "Z-score mean reversion (20)", FeatureGroup.PRICE),
]

# --- Microstructure ---
MICROSTRUCTURE_SPECS: list[FeatureSpec] = [
    FeatureSpec("order_flow_imbalance", "float", "v1", "Bid vs ask volume imbalance", FeatureGroup.MICROSTRUCTURE),
    FeatureSpec("bid_ask_spread_bps", "float", "v1", "Bid-ask spread in bps", FeatureGroup.MICROSTRUCTURE),
    FeatureSpec("volume_delta", "float", "v1", "Buy volume - sell volume", FeatureGroup.MICROSTRUCTURE),
    FeatureSpec("vwap_deviation_bps", "float", "v1", "Price deviation from VWAP in bps", FeatureGroup.MICROSTRUCTURE),
    FeatureSpec("liquidity_pressure", "float", "v1", "Volume / depth proxy", FeatureGroup.MICROSTRUCTURE),
]

# --- Regime ---
REGIME_SPECS: list[FeatureSpec] = [
    FeatureSpec("vol_cluster_20", "float", "v1", "Volatility clustering (20)", FeatureGroup.REGIME),
    FeatureSpec("hurst_exponent", "float", "v1", "Hurst exponent (trend vs mean-reversion)", FeatureGroup.REGIME),
    FeatureSpec("trend_strength_index", "float", "v1", "ADX-like trend strength", FeatureGroup.REGIME),
    FeatureSpec("market_corr_rolling", "float", "v1", "Rolling correlation to index", FeatureGroup.REGIME),
    FeatureSpec("sector_dispersion", "float", "v1", "Cross-sectional sector dispersion", FeatureGroup.REGIME),
]

# --- Intraday alpha (blueprint §3) ---
INTRADAY_ALPHA_SPECS: list[FeatureSpec] = [
    FeatureSpec("ofi_5", "float", "v1", "Order flow imbalance 5 bars", FeatureGroup.INTRADAY_ALPHA),
    FeatureSpec("ofi_15", "float", "v1", "Order flow imbalance 15 bars", FeatureGroup.INTRADAY_ALPHA),
    FeatureSpec("ofi_30", "float", "v1", "Order flow imbalance 30 bars", FeatureGroup.INTRADAY_ALPHA),
    FeatureSpec(
        "poc_deviation_bps",
        "float",
        "v1",
        "Price deviation from session POC (volume profile)",
        FeatureGroup.INTRADAY_ALPHA,
    ),
    FeatureSpec(
        "value_area_upper_dev_bps", "float", "v1", "Deviation from value area upper (70%)", FeatureGroup.INTRADAY_ALPHA
    ),
    FeatureSpec(
        "value_area_lower_dev_bps", "float", "v1", "Deviation from value area lower (70%)", FeatureGroup.INTRADAY_ALPHA
    ),
    FeatureSpec(
        "minute_of_day_sin", "float", "v1", "Intraday seasonality sin(minute/390)", FeatureGroup.INTRADAY_ALPHA
    ),
    FeatureSpec(
        "minute_of_day_cos", "float", "v1", "Intraday seasonality cos(minute/390)", FeatureGroup.INTRADAY_ALPHA
    ),
    FeatureSpec(
        "vwap_dev_curvature", "float", "v1", "Second derivative of (p - VWAP) w.r.t. time", FeatureGroup.INTRADAY_ALPHA
    ),
    FeatureSpec(
        "liquidity_vacuum",
        "float",
        "v1",
        "Spread > k*median(spread) or vol < k*median(vol)",
        FeatureGroup.INTRADAY_ALPHA,
    ),
    FeatureSpec("vol_of_vol_20", "float", "v1", "Std of rolling 5-bar vol over 20 bars", FeatureGroup.INTRADAY_ALPHA),
    FeatureSpec(
        "microstructure_noise_resid",
        "float",
        "v1",
        "Residual from rolling MA on mid (noise filter)",
        FeatureGroup.INTRADAY_ALPHA,
    ),
    FeatureSpec(
        "spread_widening_signal",
        "float",
        "v1",
        "Predicted Δ spread from lagged spread/vol/OFI",
        FeatureGroup.INTRADAY_ALPHA,
    ),
    FeatureSpec(
        "cross_stock_dispersion",
        "float",
        "v1",
        "Std of returns across sector constituents",
        FeatureGroup.INTRADAY_ALPHA,
    ),
    FeatureSpec(
        "sector_momentum_rank", "float", "v1", "Rank of sector return over last N bars", FeatureGroup.INTRADAY_ALPHA
    ),
]

# --- Cross-asset ---
CROSS_ASSET_SPECS: list[FeatureSpec] = [
    FeatureSpec("index_correlation_5d", "float", "v1", "5-day correlation to index", FeatureGroup.CROSS_ASSET),
    FeatureSpec("india_vix", "float", "v1", "India VIX level", FeatureGroup.CROSS_ASSET),
    FeatureSpec("usdinr_impact", "float", "v1", "USDINR move impact proxy", FeatureGroup.CROSS_ASSET),
    FeatureSpec("global_spillover", "float", "v1", "Global index spillover factor", FeatureGroup.CROSS_ASSET),
]

FEATURE_SPECS: list[FeatureSpec] = (
    PRICE_SPECS + MICROSTRUCTURE_SPECS + REGIME_SPECS + INTRADAY_ALPHA_SPECS + CROSS_ASSET_SPECS
)

FEATURE_VERSION = "v1"
