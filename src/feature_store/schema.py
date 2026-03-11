"""Feature store schema: time-series features for ML, versioned."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class FeatureSpec:
    """Spec for one feature: name, dtype, version."""

    name: str
    dtype: str  # float, int, category
    version: str = "v1"
    description: str = ""


@dataclass
class FeatureVector:
    """Single timestamp feature vector."""

    symbol: str
    ts: datetime
    features: dict[str, Any]


def default_feature_specs() -> list[FeatureSpec]:
    """Common technical features for ML strategies."""
    return [
        FeatureSpec("returns_1d", "float", "v1", "1-day return"),
        FeatureSpec("returns_5d", "float", "v1", "5-day return"),
        FeatureSpec("volatility_20d", "float", "v1", "20-day vol"),
        FeatureSpec("rsi_14", "float", "v1", "RSI 14"),
        FeatureSpec("macd_hist", "float", "v1", "MACD histogram"),
        FeatureSpec("volume_ratio", "float", "v1", "Volume / MA(20)"),
    ]
