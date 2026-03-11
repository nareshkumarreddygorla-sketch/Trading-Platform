"""Tests for AI feature engineering pipeline."""

from datetime import UTC, datetime, timedelta

import pytest

from src.ai.feature_engineering import compute_price_features, compute_regime_features
from src.ai.feature_engineering.specs import FEATURE_SPECS, FeatureGroup
from src.core.events import Bar, Exchange


@pytest.fixture
def sample_bars():
    bars = []
    base = 100.0
    ts = datetime.now(UTC) - timedelta(days=100)
    for i in range(100):
        o = base
        base = base * (1 + (i % 5 - 2) * 0.005)
        h = max(o, base) * 1.01
        l = min(o, base) * 0.99
        bars.append(
            Bar(
                symbol="RELIANCE",
                exchange=Exchange.NSE,
                interval="1d",
                open=o,
                high=h,
                low=l,
                close=base,
                volume=1_000_000 + i * 1000,
                ts=ts + timedelta(days=i),
                source="test",
            )
        )
    return bars


def test_price_features(sample_bars):
    features = compute_price_features(sample_bars)
    assert "returns_1m" in features or "atr_14" in features
    assert "rolling_vol_20" in features
    assert "zscore_20" in features


def test_regime_features(sample_bars):
    features = compute_regime_features(sample_bars)
    assert "vol_cluster_20" in features
    assert "hurst_exponent" in features
    assert "trend_strength_index" in features


def test_feature_specs():
    assert len(FEATURE_SPECS) > 0
    groups = {s.group for s in FEATURE_SPECS}
    assert FeatureGroup.PRICE in groups
    assert FeatureGroup.REGIME in groups
