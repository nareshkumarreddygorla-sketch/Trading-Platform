"""Tests for regime classifier."""

import numpy as np

from src.ai.regime import RegimeClassifier, RegimeLabel


def test_regime_classifier_trending():
    clf = RegimeClassifier(vol_high_percentile=80, vol_low_percentile=20)
    for _ in range(50):
        clf.update_vol_history(0.01)
    returns = np.random.randn(20) * 0.01
    vol = 0.01
    trend = 0.5
    r = clf.classify(returns, vol, trend)
    assert r.label in (RegimeLabel.TRENDING_UP, RegimeLabel.SIDEWAYS, RegimeLabel.LOW_VOLATILITY)
    assert 0 <= r.confidence <= 1


def test_regime_classifier_crisis():
    clf = RegimeClassifier(crisis_vol_multiplier=2.0)
    for _ in range(50):
        clf.update_vol_history(0.01)
    r = clf.classify(np.random.randn(20), volatility=0.05, trend_strength=0.0)
    # 0.05 >> 0.01*2 => crisis possible
    assert r.label in (RegimeLabel.CRISIS, RegimeLabel.HIGH_VOLATILITY, RegimeLabel.SIDEWAYS)


def test_strategies_for_regime():
    clf = RegimeClassifier()
    allowed = clf.strategies_for_regime(RegimeLabel.TRENDING_UP)
    assert isinstance(allowed, list)
    allowed_crisis = clf.strategies_for_regime(RegimeLabel.CRISIS)
    assert allowed_crisis == []
