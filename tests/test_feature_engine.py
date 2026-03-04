"""Feature engine: deterministic output for same bars; no lookahead."""
from datetime import datetime, timezone

import pytest

from src.core.events import Bar, Exchange
from src.ai.feature_engine import FeatureEngine


def _bar(ts: float, o: float, h: float, l: float, c: float, v: float) -> Bar:
    return Bar(
        symbol="TEST",
        exchange=Exchange.NSE,
        interval="1m",
        open=o, high=h, low=l, close=c, volume=v,
        ts=datetime.fromtimestamp(ts, tz=timezone.utc),
        source="test",
    )


@pytest.fixture
def engine():
    return FeatureEngine()


def test_build_features_deterministic(engine):
    bars = [_bar(1000 + i, 100 + i, 101 + i, 99 + i, 100.5 + i, 1000 * (i + 1)) for i in range(30)]
    f1 = engine.build_features(bars)
    f2 = engine.build_features(bars)
    assert f1 == f2


def test_build_features_no_lookahead(engine):
    bars = [_bar(1000 + i, 100.0, 101.0, 99.0, 100.0, 1000.0) for i in range(25)]
    f = engine.build_features(bars)
    assert "returns_1" in f
    assert "rolling_volatility" in f
    assert "atr" in f
    assert "rsi" in f
    assert "ema_spread" in f
    assert "volume_spike" in f
    assert "close" in f
    assert f["close"] == 100.0


def test_build_features_empty_returns_empty(engine):
    assert engine.build_features([]) == {}
