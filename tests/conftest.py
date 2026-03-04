"""Pytest fixtures: app, registry, sample bars."""
import pytest
from datetime import datetime, timezone, timedelta

from src.core.events import Bar, Exchange
from src.api.app import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def sample_bars():
    """Generate 100 synthetic 1d bars."""
    bars = []
    base = 100.0
    ts = datetime.now(timezone.utc) - timedelta(days=100)
    for i in range(100):
        o = base
        base = base * (1 + (i % 5 - 2) * 0.01)
        h = max(o, base) * 1.01
        l = min(o, base) * 0.99
        c = base
        bars.append(Bar(
            symbol="RELIANCE",
            exchange=Exchange.NSE,
            interval="1d",
            open=o, high=h, low=l, close=c,
            volume=1_000_000 + i * 1000,
            ts=ts + timedelta(days=i),
            source="test",
        ))
    return bars
