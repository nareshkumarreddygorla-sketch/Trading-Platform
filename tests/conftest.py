"""Pytest fixtures: app, registry, sample bars."""

from datetime import UTC, datetime, timedelta

import pytest

from src.api.app import create_app
from src.core.events import Bar, Exchange


@pytest.fixture(autouse=True)
def _isolate_circuit_state(tmp_path, monkeypatch):
    """Prevent tests from reading/writing the shared circuit_state.json file."""
    import src.risk_engine.manager as rm_mod

    test_state_dir = tmp_path / "data"
    test_state_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(rm_mod, "_CIRCUIT_STATE_DIR", test_state_dir)
    monkeypatch.setattr(rm_mod, "_CIRCUIT_STATE_PATH", test_state_dir / "circuit_state.json")


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def sample_bars():
    """Generate 100 synthetic 1d bars."""
    bars = []
    base = 100.0
    ts = datetime.now(UTC) - timedelta(days=100)
    for i in range(100):
        o = base
        base = base * (1 + (i % 5 - 2) * 0.01)
        h = max(o, base) * 1.01
        l = min(o, base) * 0.99
        c = base
        bars.append(
            Bar(
                symbol="RELIANCE",
                exchange=Exchange.NSE,
                interval="1d",
                open=o,
                high=h,
                low=l,
                close=c,
                volume=1_000_000 + i * 1000,
                ts=ts + timedelta(days=i),
                source="test",
            )
        )
    return bars
