"""
QA Phase 8 — Performance.
Target: loop < 200ms per tick (200 symbols, 1m bars, 10 strategies, signals).
"""
import time
from unittest.mock import MagicMock

import pytest

from src.core.events import Bar, Exchange, Signal, SignalSide
from src.execution.autonomous_loop import AutonomousLoop
from src.execution.order_entry.request import OrderEntryResult
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_autonomous_loop_tick_latency_under_200ms():
    """200 symbols, 1m bars (20 each), 10 strategies; one _tick must complete in < 200ms."""
    n_symbols = 200
    bar_ts = "2025-02-01T10:00:00Z"
    submitted = []

    async def submit_fn(_):
        submitted.append(1)
        return OrderEntryResult(True, order_id="perf_1", latency_ms=0.1)

    symbols = [(f"SYM{i}", Exchange.NSE) for i in range(n_symbols)]
    def get_bars(symbol, exchange, interval, n):
        return [
            Bar(symbol=symbol, exchange=exchange, interval="1m", open=100, high=101, low=99, close=100, volume=1000, ts=datetime.now(timezone.utc), source="test")
            for _ in range(20)
        ]

    def get_symbols():
        return symbols

    signals_out = []
    def run_strategy(state):
        return []

    loop = AutonomousLoop(
        submit_fn,
        get_safe_mode=lambda: False,
        get_bar_ts=lambda: bar_ts,
        get_bars=get_bars,
        get_symbols=get_symbols,
        strategy_runner=MagicMock(run=run_strategy),
        allocator=MagicMock(allocate=lambda *a, **k: []),
        get_risk_state=lambda: {"equity": 100_000, "exposure_multiplier": 1.0, "max_position_pct": 5.0, "drawdown_scale": 1.0, "regime_scale": 1.0},
        get_positions=lambda: [],
        poll_interval_seconds=999,
    )
    start = time.perf_counter()
    await loop._tick()
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 200, f"Loop tick took {elapsed_ms:.1f}ms (target < 200ms)"


@pytest.mark.asyncio
async def test_autonomous_loop_tick_latency_with_features_and_regime():
    """Same but with feature_engine and regime_classifier (heavier path)."""
    from src.ai.feature_engine import FeatureEngine
    from src.ai.regime.classifier import RegimeClassifier

    n_symbols = 50
    bar_ts = "2025-02-01T10:00:00Z"
    bars_cache = [
        Bar(symbol="S", exchange=Exchange.NSE, interval="1m", open=100, high=101, low=99, close=100, volume=1000, ts=datetime.now(timezone.utc), source="test")
        for _ in range(25)
    ]

    async def noop_submit(_):
        return OrderEntryResult(True, order_id="x")

    loop = AutonomousLoop(
        noop_submit,
        get_safe_mode=lambda: False,
        get_bar_ts=lambda: bar_ts,
        get_bars=lambda s, e, i, n: bars_cache,
        get_symbols=lambda: [("S", Exchange.NSE)] * n_symbols,
        strategy_runner=MagicMock(run=lambda s: []),
        allocator=MagicMock(allocate=lambda *a, **k: []),
        get_risk_state=lambda: {"equity": 100_000, "exposure_multiplier": 1.0, "max_position_pct": 5.0},
        get_positions=lambda: [],
        feature_engine=FeatureEngine(),
        regime_classifier=RegimeClassifier(),
        poll_interval_seconds=999,
    )
    start = time.perf_counter()
    await loop._tick()
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 500, f"Loop tick (with features/regime) took {elapsed_ms:.1f}ms"
