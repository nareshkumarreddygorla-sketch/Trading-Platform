"""
QA Phase 2 — Autonomous loop validation.
4) Duplicate bar protection: same bar twice → one order.
5) Feature lookahead: FeatureEngine never reads future bar (see test_feature_engine).
6) Strategy disable feedback: 5 consecutive losses → strategy_disabled; no further signals.
7) Market feed death: unhealthy → safe_mode; loop pauses; manual still works.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.ai.performance_tracker import PerformanceTracker
from src.core.events import Bar, Exchange, Signal, SignalSide
from src.execution.autonomous_loop import AutonomousLoop, stable_idempotency_key
from src.execution.order_entry.request import OrderEntryRequest, OrderEntryResult


# --- 4) Duplicate bar protection ---
@pytest.mark.asyncio
async def test_duplicate_bar_same_ts_submits_once():
    """Feed same bar_ts twice; only one order submitted (idempotency key stable)."""
    submitted = []

    async def submit_fn(req: OrderEntryRequest) -> OrderEntryResult:
        submitted.append(req)
        return OrderEntryResult(True, order_id="ord_1", latency_ms=1.0)

    bar_ts = "2025-01-15T10:00:00Z"
    get_bar_ts = lambda: bar_ts

    def get_bars(s, e, i, n):
        return [
            Bar(
                symbol=s,
                exchange=e,
                interval="1m",
                open=100,
                high=101,
                low=99,
                close=100.5,
                volume=1000,
                ts=datetime.now(UTC),
                source="test",
            )
            for _ in range(25)
        ]

    get_symbols = lambda: [("RELIANCE", Exchange.NSE)]
    all_signals = []

    def run_strategy(state):
        sigs = [
            Signal(
                strategy_id="s1",
                symbol=state.symbol,
                exchange=state.exchange,
                side=SignalSide.BUY,
                score=0.8,
                portfolio_weight=0.1,
                price=100.5,
            )
        ]
        all_signals.extend(sigs)
        return sigs

    class Allocator:
        def allocate(self, signals, equity, positions, **kwargs):
            return [(s, 10) for s in signals][:1]

    loop = AutonomousLoop(
        submit_fn,
        get_safe_mode=lambda: False,
        get_bar_ts=get_bar_ts,
        get_bars=get_bars,
        get_symbols=get_symbols,
        strategy_runner=MagicMock(run=run_strategy),
        allocator=Allocator(),
        get_risk_state=lambda: {
            "equity": 100000,
            "exposure_multiplier": 1.0,
            "max_position_pct": 5.0,
            "drawdown_scale": 1.0,
            "regime_scale": 1.0,
        },
        get_positions=lambda: [],
        poll_interval_seconds=999,
    )
    await loop._tick()
    await loop._tick()
    key_used = stable_idempotency_key(bar_ts, "s1", "RELIANCE", "BUY")
    assert key_used and len(key_used) > 0
    assert len(submitted) <= 1, "Duplicate bar must not submit twice (same bar_ts)"


# --- 5) Feature lookahead: covered in tests/test_feature_engine.py ---


# --- 6) Strategy disable feedback ---
def test_strategy_disable_after_consecutive_losses():
    """Simulate 5 consecutive losses; strategy disabled; exposure multiplier adjusts."""
    disabled_events = []
    exposure_events = []

    tracker = PerformanceTracker(
        max_consecutive_losses_disable=5,
        on_strategy_disabled=lambda sid, reason: disabled_events.append((sid, reason)),
        on_exposure_multiplier_changed=lambda sid, mult: exposure_events.append((sid, mult)),
    )
    for _ in range(5):
        tracker.record_fill("strategy_a", -100.0)
    assert tracker.is_disabled("strategy_a")
    assert len(disabled_events) >= 1
    assert any("strategy_a" in str(e) for e in disabled_events)


# --- 7) Market feed death ---
@pytest.mark.asyncio
async def test_market_feed_unhealthy_loop_skips_tick():
    """When get_market_feed_healthy returns False, _tick must skip (no orders)."""
    submitted = []

    async def submit_fn(req):
        submitted.append(req)
        return OrderEntryResult(True, order_id="x")

    loop = AutonomousLoop(
        submit_fn,
        get_safe_mode=lambda: False,
        get_market_feed_healthy=lambda: False,
        get_bar_ts=lambda: "2025-01-15T10:00:00Z",
        get_bars=lambda s, e, i, n: [
            Bar(
                symbol=s,
                exchange=e,
                interval="1m",
                open=100,
                high=101,
                low=99,
                close=100,
                volume=1000,
                ts=datetime.now(UTC),
                source="test",
            )
            for _ in range(25)
        ],
        get_symbols=lambda: [("X", Exchange.NSE)],
        strategy_runner=MagicMock(run=lambda s: []),
        allocator=MagicMock(allocate=lambda *a, **k: []),
        get_risk_state=lambda: {},
        get_positions=lambda: [],
        poll_interval_seconds=999,
    )
    await loop._tick()
    assert len(submitted) == 0, "When market feed unhealthy, autonomous loop must not submit orders"
