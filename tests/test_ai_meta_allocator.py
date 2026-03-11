"""Tests for meta-allocator and decay detection."""

import numpy as np

from src.ai.meta_allocator import DecayDetector, MetaAllocator
from src.ai.meta_allocator.weights import compute_kelly_weights, compute_risk_parity_weights


def test_decay_detector():
    det = DecayDetector(lookback=20)
    # First half positive, second half negative => decay
    returns = np.concatenate([np.ones(10) * 0.01, np.ones(10) * -0.01])
    assert det.detect(returns)
    # No decay
    assert not det.detect(np.ones(20) * 0.01)


def test_risk_parity_weights():
    returns = {"a": [0.01, 0.02, -0.01], "b": [0.005, 0.005, 0.005]}
    w = compute_risk_parity_weights(returns)
    assert set(w.keys()) == {"a", "b"}
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_kelly_weights():
    params = {"s1": (1.0, 0.55), "s2": (0.5, 0.52)}
    w = compute_kelly_weights(params, fraction=0.25)
    assert sum(w.values()) <= 1.0 + 1e-6


def test_meta_allocator_allocate():
    allocator = MetaAllocator(min_sharpe=0.3, min_win_rate=0.4)
    allocator.update_stats("ema", sharpe=0.8, win_rate=0.52, max_dd=0.05, n_trades=50, confidence=0.6)
    allocator.update_returns("ema", 0.01)
    allocator.update_returns("ema", -0.005)
    allocator.update_stats("rsi", sharpe=0.2, win_rate=0.4, max_dd=0.02, n_trades=30, confidence=0.4)
    result = allocator.allocate(["ema", "rsi"], equity=100_000)
    ema_alloc = next(a for a in result if a.strategy_id == "ema")
    rsi_alloc = next(a for a in result if a.strategy_id == "rsi")
    assert ema_alloc.enabled is True
    assert rsi_alloc.enabled is False or rsi_alloc.weight == 0.0
