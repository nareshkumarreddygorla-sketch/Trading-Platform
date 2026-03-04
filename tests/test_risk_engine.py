import pytest
from src.core.events import Signal, SignalSide, Exchange
from src.risk_engine import RiskLimits, RiskManager, LimitCheckResult


def test_limits_position_size():
    limits = RiskLimits(max_position_pct=5.0)
    r = limits.check_position_size(100_000, 6_000)
    assert r.allowed is False
    r = limits.check_position_size(100_000, 4_000)
    assert r.allowed is True


def test_limits_open_positions():
    limits = RiskLimits(max_open_positions=5)
    assert limits.check_open_positions(4).allowed is True
    assert limits.check_open_positions(5).allowed is False


def test_risk_manager_can_place_order():
    mgr = RiskManager(equity=100_000)
    sig = Signal(
        strategy_id="test",
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        side=SignalSide.BUY,
        score=0.8,
        portfolio_weight=0.2,
        risk_level="NORMAL",
        reason="test",
        price=500.0,
    )
    r = mgr.can_place_order(sig, quantity=10, price=500.0)
    assert r.allowed is True
    assert mgr.max_quantity_for_signal(500.0) == 10  # 5% of 100k = 5k, 5k/500 = 10
