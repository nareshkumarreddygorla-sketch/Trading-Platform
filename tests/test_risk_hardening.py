"""Risk hardening: drawdown auto-trip, sector/VaR/per-symbol/consecutive-loss, volatility scaling."""

import pytest

from src.core.events import Exchange, Position, Signal, SignalSide
from src.risk_engine import RiskManager
from src.risk_engine.circuit_breaker import CircuitBreaker
from src.risk_engine.limits import RiskLimits


@pytest.fixture
def limits():
    return RiskLimits(
        max_position_pct=10.0,
        max_daily_loss_pct=2.0,
        max_open_positions=2,
        max_sector_concentration_pct=25.0,
        var_limit_pct=50.0,
        max_per_symbol_pct=15.0,
        max_consecutive_losses=3,
        circuit_breaker_drawdown_pct=5.0,
    )


@pytest.fixture
def risk_manager(limits):
    return RiskManager(equity=100_000.0, limits=limits)


def test_drawdown_auto_trip(risk_manager):
    """Circuit breaker trips when drawdown exceeds limit."""
    cb = CircuitBreaker(risk_manager)
    risk_manager.update_equity(100_000.0)
    cb.update_equity(100_000.0)
    assert cb.allow_order()
    # Draw down 6%
    risk_manager.update_equity(94_000.0)
    cb.update_equity(94_000.0)
    assert not cb.allow_order()
    assert risk_manager.is_circuit_open()


def test_sector_breach_rejection(limits):
    """Order rejected when sector concentration would exceed limit."""
    limits.max_sector_concentration_pct = 20.0
    limits.max_open_positions = 10
    rm = RiskManager(equity=100_000.0, limits=limits)
    rm.positions = [
        Position(symbol="A", exchange=Exchange.NSE, side=SignalSide.BUY, quantity=100, avg_price=200, strategy_id="s1"),
        Position(symbol="B", exchange=Exchange.NSE, side=SignalSide.BUY, quantity=100, avg_price=200, strategy_id="s1"),
    ]
    # Sector GENERIC: 20k + 20k = 40k. Add 10k -> 50k = 50% > 20%
    sig = Signal(
        strategy_id="s1",
        symbol="C",
        exchange=Exchange.NSE,
        side=SignalSide.BUY,
        score=0.8,
        portfolio_weight=0.1,
        price=100.0,
    )
    r = rm.can_place_order(sig, 100, 100.0)
    assert not r.allowed
    assert "sector" in r.reason.lower()


def test_leverage_breach_rejection(limits):
    """Order rejected when total notional exposure exceeds leverage limit (200% of equity)."""
    limits.max_sector_concentration_pct = 100.0
    limits.max_open_positions = 10
    limits.max_position_pct = 100.0  # allow large positions
    limits.max_per_symbol_pct = 100.0
    rm = RiskManager(equity=100_000.0, limits=limits)
    rm.positions = [
        Position(
            symbol="X", exchange=Exchange.NSE, side=SignalSide.BUY, quantity=1500, avg_price=100, strategy_id="s1"
        ),
    ]
    # 150k notional existing. Add 60k -> 210k = 210% > 200% leverage limit
    sig = Signal(
        strategy_id="s1",
        symbol="Y",
        exchange=Exchange.NSE,
        side=SignalSide.BUY,
        score=0.8,
        portfolio_weight=0.1,
        price=100.0,
    )
    r = rm.can_place_order(sig, 600, 100.0)
    assert not r.allowed
    assert "leverage" in r.reason.lower()


def test_consecutive_loss_disable(limits):
    """Orders rejected after N consecutive losing trades."""
    limits.max_consecutive_losses = 2
    limits.max_open_positions = 10
    rm = RiskManager(equity=100_000.0, limits=limits)
    rm.register_pnl(-100.0)
    rm.register_pnl(-100.0)
    sig = Signal(
        strategy_id="s1",
        symbol="Z",
        exchange=Exchange.NSE,
        side=SignalSide.BUY,
        score=0.8,
        portfolio_weight=0.1,
        price=10.0,
    )
    r = rm.can_place_order(sig, 10, 10.0)
    assert not r.allowed
    assert "consecutive" in r.reason.lower()
    rm.register_pnl(50.0)
    r2 = rm.can_place_order(sig, 10, 10.0)
    assert r2.allowed


def test_volatility_spike_scaling(risk_manager):
    """High vol reduces exposure multiplier."""
    risk_manager.set_volatility_scaling(current_vol=30.0, reference_vol=15.0)
    assert risk_manager._exposure_multiplier < 1.0
    risk_manager.set_volatility_scaling(current_vol=10.0, reference_vol=15.0)
    assert risk_manager._exposure_multiplier > 1.0
    assert risk_manager._exposure_multiplier <= 1.5


def test_per_symbol_cap_rejection(limits):
    """Order rejected when per-symbol exposure would exceed limit."""
    limits.max_per_symbol_pct = 10.0
    limits.max_open_positions = 10
    rm = RiskManager(equity=100_000.0, limits=limits)
    rm.positions = [
        Position(
            symbol="INFY", exchange=Exchange.NSE, side=SignalSide.BUY, quantity=100, avg_price=1500, strategy_id="s1"
        ),
    ]
    # 150k notional = 150% - wait, equity 100k so 150k is 150%. So we're already over. Add more INFY:
    sig = Signal(
        strategy_id="s1",
        symbol="INFY",
        exchange=Exchange.NSE,
        side=SignalSide.BUY,
        score=0.8,
        portfolio_weight=0.1,
        price=1500.0,
    )
    r = rm.can_place_order(sig, 1, 1500.0)
    assert not r.allowed
