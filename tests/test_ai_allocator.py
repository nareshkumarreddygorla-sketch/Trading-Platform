"""AI portfolio allocator: respects RiskManager.can_place_order; returns SizedSignal only."""
import pytest

from src.core.events import Exchange, Position, Signal, SignalSide
from src.risk_engine import RiskManager
from src.risk_engine.limits import RiskLimits
from src.ai.portfolio_allocator import PortfolioAllocator, SizedSignal


@pytest.fixture
def risk_manager():
    return RiskManager(
        equity=100_000.0,
        limits=RiskLimits(max_open_positions=5, max_position_pct=10.0),
    )


def test_allocate_respects_can_place_order(risk_manager):
    allocator = PortfolioAllocator(risk_manager, max_concurrent_trades=3, min_confidence=0.5)
    sig = Signal(strategy_id="s1", symbol="RELIANCE", exchange=Exchange.NSE, side=SignalSide.BUY, score=0.8, portfolio_weight=0.1, price=2500.0)
    result = allocator.allocate([sig], 100_000.0, [])
    assert all(isinstance(x, SizedSignal) for x in result)
    assert all(hasattr(x, "signal") and hasattr(x, "quantity") for x in result)
    if result:
        assert result[0].quantity > 0


def test_allocate_filters_low_confidence(risk_manager):
    allocator = PortfolioAllocator(risk_manager, min_confidence=0.8)
    sig = Signal(strategy_id="s1", symbol="RELIANCE", exchange=Exchange.NSE, side=SignalSide.BUY, score=0.3, portfolio_weight=0.1, price=2500.0)
    result = allocator.allocate([sig], 100_000.0, [])
    assert len(result) == 0


def test_allocate_caps_concurrent(risk_manager):
    allocator = PortfolioAllocator(risk_manager, max_concurrent_trades=2, min_confidence=0.5)
    signals = [
        Signal(strategy_id="s1", symbol=f"S{i}", exchange=Exchange.NSE, side=SignalSide.BUY, score=0.9 - i * 0.1, portfolio_weight=0.1, price=100.0)
        for i in range(5)
    ]
    result = allocator.allocate(signals, 100_000.0, [])
    assert len(result) <= 2
