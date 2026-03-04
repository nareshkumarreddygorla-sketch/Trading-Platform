"""Tests for AI risk gate: all AI decisions pass risk engine."""
import pytest
from src.core.events import Signal, SignalSide, Exchange
from src.risk_engine import RiskManager, RiskLimits
from src.ai.risk_gate import AIRiskGate


@pytest.fixture
def risk_manager():
    return RiskManager(equity=100_000, limits=RiskLimits(max_position_pct=5, max_open_positions=5))


@pytest.fixture
def ai_gate(risk_manager):
    return AIRiskGate(risk_manager)


@pytest.fixture
def sample_signal():
    return Signal(
        strategy_id="ensemble",
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        side=SignalSide.BUY,
        score=0.8,
        portfolio_weight=0.2,
        risk_level="NORMAL",
        reason="ai_ensemble",
        price=500.0,
    )


def test_allow_signal(ai_gate, sample_signal):
    r = ai_gate.allow_signal(sample_signal, quantity=10, price=500.0)
    assert r.allowed is True


def test_allow_signal_blocked_when_over_limit(ai_gate, sample_signal):
    # Quantity too large for 5% of 100k = 5000 => 10 shares at 500 = 5000 ok; 20 would be 10k
    r = ai_gate.allow_signal(sample_signal, quantity=20, price=500.0)
    assert r.allowed is False


def test_allow_parameter_change(ai_gate):
    assert ai_gate.allow_parameter_change("max_position_pct", 4.0) is True
    assert ai_gate.allow_parameter_change("max_position_pct", 25.0) is False
    assert ai_gate.allow_parameter_change("max_daily_loss_pct", 2.0) is True


def test_parameter_blocked_when_circuit_open(ai_gate, risk_manager):
    risk_manager.open_circuit()
    assert ai_gate.allow_parameter_change("max_position_pct", 4.0) is False
