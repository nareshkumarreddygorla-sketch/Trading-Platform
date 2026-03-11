"""
Comprehensive tests for the trading platform's risk engine and execution layer.

Covers:
  - RiskManager: equity updates, order gating, position tracking, PnL, snapshots
  - CircuitBreaker: trip, reset, half-open, force_reset, cooldown, drawdown
  - RiskLimits: position size, open positions, __post_init__ clamping
  - PortfolioVaR: parametric, historical, empty returns
  - RiskMetrics: var_parametric, cvar_parametric, sharpe, kelly, max_drawdown
  - Execution: RateLimiter, KillSwitch, IdempotencyStore

Run:
    PYTHONPATH=. pytest tests/test_risk_execution_v2.py -v --tb=short
"""

import asyncio
from unittest.mock import patch

import numpy as np

from src.core.events import Exchange, Position, Signal, SignalSide
from src.execution.order_entry.idempotency import IdempotencyStore
from src.execution.order_entry.kill_switch import KillReason, KillSwitch, KillSwitchState
from src.execution.order_entry.rate_limiter import OrderRateLimiter, RateLimitConfig
from src.risk_engine.circuit_breaker import CircuitBreaker, CircuitState
from src.risk_engine.limits import RiskLimits
from src.risk_engine.manager import RiskManager
from src.risk_engine.metrics import (
    RiskMetrics,
    compute_risk_metrics,
    cvar_parametric,
    kelly_fraction,
    max_drawdown,
    sharpe_annual,
    var_parametric,
)
from src.risk_engine.var import PortfolioVaR

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(
    symbol: str = "RELIANCE",
    score: float = 0.7,
    price: float = 2500.0,
    strategy_id: str = "test_strategy",
    side: SignalSide = SignalSide.BUY,
) -> Signal:
    return Signal(
        strategy_id=strategy_id,
        symbol=symbol,
        exchange=Exchange.NSE,
        side=side,
        score=score,
        portfolio_weight=1.0,
        risk_level="NORMAL",
        reason="test",
        price=price,
    )


def _make_position(
    symbol: str = "RELIANCE",
    side: SignalSide = SignalSide.BUY,
    quantity: float = 10,
    avg_price: float = 2500.0,
) -> Position:
    return Position(
        symbol=symbol,
        exchange=Exchange.NSE,
        side=side,
        quantity=quantity,
        avg_price=avg_price,
    )


def _run_async(coro):
    """Run an async coroutine in a fresh event loop (for tests that need async)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ============================================================================
# 1. RiskManager Tests
# ============================================================================


class TestRiskManagerUpdateEquity:
    """update_equity: negative equity triggers circuit, positive does not."""

    def test_negative_equity_opens_circuit(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            assert rm.is_circuit_open() is False

            rm.update_equity(-5_000)

            assert rm.equity == -5_000
            assert rm.is_circuit_open() is True

    def test_positive_equity_no_circuit_change(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            rm.update_equity(120_000)

            assert rm.equity == 120_000
            assert rm.is_circuit_open() is False


class TestRiskManagerCanPlaceOrder:
    """can_place_order: circuit, force_reduce, zero equity, basic approval."""

    def test_circuit_open_rejects_normal_order(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            rm.open_circuit("test")
            sig = _make_signal()

            result = rm.can_place_order(sig, quantity=2, price=2500)

            assert result.allowed is False
            assert "circuit_breaker_open" in result.reason

    def test_force_reduce_without_is_reducing_rejects(self):
        rm = RiskManager(equity=100_000, load_persisted_state=False)
        sig = _make_signal()

        result = rm.can_place_order(sig, quantity=2, price=2500, is_reducing=False, force_reduce=True)

        assert result.allowed is False
        assert "force_reduce_requires_is_reducing" in result.reason

    def test_force_reduce_with_is_reducing_bypasses_circuit(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            rm.open_circuit("test")
            sig = _make_signal()

            result = rm.can_place_order(sig, quantity=2, price=2500, is_reducing=True, force_reduce=True)

            assert result.allowed is True

    def test_zero_equity_rejects(self):
        rm = RiskManager(equity=0, load_persisted_state=False)
        sig = _make_signal()

        result = rm.can_place_order(sig, quantity=2, price=2500)

        assert result.allowed is False
        assert "zero_or_negative_equity" in result.reason

    def test_valid_small_order_approved(self):
        rm = RiskManager(equity=100_000, load_persisted_state=False)
        sig = _make_signal(price=500)

        result = rm.can_place_order(sig, quantity=10, price=500)

        assert result.allowed is True

    def test_zero_quantity_rejected(self):
        rm = RiskManager(equity=100_000, load_persisted_state=False)
        sig = _make_signal()

        result = rm.can_place_order(sig, quantity=0, price=2500)

        assert result.allowed is False
        assert "invalid_quantity_or_price" in result.reason

    def test_zero_price_rejected(self):
        rm = RiskManager(equity=100_000, load_persisted_state=False)
        sig = _make_signal()

        result = rm.can_place_order(sig, quantity=2, price=0)

        assert result.allowed is False
        assert "invalid_quantity_or_price" in result.reason


class TestRiskManagerMaxQuantity:
    """max_quantity_for_signal: uses effective equity and position pct."""

    def test_basic_max_quantity(self):
        rm = RiskManager(equity=100_000, load_persisted_state=False)
        # Default max_position_pct = 5%, effective equity = 100k
        # max_val = 100_000 * 0.05 = 5000; 5000 / 500 = 10
        assert rm.max_quantity_for_signal(500.0) == 10

    def test_max_quantity_with_exposure_multiplier(self):
        rm = RiskManager(equity=100_000, load_persisted_state=False)
        rm.set_exposure_multiplier(0.5)
        # effective equity = 100_000 * 0.5 = 50_000
        # max_val = 50_000 * 0.05 = 2500; 2500/500 = 5
        assert rm.max_quantity_for_signal(500.0) == 5

    def test_max_quantity_zero_price(self):
        rm = RiskManager(equity=100_000, load_persisted_state=False)
        assert rm.max_quantity_for_signal(0) == 0

    def test_max_quantity_negative_price(self):
        rm = RiskManager(equity=100_000, load_persisted_state=False)
        assert rm.max_quantity_for_signal(-100) == 0


class TestRiskManagerPositions:
    """record_trade (add_position, add_or_merge_position) updates positions correctly."""

    def test_add_position(self):
        rm = RiskManager(equity=100_000, load_persisted_state=False)
        pos = _make_position()
        rm.add_position(pos)
        assert len(rm.positions) == 1
        assert rm.positions[0].symbol == "RELIANCE"

    def test_add_or_merge_increases_quantity(self):
        rm = RiskManager(equity=100_000, load_persisted_state=False)
        rm.add_position(_make_position(quantity=10, avg_price=2500))
        rm.add_or_merge_position(_make_position(quantity=5, avg_price=2600))

        assert len(rm.positions) == 1
        assert rm.positions[0].quantity == 15
        # VWAP: (10*2500 + 5*2600) / 15 = 38000/15 = 2533.33
        expected_avg = (10 * 2500 + 5 * 2600) / 15
        assert abs(rm.positions[0].avg_price - expected_avg) < 0.01

    def test_remove_position(self):
        rm = RiskManager(equity=100_000, load_persisted_state=False)
        rm.add_position(_make_position())
        assert len(rm.positions) == 1
        rm.remove_position("RELIANCE", "NSE")
        assert len(rm.positions) == 0


class TestRiskManagerDailyPnL:
    """Daily PnL tracking and daily_loss breach."""

    def test_register_pnl_accumulates(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            rm.register_pnl(-200)
            rm.register_pnl(-300)
            assert rm.daily_pnl == -500

    def test_daily_loss_breach_blocks_new_orders(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(
                equity=100_000,
                limits=RiskLimits(max_daily_loss_pct=2.0),
                load_persisted_state=False,
            )
            rm.register_pnl(-3000)  # 3% > 2% limit
            sig = _make_signal(price=500)

            result = rm.can_place_order(sig, quantity=10, price=500)

            assert result.allowed is False
            assert "daily loss" in result.reason

    def test_reset_daily_pnl(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            rm.register_pnl(-500)
            rm.reset_daily_pnl()
            assert rm.daily_pnl == 0.0
            assert rm._consecutive_losses == 0


class TestRiskManagerConsecutiveLosses:
    """Consecutive losses tracking."""

    def test_consecutive_losses_increment(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            rm.register_pnl(-100)
            rm.register_pnl(-200)
            rm.register_pnl(-50)
            assert rm._consecutive_losses == 3

    def test_positive_pnl_resets_consecutive_losses(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            rm.register_pnl(-100)
            rm.register_pnl(-200)
            assert rm._consecutive_losses == 2
            rm.register_pnl(500)
            assert rm._consecutive_losses == 0

    def test_consecutive_losses_breach_blocks_orders(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(
                equity=100_000,
                limits=RiskLimits(max_consecutive_losses=3),
                load_persisted_state=False,
            )
            for _ in range(3):
                rm.register_pnl(-10)
            sig = _make_signal(price=500)

            result = rm.can_place_order(sig, quantity=10, price=500)

            assert result.allowed is False
            assert "consecutive_losses" in result.reason


class TestRiskManagerSnapshot:
    """risk_snapshot returns correct data."""

    def test_snapshot_fields(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            rm.register_pnl(-200)
            rm.add_position(_make_position())

            snap = rm.risk_snapshot()

            assert snap["equity"] == 100_000
            assert snap["daily_pnl"] == -200
            assert snap["circuit_open"] is False
            assert snap["n_positions"] == 1
            assert "timestamp" in snap
            assert "sector_exposures" in snap
            assert snap["consecutive_losses"] == 1

    def test_snapshot_circuit_open(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            rm.open_circuit("test")
            snap = rm.risk_snapshot()
            assert snap["circuit_open"] is True


class TestRiskManagerPositionLimits:
    """Position limit checks in can_place_order."""

    def test_open_positions_limit_rejects(self):
        rm = RiskManager(
            equity=100_000,
            limits=RiskLimits(max_open_positions=2),
            load_persisted_state=False,
        )
        rm.add_position(_make_position(symbol="A"))
        rm.add_position(_make_position(symbol="B"))
        sig = _make_signal(symbol="C", price=500)

        result = rm.can_place_order(sig, quantity=10, price=500)

        assert result.allowed is False
        assert "open positions" in result.reason

    def test_position_size_limit_rejects(self):
        rm = RiskManager(
            equity=100_000,
            limits=RiskLimits(max_position_pct=5.0),
            load_persisted_state=False,
        )
        sig = _make_signal(price=100)

        # 60 * 100 = 6000 = 6% > 5%
        result = rm.can_place_order(sig, quantity=60, price=100)

        assert result.allowed is False
        assert "position" in result.reason


# ============================================================================
# 2. CircuitBreaker Tests
# ============================================================================


class TestCircuitBreakerTrip:
    """trip() opens circuit."""

    def test_trip_opens_circuit(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            cb = CircuitBreaker(rm)
            assert cb.state == CircuitState.CLOSED

            cb.trip()

            assert cb.state == CircuitState.OPEN
            assert rm.is_circuit_open() is True

    def test_double_trip_is_idempotent(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            cb = CircuitBreaker(rm)
            cb.trip()
            cb.trip()  # should not raise

            assert cb.state == CircuitState.OPEN


class TestCircuitBreakerReset:
    """reset() goes to HALF_OPEN."""

    def test_reset_goes_to_half_open(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            cb = CircuitBreaker(rm)
            cb.trip()
            cb.reset(current_equity=100_000)

            assert cb.state == CircuitState.HALF_OPEN
            # Exposure multiplier should be set to 50%
            assert rm._exposure_multiplier == 0.5


class TestCircuitBreakerAllowOrder:
    """allow_order in different states."""

    def test_closed_allows(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            cb = CircuitBreaker(rm)

            assert cb.allow_order() is True

    def test_open_blocks(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            cb = CircuitBreaker(rm)
            cb.trip()

            assert cb.allow_order() is False

    def test_half_open_allows_limited_trades(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            cb = CircuitBreaker(rm)
            cb.trip()
            cb._half_open_observation_secs = 99999  # prevent auto-promotion
            cb.reset(current_equity=100_000)
            # Disable auto-promotion so allow_order() tests pure HALF_OPEN limiting
            cb.check_half_open_promotion = lambda: None

            max_trades = cb._half_open_max_trades  # 3

            results = [cb.allow_order() for _ in range(max_trades + 3)]

            allowed_count = sum(1 for r in results if r)
            assert allowed_count == max_trades

    def test_half_open_blocks_after_limit_exhausted(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            cb = CircuitBreaker(rm)
            cb.trip()
            cb._half_open_observation_secs = 99999
            cb.reset(current_equity=100_000)
            # Disable auto-promotion so we test pure HALF_OPEN blocking
            cb.check_half_open_promotion = lambda: None

            for _ in range(cb._half_open_max_trades):
                cb.allow_order()

            # Next order should be blocked
            assert cb.allow_order() is False


class TestCircuitBreakerHalfOpenPromotion:
    """check_half_open_promotion after observation period."""

    def test_promotion_after_observation_and_trades(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            cb = CircuitBreaker(rm)
            cb.trip()
            cb._half_open_observation_secs = 0  # immediate promotion possible
            cb.reset(current_equity=100_000)

            # Execute enough trades
            for _ in range(cb._half_open_max_trades):
                cb.allow_order()

            # Now check promotion
            cb.check_half_open_promotion()
            assert cb.state == CircuitState.CLOSED
            assert rm._exposure_multiplier == 1.0

    def test_no_promotion_without_enough_trades(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            cb = CircuitBreaker(rm)
            cb.trip()
            cb._half_open_observation_secs = 0
            cb.reset(current_equity=100_000)

            # Don't execute enough trades
            cb.allow_order()  # only 1 of 3

            cb.check_half_open_promotion()
            assert cb.state == CircuitState.HALF_OPEN


class TestCircuitBreakerForceReset:
    """force_reset skips HALF_OPEN; cooldown prevents rapid resets."""

    def test_force_reset_goes_directly_to_closed(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            cb = CircuitBreaker(rm)
            cb.trip()
            assert cb.state == CircuitState.OPEN

            cb.force_reset(current_equity=100_000)

            assert cb.state == CircuitState.CLOSED
            assert rm._exposure_multiplier == 1.0
            assert rm.is_circuit_open() is False

    def test_force_reset_cooldown_prevents_rapid_resets(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(equity=100_000, load_persisted_state=False)
            cb = CircuitBreaker(rm)

            # First force_reset should work
            cb.trip()
            cb.force_reset(current_equity=100_000)
            assert cb.state == CircuitState.CLOSED

            # Trip again immediately
            cb.trip()
            assert cb.state == CircuitState.OPEN

            # Second force_reset within cooldown should be rejected
            cb.force_reset(current_equity=100_000)
            assert cb.state == CircuitState.OPEN  # still open; cooldown blocked it


class TestCircuitBreakerUpdateEquity:
    """update_equity triggers trip on drawdown."""

    def test_drawdown_triggers_trip(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(
                equity=100_000,
                limits=RiskLimits(circuit_breaker_drawdown_pct=5.0),
                load_persisted_state=False,
            )
            cb = CircuitBreaker(rm)

            # Set peak equity
            cb.update_equity(100_000)
            # 6% drawdown > 5% limit
            cb.update_equity(94_000)

            assert cb.state == CircuitState.OPEN

    def test_small_drawdown_no_trip(self, tmp_path):
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", tmp_path / "cs.json"),
        ):
            rm = RiskManager(
                equity=100_000,
                limits=RiskLimits(circuit_breaker_drawdown_pct=5.0),
                load_persisted_state=False,
            )
            cb = CircuitBreaker(rm)

            cb.update_equity(100_000)
            cb.update_equity(96_000)  # 4% < 5%

            assert cb.state == CircuitState.CLOSED


# ============================================================================
# 3. RiskLimits Tests
# ============================================================================


class TestRiskLimitsChecks:
    """Position size and open positions checks."""

    def test_position_size_pass(self):
        limits = RiskLimits(max_position_pct=5.0)
        result = limits.check_position_size(100_000, 4_000)
        assert result.allowed is True

    def test_position_size_fail(self):
        limits = RiskLimits(max_position_pct=5.0)
        result = limits.check_position_size(100_000, 6_000)
        assert result.allowed is False

    def test_open_positions_pass(self):
        limits = RiskLimits(max_open_positions=10)
        assert limits.check_open_positions(9).allowed is True

    def test_open_positions_fail_at_limit(self):
        limits = RiskLimits(max_open_positions=10)
        assert limits.check_open_positions(10).allowed is False


class TestRiskLimitsPostInit:
    """__post_init__ validation clamps values."""

    def test_clamp_max_position_pct_below_minimum(self):
        limits = RiskLimits(max_position_pct=0.001)
        assert limits.max_position_pct == 0.1  # clamped to 0.1

    def test_clamp_max_position_pct_above_maximum(self):
        limits = RiskLimits(max_position_pct=50.0)
        assert limits.max_position_pct == 25.0  # clamped to 25.0

    def test_clamp_max_daily_loss_pct(self):
        limits = RiskLimits(max_daily_loss_pct=0.001)
        assert limits.max_daily_loss_pct == 0.1

        limits2 = RiskLimits(max_daily_loss_pct=20.0)
        assert limits2.max_daily_loss_pct == 10.0

    def test_clamp_max_open_positions(self):
        limits = RiskLimits(max_open_positions=0)
        assert limits.max_open_positions == 1  # clamped to 1

        limits2 = RiskLimits(max_open_positions=100)
        assert limits2.max_open_positions == 50  # clamped to 50

    def test_clamp_max_consecutive_losses(self):
        limits = RiskLimits(max_consecutive_losses=0)
        assert limits.max_consecutive_losses == 1

        limits2 = RiskLimits(max_consecutive_losses=50)
        assert limits2.max_consecutive_losses == 20

    def test_normal_values_not_clamped(self):
        limits = RiskLimits(
            max_position_pct=5.0, max_daily_loss_pct=2.0, max_open_positions=10, max_consecutive_losses=5
        )
        assert limits.max_position_pct == 5.0
        assert limits.max_daily_loss_pct == 2.0
        assert limits.max_open_positions == 10
        assert limits.max_consecutive_losses == 5


# ============================================================================
# 4. PortfolioVaR Tests
# ============================================================================


class TestPortfolioVaRParametric:
    """Parametric VaR method."""

    def test_parametric_var_positive(self):
        var_calc = PortfolioVaR(min_history=5, var_method="parametric")
        # Feed returns
        for i in range(30):
            ret = 0.001 + ((i * 7 + 3) % 11 - 5) * 0.005
            var_calc.update_returns("RELIANCE", ret)

        positions = [{"symbol": "RELIANCE", "notional": 50_000}]
        result = var_calc.compute(positions, 100_000, method="parametric")

        assert result.var_95 > 0
        assert result.var_99 > result.var_95
        assert result.method == "parametric"
        assert result.n_positions == 1

    def test_parametric_var_scales_with_notional(self):
        var_calc = PortfolioVaR(min_history=5, var_method="parametric")
        for i in range(30):
            var_calc.update_returns("RELIANCE", 0.001 + ((i * 7) % 11 - 5) * 0.005)

        pos_small = [{"symbol": "RELIANCE", "notional": 10_000}]
        pos_large = [{"symbol": "RELIANCE", "notional": 50_000}]

        r_small = var_calc.compute(pos_small, 100_000, method="parametric")
        r_large = var_calc.compute(pos_large, 100_000, method="parametric")

        assert r_large.var_95 > r_small.var_95


class TestPortfolioVaRHistorical:
    """Historical VaR method."""

    def test_historical_var_with_sufficient_data(self):
        var_calc = PortfolioVaR(min_history=5, var_method="historical")
        np.random.seed(42)
        for i in range(60):
            ret = np.random.normal(0.001, 0.02)
            var_calc.update_returns("RELIANCE", float(ret))

        positions = [{"symbol": "RELIANCE", "notional": 50_000}]
        result = var_calc.compute(positions, 100_000, method="historical")

        assert result.var_95 > 0
        assert result.var_99 >= result.var_95

    def test_historical_var_falls_back_with_insufficient_data(self):
        var_calc = PortfolioVaR(min_history=5, var_method="historical")
        # Only feed 3 observations (less than 20 needed for historical)
        for i in range(3):
            var_calc.update_returns("RELIANCE", 0.01 * (i + 1))

        positions = [{"symbol": "RELIANCE", "notional": 50_000}]
        result = var_calc.compute(positions, 100_000, method="historical")

        # Should fall back to cornish_fisher
        assert result.method == "cornish_fisher"
        assert result.var_95 > 0


class TestPortfolioVaREmpty:
    """VaR with empty returns and edge cases."""

    def test_empty_positions_returns_zero_var(self):
        var_calc = PortfolioVaR(min_history=5)
        result = var_calc.compute([], 100_000)

        assert result.var_95 == 0.0
        assert result.var_99 == 0.0

    def test_zero_portfolio_value_returns_zero_var(self):
        var_calc = PortfolioVaR(min_history=5)
        positions = [{"symbol": "RELIANCE", "notional": 50_000}]
        result = var_calc.compute(positions, 0.0)

        assert result.var_95 == 0.0
        assert result.var_99 == 0.0


# ============================================================================
# 5. RiskMetrics Tests
# ============================================================================


class TestRiskMetricsFunctions:
    """Test individual risk metric computation functions."""

    def test_var_parametric_normal(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 200)
        var = var_parametric(returns, confidence=0.95)
        assert var > 0

    def test_var_parametric_insufficient_data(self):
        returns = np.array([0.01])
        var = var_parametric(returns, confidence=0.95)
        assert var == 0.0

    def test_cvar_exceeds_var(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 200)
        var = var_parametric(returns, 0.95)
        cvar = cvar_parametric(returns, 0.95)
        # CVaR (expected shortfall) should be >= VaR
        assert cvar >= var

    def test_sharpe_annual(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        s = sharpe_annual(returns)
        assert s > 0  # positive mean with moderate vol should give positive Sharpe

    def test_sharpe_zero_vol(self):
        returns = np.array([0.0, 0.0, 0.0])
        s = sharpe_annual(returns)
        assert s == 0.0

    def test_max_drawdown(self):
        prices = np.array([100, 110, 105, 95, 100, 90, 95])
        md, duration = max_drawdown(prices)
        assert md < 0  # drawdown is negative
        # Max drawdown should be from 110 to 90 => (90-110)/110 ~ -18.18%
        assert md < -0.10

    def test_kelly_fraction_positive_returns(self):
        np.random.seed(42)
        returns = np.random.normal(0.005, 0.02, 100)
        kf = kelly_fraction(returns)
        assert 0 <= kf <= 1.0

    def test_compute_risk_metrics_integration(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 200)
        metrics = compute_risk_metrics(returns)
        assert isinstance(metrics, RiskMetrics)
        assert metrics.var_95 > 0
        assert metrics.cvar_95 >= metrics.var_95
        assert metrics.max_drawdown < 0
        assert metrics.max_drawdown_duration >= 0

    def test_compute_risk_metrics_insufficient_data(self):
        returns = np.array([0.01])
        metrics = compute_risk_metrics(returns)
        assert metrics.var_95 == 0.0
        assert metrics.cvar_95 == 0.0
        assert metrics.sharpe == 0.0


# ============================================================================
# 6. Execution: RateLimiter Tests
# ============================================================================


class TestOrderRateLimiter:
    """RateLimiter allows within limit, blocks when exceeded."""

    def test_allows_within_limit(self):
        config = RateLimitConfig(max_orders_per_minute=5, window_seconds=60)
        limiter = OrderRateLimiter(config)

        for _ in range(5):
            assert limiter.allow() is True

    def test_blocks_when_exceeded(self):
        config = RateLimitConfig(max_orders_per_minute=3, window_seconds=60)
        limiter = OrderRateLimiter(config)

        for _ in range(3):
            limiter.allow()

        assert limiter.allow() is False

    def test_reset_clears_state(self):
        config = RateLimitConfig(max_orders_per_minute=2, window_seconds=60)
        limiter = OrderRateLimiter(config)

        limiter.allow()
        limiter.allow()
        assert limiter.allow() is False

        limiter.reset()
        assert limiter.allow() is True

    def test_default_config(self):
        limiter = OrderRateLimiter()
        # Default is 20 orders per minute; should allow at least a few
        for _ in range(10):
            assert limiter.allow() is True


# ============================================================================
# 7. Execution: KillSwitch Tests
# ============================================================================


class TestKillSwitch:
    """KillSwitch engage/disengage (async)."""

    def test_arm_and_check(self, tmp_path):
        async def _test():
            import src.execution.order_entry.kill_switch as ks_mod

            original = ks_mod._KILL_STATE_PATH
            ks_mod._KILL_STATE_PATH = str(tmp_path / "kill_switch_state.json")
            try:
                ks = KillSwitch()
                assert await ks.is_armed() is False

                await ks.arm(KillReason.MANUAL, detail="test")
                assert await ks.is_armed() is True

                state = await ks.get_state()
                assert state.armed is True
                assert state.reason == KillReason.MANUAL
            finally:
                ks_mod._KILL_STATE_PATH = original

        _run_async(_test())

    def test_disarm(self, tmp_path):
        async def _test():
            import src.execution.order_entry.kill_switch as ks_mod

            original = ks_mod._KILL_STATE_PATH
            ks_mod._KILL_STATE_PATH = str(tmp_path / "kill_switch_state.json")
            try:
                ks = KillSwitch()
                await ks.arm(KillReason.MANUAL, detail="test")
                assert await ks.is_armed() is True

                await ks.disarm()
                assert await ks.is_armed() is False
            finally:
                ks_mod._KILL_STATE_PATH = original

        _run_async(_test())

    def test_allow_reduce_only_order_long_sell(self):
        """Selling to close a long position should be allowed in kill-switch."""
        state = KillSwitchState(armed=True, allow_reduce_only=True)
        allowed = KillSwitch.allow_reduce_only_order(state, "RELIANCE", "SELL", 10, 20)
        assert allowed is True

    def test_allow_reduce_only_order_blocks_buy_on_long(self):
        """Buying when already long should be blocked in kill-switch."""
        state = KillSwitchState(armed=True, allow_reduce_only=True)
        allowed = KillSwitch.allow_reduce_only_order(state, "RELIANCE", "BUY", 5, 20)
        assert allowed is False

    def test_allow_reduce_only_order_short_buy(self):
        """Buying to close a short position should be allowed in kill-switch."""
        state = KillSwitchState(armed=True, allow_reduce_only=True)
        allowed = KillSwitch.allow_reduce_only_order(state, "RELIANCE", "BUY", 5, -10)
        assert allowed is True

    def test_unarmed_allows_everything(self):
        state = KillSwitchState(armed=False)
        assert KillSwitch.allow_reduce_only_order(state, "A", "BUY", 100, 0) is True
        assert KillSwitch.allow_reduce_only_order(state, "A", "SELL", 100, 0) is True

    def test_manual_reason_cannot_be_downgraded(self):
        """Manual kill reason should not be overwritten by auto-disarmable reason."""

        async def _test():
            ks = KillSwitch()
            await ks.arm(KillReason.MAX_DAILY_LOSS, detail="big loss")
            state1 = await ks.get_state()
            assert state1.reason == KillReason.MAX_DAILY_LOSS

            # Try to overwrite with a recoverable reason
            await ks.arm(KillReason.MARKET_FEED_FAILURE, detail="feed down")
            state2 = await ks.get_state()
            # Should still be MAX_DAILY_LOSS (manual-only reasons are not downgraded)
            assert state2.reason == KillReason.MAX_DAILY_LOSS

        _run_async(_test())


# ============================================================================
# 8. Execution: IdempotencyGuard Tests
# ============================================================================


class TestIdempotencyGuard:
    """IdempotencyGuard dedup detection (in-memory fallback)."""

    def test_set_returns_true_for_new_key(self):
        async def _test():
            store = IdempotencyStore(redis_url="redis://nonexistent:6379/0")
            result = await store.set("key1", "order_123")
            assert result is True

        _run_async(_test())

    def test_set_returns_false_for_duplicate_key(self):
        async def _test():
            store = IdempotencyStore(redis_url="redis://nonexistent:6379/0")
            await store.set("key1", "order_123")
            result = await store.set("key1", "order_456")
            assert result is False

        _run_async(_test())

    def test_get_returns_stored_data(self):
        async def _test():
            store = IdempotencyStore(redis_url="redis://nonexistent:6379/0")
            await store.set("key1", "order_123", "broker_456", "PENDING")
            data = await store.get("key1")
            assert data is not None
            assert data["order_id"] == "order_123"
            assert data["broker_order_id"] == "broker_456"
            assert data["status"] == "PENDING"

        _run_async(_test())

    def test_get_returns_none_for_missing_key(self):
        async def _test():
            store = IdempotencyStore(redis_url="redis://nonexistent:6379/0")
            data = await store.get("nonexistent_key")
            assert data is None

        _run_async(_test())

    def test_update_overwrites_existing(self):
        async def _test():
            store = IdempotencyStore(redis_url="redis://nonexistent:6379/0")
            await store.set("key1", "order_123", None, "PENDING")
            await store.update("key1", "order_123", "broker_789", "FILLED")
            data = await store.get("key1")
            assert data is not None
            assert data["broker_order_id"] == "broker_789"
            assert data["status"] == "FILLED"

        _run_async(_test())

    def test_derive_key_deterministic(self):
        k1 = IdempotencyStore.derive_key("s1", "REL", "BUY", 10, 500.0, "2024-01-01T00:00:00Z")
        k2 = IdempotencyStore.derive_key("s1", "REL", "BUY", 10, 500.0, "2024-01-01T00:00:00Z")
        assert k1 == k2

    def test_derive_key_different_inputs(self):
        k1 = IdempotencyStore.derive_key("s1", "REL", "BUY", 10, 500.0, "2024-01-01T00:00:00Z")
        k2 = IdempotencyStore.derive_key("s1", "REL", "SELL", 10, 500.0, "2024-01-01T00:00:00Z")
        assert k1 != k2

    def test_is_available_returns_true_for_mem_fallback(self):
        async def _test():
            store = IdempotencyStore(redis_url="redis://nonexistent:6379/0")
            available = await store.is_available()
            assert available is True  # In-memory fallback is always available

        _run_async(_test())

    def test_set_if_new_or_get_new_key(self):
        async def _test():
            store = IdempotencyStore(redis_url="redis://nonexistent:6379/0")
            is_new, existing = await store.set_if_new_or_get("key1", "order_1", None, "PENDING")
            assert is_new is True
            assert existing is None

        _run_async(_test())

    def test_set_if_new_or_get_existing_key(self):
        async def _test():
            store = IdempotencyStore(redis_url="redis://nonexistent:6379/0")
            await store.set("key1", "order_1", None, "PENDING")
            is_new, existing = await store.set_if_new_or_get("key1", "order_2", None, "PENDING")
            assert is_new is False
            assert existing is not None
            assert existing["order_id"] == "order_1"

        _run_async(_test())


# ============================================================================
# 9. Additional Edge Cases and Integration
# ============================================================================


class TestRiskManagerExposureMultiplier:
    """Exposure multiplier affects effective equity."""

    def test_exposure_multiplier_clamped(self):
        rm = RiskManager(equity=100_000, load_persisted_state=False)
        rm.set_exposure_multiplier(3.0)
        assert rm._exposure_multiplier == 1.5  # clamped to max

        rm.set_exposure_multiplier(-1.0)
        assert rm._exposure_multiplier == 0.5  # clamped to min

    def test_effective_equity(self):
        rm = RiskManager(equity=100_000, load_persisted_state=False)
        assert rm.effective_equity() == 100_000

        rm.set_exposure_multiplier(0.5)
        assert rm.effective_equity() == 50_000

    def test_volatility_scaling(self):
        rm = RiskManager(equity=100_000, load_persisted_state=False)
        rm.set_volatility_scaling(current_vol=30.0, reference_vol=15.0)
        # mult = 15 / 30 = 0.5
        assert rm._exposure_multiplier == 0.5

        rm.set_volatility_scaling(current_vol=0.0)
        assert rm._exposure_multiplier == 1.0


class TestRiskManagerCheckDrawdown:
    """check_drawdown returns True when threshold breached."""

    def test_drawdown_exceeds_limit(self):
        rm = RiskManager(
            equity=100_000,
            limits=RiskLimits(circuit_breaker_drawdown_pct=5.0),
            load_persisted_state=False,
        )
        # 6% drawdown
        assert rm.check_drawdown(100_000, 94_000) is True

    def test_drawdown_within_limit(self):
        rm = RiskManager(
            equity=100_000,
            limits=RiskLimits(circuit_breaker_drawdown_pct=5.0),
            load_persisted_state=False,
        )
        # 3% drawdown
        assert rm.check_drawdown(100_000, 97_000) is False

    def test_drawdown_zero_peak(self):
        rm = RiskManager(equity=100_000, load_persisted_state=False)
        assert rm.check_drawdown(0, 97_000) is False


class TestRiskLimitsAdditionalChecks:
    """Additional limit check methods."""

    def test_check_per_symbol_exposure(self):
        limits = RiskLimits(max_per_symbol_pct=10.0)
        # 15% of equity in one symbol
        result = limits.check_per_symbol_exposure(100_000, 15_000)
        assert result.allowed is False

        result = limits.check_per_symbol_exposure(100_000, 8_000)
        assert result.allowed is True

    def test_check_sector_concentration(self):
        limits = RiskLimits(max_sector_concentration_pct=25.0)
        result = limits.check_sector_concentration(100_000, 30_000)
        assert result.allowed is False

        result = limits.check_sector_concentration(100_000, 20_000)
        assert result.allowed is True

    def test_check_leverage(self):
        limits = RiskLimits()
        # 250% leverage > 200% limit
        result = limits.check_leverage(100_000, 250_000)
        assert result.allowed is False

        result = limits.check_leverage(100_000, 150_000)
        assert result.allowed is True

    def test_check_consecutive_losses_boundary(self):
        limits = RiskLimits(max_consecutive_losses=5)
        assert limits.check_consecutive_losses(4).allowed is True
        assert limits.check_consecutive_losses(5).allowed is False
        assert limits.check_consecutive_losses(6).allowed is False
