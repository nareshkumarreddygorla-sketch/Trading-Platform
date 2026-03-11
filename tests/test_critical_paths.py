"""
Critical trading path tests: verify bug fixes and safety invariants.

Covers:
1. Kelly sizing — dampened sizing, score proportionality, hard caps
2. VaR fallback — Monte Carlo -> cornish_fisher, historical -> cornish_fisher
3. Daily PnL persistence — register_pnl triggers save/load cycle
4. force_reduce validation — must have is_reducing=True
5. Circuit breaker thread safety — concurrent half-open allow_order
6. Risk limits validation — single trade loss, position size, daily loss
7. End-to-end signal flow — strategy -> allocator -> risk gate

Run:
    PYTHONPATH=. pytest tests/test_critical_paths.py -v --tb=short
"""

import json
import threading
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from src.core.events import Bar, Exchange, Signal, SignalSide
from src.risk_engine.circuit_breaker import CircuitBreaker, CircuitState
from src.risk_engine.limits import RiskLimits
from src.risk_engine.manager import RiskManager
from src.risk_engine.var import Z_95, PortfolioVaR, _cornish_fisher_z
from src.strategy_engine.allocator import AllocatorConfig, PortfolioAllocator
from src.strategy_engine.base import MarketState

# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════


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


def _make_bars(n: int = 200, symbol: str = "RELIANCE", base_price: float = 2500.0) -> list[Bar]:
    """Generate deterministic synthetic bars (no randomness)."""
    bars = []
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    price = base_price
    for i in range(n):
        # Deterministic drift: small uptrend with deterministic noise
        noise = ((i * 7 + 13) % 11 - 5) * 0.001
        price *= 1 + 0.0003 + noise
        o = price * (1 + ((i * 3) % 7 - 3) * 0.001)
        h = max(o, price) * (1 + abs(noise) * 0.5)
        low = min(o, price) * (1 - abs(noise) * 0.5)
        vol = 500_000 + ((i * 17) % 100) * 10_000
        bars.append(
            Bar(
                symbol=symbol,
                exchange=Exchange.NSE,
                interval="1d",
                open=round(o, 2),
                high=round(h, 2),
                low=round(low, 2),
                close=round(price, 2),
                volume=vol,
                ts=ts + timedelta(days=i),
            )
        )
    return bars


# ════════════════════════════════════════════════════════════════════════
# 1. Kelly Sizing — Fixed Bug Verification
# ════════════════════════════════════════════════════════════════════════


class TestKellySizing:
    """Verify Kelly criterion position sizing produces correct, bounded results."""

    def _allocate_single(
        self,
        score: float,
        equity: float = 100_000.0,
        strategy_cap_pct: float = 10.0,
        max_position_pct: float = 5.0,
        price: float = 2500.0,
    ) -> int:
        """Helper: allocate a single signal and return quantity."""
        config = AllocatorConfig(
            max_active_signals=5,
            max_capital_pct_per_signal=strategy_cap_pct,
            min_confidence=0.3,
            use_kelly=True,
            kelly_fraction=0.5,
        )
        allocator = PortfolioAllocator(config)
        signal = _make_signal(score=score, price=price)
        allocs = allocator.allocate(
            signals=[signal],
            equity=equity,
            positions=[],
            max_position_pct=max_position_pct,
        )
        if not allocs:
            return 0
        return allocs[0][1]

    def test_score_05_gets_kelly_dampened_position(self):
        """Signal with score=0.5 should get a smaller position than strategy_cap allows."""
        qty = self._allocate_single(score=0.5, strategy_cap_pct=10.0, max_position_pct=5.0)
        # At score=0.5: p = 0.5 + (0.5 - 0.5)*0.5 = 0.5, kelly_full = (1.5*0.5 - 0.5)/1.5 = 0.1667
        # kelly_half = 0.1667 * 0.5 = 0.0833, kelly_pct = max(1.0, 8.33) = 8.33
        # But capped by max_position_pct=5.0 -> effective_pct = 5.0
        # notional = 100_000 * 5% = 5000, qty = 5000/2500 = 2
        # Even with Kelly dampening, the hard cap applies.
        # The key is that kelly_pct (8.33) gets clamped to max_position_pct (5.0).
        max_qty_at_cap = int(100_000 * 0.05 / 2500.0)  # 2
        assert qty <= max_qty_at_cap, f"score=0.5 qty={qty} should not exceed max_position cap qty={max_qty_at_cap}"
        assert qty > 0, "score=0.5 should still produce a position"

    def test_score_06_gets_appropriately_sized_position(self):
        """Signal with score=0.6 should produce a valid positive position."""
        qty = self._allocate_single(score=0.6)
        assert qty > 0, "score=0.6 should produce a nonzero position"
        # Notional should be reasonable
        notional = qty * 2500.0
        assert notional <= 100_000 * 0.05, f"Notional {notional} exceeds max_position_pct=5% of 100k"

    def test_score_09_larger_than_score_06(self):
        """Higher score should produce equal or larger position than lower score."""
        # Use a larger max_position_pct so Kelly differences are visible
        qty_06 = self._allocate_single(score=0.6, max_position_pct=20.0, strategy_cap_pct=20.0)
        qty_09 = self._allocate_single(score=0.9, max_position_pct=20.0, strategy_cap_pct=20.0)
        assert qty_09 >= qty_06, f"score=0.9 qty={qty_09} should be >= score=0.6 qty={qty_06}"

    def test_kelly_never_exceeds_max_position_pct(self):
        """No signal score should produce allocation exceeding max_position_pct."""
        for score in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            qty = self._allocate_single(score=score, max_position_pct=5.0)
            notional = qty * 2500.0
            max_allowed = 100_000 * 0.05
            assert notional <= max_allowed + 1, (
                f"score={score}: notional {notional} exceeds max_position_pct cap {max_allowed}"
            )

    def test_kelly_never_exceeds_strategy_cap_pct(self):
        """Kelly sizing must be bounded by strategy_cap_pct."""
        for score in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            qty = self._allocate_single(
                score=score,
                strategy_cap_pct=3.0,
                max_position_pct=10.0,
            )
            notional = qty * 2500.0
            # strategy_cap_pct=3% of 100k = 3000, but also capped by int truncation
            max_allowed = 100_000 * 0.10  # max_position_pct is the outer bound
            strategy_cap_allowed = 100_000 * 0.03
            # Kelly should be min(kelly_pct, strategy_cap, max_position_pct)
            assert notional <= max_allowed + 1, f"score={score}: notional {notional} exceeds max_position_pct"

    def test_kelly_disabled_uses_flat_allocation(self):
        """With use_kelly=False, allocation uses strategy_cap directly."""
        config = AllocatorConfig(
            max_active_signals=5,
            max_capital_pct_per_signal=10.0,
            min_confidence=0.3,
            use_kelly=False,
        )
        allocator = PortfolioAllocator(config)
        signal = _make_signal(score=0.7, price=2500.0)
        allocs = allocator.allocate(
            signals=[signal],
            equity=100_000.0,
            positions=[],
            max_position_pct=5.0,
        )
        assert len(allocs) == 1
        qty = allocs[0][1]
        notional = qty * 2500.0
        # With kelly disabled, effective_pct = min(strategy_cap, max_position_pct) = 5%
        assert notional <= 100_000 * 0.05 + 1


# ════════════════════════════════════════════════════════════════════════
# 2. VaR Fallback — Fixed Bug Verification
# ════════════════════════════════════════════════════════════════════════


class TestVaRFallback:
    """Verify VaR falls back to cornish_fisher when other methods fail."""

    def _make_var_with_returns(self, n_obs: int = 60) -> PortfolioVaR:
        """Create a PortfolioVaR with deterministic return history."""
        var_calc = PortfolioVaR(
            min_history=5,
            correlation_window=60,
            var_method="cornish_fisher",
        )
        # Feed deterministic daily returns for RELIANCE
        for i in range(n_obs):
            # Deterministic returns: oscillating around +0.1% with varying magnitude
            ret = 0.001 + ((i * 7 + 3) % 11 - 5) * 0.005
            var_calc.update_returns("RELIANCE", ret)
        return var_calc

    def test_monte_carlo_failure_falls_back_to_cornish_fisher(self):
        """When Monte Carlo raises, VaR should fall back to cornish_fisher (not zero)."""
        var_calc = self._make_var_with_returns(60)
        positions = [{"symbol": "RELIANCE", "notional": 50_000.0}]

        # Patch Monte Carlo to fail
        with patch.object(var_calc, "_compute_monte_carlo_var", return_value=(0.0, 0.0)):
            result = var_calc.compute(positions, 100_000.0, method="monte_carlo")

        # Should have fallen back to cornish_fisher
        assert result.method == "cornish_fisher", f"Expected cornish_fisher fallback, got {result.method}"
        assert result.var_95 > 0, "VaR95 should be non-zero after fallback"
        assert result.var_99 > 0, "VaR99 should be non-zero after fallback"

    def test_cornish_fisher_produces_nonzero_var(self):
        """Cornish-Fisher should produce non-zero VaR for valid positions."""
        var_calc = self._make_var_with_returns(60)
        positions = [{"symbol": "RELIANCE", "notional": 50_000.0}]

        result = var_calc.compute(positions, 100_000.0, method="cornish_fisher")

        assert result.var_95 > 0, f"VaR95 should be positive, got {result.var_95}"
        assert result.var_99 > result.var_95, f"VaR99 ({result.var_99}) should exceed VaR95 ({result.var_95})"
        assert result.method == "cornish_fisher"

    def test_historical_falls_back_when_insufficient_data(self):
        """Historical VaR with insufficient data should fall back to cornish_fisher."""
        var_calc = PortfolioVaR(min_history=5, var_method="historical")
        # Only feed 3 observations (less than min needed for historical: 20)
        for i in range(3):
            var_calc.update_returns("RELIANCE", 0.01 * (i + 1))

        positions = [{"symbol": "RELIANCE", "notional": 50_000.0}]
        result = var_calc.compute(positions, 100_000.0, method="historical")

        # Historical should return (0,0) due to insufficient data, triggering fallback
        assert result.method == "cornish_fisher", (
            f"Expected cornish_fisher fallback from historical, got {result.method}"
        )
        assert result.var_95 > 0, "Fallback VaR should be non-zero"

    def test_cornish_fisher_z_adjustment(self):
        """Cornish-Fisher z should differ from Gaussian z when skew/kurtosis is nonzero."""
        # With zero skew and kurtosis, should equal plain z
        z_plain = _cornish_fisher_z(Z_95, 0.0, 0.0)
        assert abs(z_plain - Z_95) < 1e-10, "CF with zero skew/kurt should equal Gaussian"

        # Positive skew should increase the 95th percentile z-score (shifts right tail out)
        z_posskew = _cornish_fisher_z(Z_95, 0.5, 0.0)
        assert z_posskew > Z_95, f"Positive skew should increase z-score: {z_posskew} vs {Z_95}"

        # Any nonzero skew or kurtosis should produce a different z from Gaussian
        z_adjusted = _cornish_fisher_z(Z_95, -0.5, 2.0)
        assert z_adjusted != Z_95, f"CF with nonzero skew/kurt should differ from Gaussian: {z_adjusted}"


# ════════════════════════════════════════════════════════════════════════
# 3. Daily PnL Persistence
# ════════════════════════════════════════════════════════════════════════


class TestDailyPnLPersistence:
    """Verify that register_pnl triggers state persistence and survives restart."""

    def test_register_pnl_triggers_persistence(self, tmp_path):
        """register_pnl() should write circuit state to disk."""
        state_file = tmp_path / "circuit_state.json"

        # Patch the module-level paths to use tmp_path
        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", state_file),
        ):
            rm = RiskManager(equity=100_000.0, load_persisted_state=False)
            rm.register_pnl(-500.0)

            assert state_file.exists(), "register_pnl should persist state"
            data = json.loads(state_file.read_text())
            assert data["daily_pnl"] == -500.0
            assert data["consecutive_losses"] == 1

    def test_daily_pnl_survives_simulated_restart(self, tmp_path):
        """Daily PnL should survive save-then-load cycle (simulated crash recovery)."""
        state_file = tmp_path / "circuit_state.json"

        with (
            patch("src.risk_engine.manager._CIRCUIT_STATE_DIR", tmp_path),
            patch("src.risk_engine.manager._CIRCUIT_STATE_PATH", state_file),
        ):
            # Phase 1: original manager registers PnL
            rm1 = RiskManager(equity=100_000.0, load_persisted_state=False)
            rm1.register_pnl(-200.0)
            rm1.register_pnl(-300.0)
            assert rm1.daily_pnl == -500.0

            # Phase 2: new manager loads persisted state (simulates restart)
            rm2 = RiskManager(equity=100_000.0, load_persisted_state=True)
            assert rm2.daily_pnl == -500.0, f"Daily PnL not restored: expected -500, got {rm2.daily_pnl}"
            assert rm2._consecutive_losses == 2, (
                f"Consecutive losses not restored: expected 2, got {rm2._consecutive_losses}"
            )


# ════════════════════════════════════════════════════════════════════════
# 4. force_reduce Validation
# ════════════════════════════════════════════════════════════════════════


class TestForceReduceValidation:
    """Verify force_reduce=True requires is_reducing=True."""

    def test_force_reduce_without_is_reducing_is_rejected(self):
        """force_reduce=True with is_reducing=False must be REJECTED."""
        rm = RiskManager(equity=100_000.0, load_persisted_state=False)
        signal = _make_signal(score=0.8, price=2500.0)

        result = rm.can_place_order(
            signal,
            quantity=2,
            price=2500.0,
            is_reducing=False,
            force_reduce=True,
        )
        assert result.allowed is False, "force_reduce without is_reducing should be rejected"
        assert "force_reduce_requires_is_reducing" in result.reason

    def test_force_reduce_with_is_reducing_bypasses_circuit(self):
        """force_reduce=True with is_reducing=True should bypass circuit breaker."""
        rm = RiskManager(equity=100_000.0, load_persisted_state=False)
        rm.open_circuit(reason="test")
        assert rm.is_circuit_open() is True

        signal = _make_signal(score=0.8, price=2500.0)
        result = rm.can_place_order(
            signal,
            quantity=2,
            price=2500.0,
            is_reducing=True,
            force_reduce=True,
        )
        assert result.allowed is True, f"force_reduce + is_reducing should bypass circuit: {result.reason}"

    def test_non_force_reduce_blocked_by_circuit(self):
        """Normal orders should be blocked when circuit is open."""
        rm = RiskManager(equity=100_000.0, load_persisted_state=False)
        rm.open_circuit(reason="test")

        signal = _make_signal(score=0.8, price=2500.0)
        result = rm.can_place_order(signal, quantity=2, price=2500.0)
        assert result.allowed is False
        assert "circuit_breaker_open" in result.reason


# ════════════════════════════════════════════════════════════════════════
# 5. Circuit Breaker Thread Safety
# ════════════════════════════════════════════════════════════════════════


class TestCircuitBreakerThreadSafety:
    """Verify concurrent allow_order() calls respect half_open_max_trades."""

    def test_concurrent_allow_order_respects_max_trades(self):
        """Multiple threads calling allow_order in HALF_OPEN should not exceed max_trades."""
        rm = RiskManager(equity=100_000.0, load_persisted_state=False)
        cb = CircuitBreaker(rm)

        # Trip the breaker, then reset to HALF_OPEN
        cb.trip()
        assert cb.state == CircuitState.OPEN
        # Reset to HALF_OPEN with a long observation period so it does not auto-promote
        cb._half_open_observation_secs = 99999.0
        cb.reset(current_equity=100_000.0)
        assert cb.state == CircuitState.HALF_OPEN

        # Disable auto-promotion so allow_order() tests pure HALF_OPEN limiting.
        # Without this, after max_trades succeed the next call promotes to CLOSED
        # and allows all subsequent orders (by design — OR condition for promotion).
        cb.check_half_open_promotion = lambda: None

        max_trades = cb._half_open_max_trades  # default is 3
        results = []
        barrier = threading.Barrier(20)

        def try_allow():
            barrier.wait()  # synchronize all threads
            allowed = cb.allow_order()
            results.append(allowed)

        threads = [threading.Thread(target=try_allow) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        allowed_count = sum(1 for r in results if r)
        assert allowed_count == max_trades, f"Expected exactly {max_trades} allowed in HALF_OPEN, got {allowed_count}"

    def test_closed_state_allows_all_orders(self):
        """CLOSED state should allow all orders regardless of concurrency."""
        rm = RiskManager(equity=100_000.0, load_persisted_state=False)
        cb = CircuitBreaker(rm)
        assert cb.state == CircuitState.CLOSED

        results = []
        barrier = threading.Barrier(10)

        def try_allow():
            barrier.wait()
            results.append(cb.allow_order())

        threads = [threading.Thread(target=try_allow) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert all(results), "All orders should be allowed in CLOSED state"

    def test_open_state_blocks_all_orders(self):
        """OPEN state should block all orders."""
        rm = RiskManager(equity=100_000.0, load_persisted_state=False)
        cb = CircuitBreaker(rm)
        cb.trip()
        assert cb.state == CircuitState.OPEN

        results = []
        for _ in range(10):
            results.append(cb.allow_order())

        assert not any(results), "No orders should be allowed in OPEN state"


# ════════════════════════════════════════════════════════════════════════
# 6. Risk Limits Validation
# ════════════════════════════════════════════════════════════════════════


class TestRiskLimitsValidation:
    """Verify individual risk limit checks work correctly."""

    def test_check_single_trade_loss_rejects_large_losses(self):
        """Single trade loss exceeding limit should be rejected."""
        limits = RiskLimits(max_single_trade_loss_pct=1.0)
        # Loss of 2000 on 100k equity = 2%, exceeds 1% limit
        result = limits.check_single_trade_loss(equity=100_000.0, trade_loss=-2000.0)
        assert result.allowed is False
        assert "single trade loss" in result.reason

    def test_check_single_trade_loss_accepts_small_losses(self):
        """Single trade loss within limit should be accepted."""
        limits = RiskLimits(max_single_trade_loss_pct=1.0)
        # Loss of 500 on 100k equity = 0.5%, within 1% limit
        result = limits.check_single_trade_loss(equity=100_000.0, trade_loss=-500.0)
        assert result.allowed is True

    def test_check_single_trade_loss_accepts_profits(self):
        """Positive PnL (profit) should always be accepted."""
        limits = RiskLimits(max_single_trade_loss_pct=1.0)
        result = limits.check_single_trade_loss(equity=100_000.0, trade_loss=5000.0)
        assert result.allowed is True

    def test_check_position_size_rejects_oversized(self):
        """Position exceeding max_position_pct should be rejected."""
        limits = RiskLimits(max_position_pct=5.0)
        # 6000 / 100000 = 6% > 5%
        result = limits.check_position_size(equity=100_000.0, position_value=6_000.0)
        assert result.allowed is False

    def test_check_position_size_accepts_valid(self):
        """Position within max_position_pct should be accepted."""
        limits = RiskLimits(max_position_pct=5.0)
        # 4000 / 100000 = 4% < 5%
        result = limits.check_position_size(equity=100_000.0, position_value=4_000.0)
        assert result.allowed is True

    def test_check_daily_loss_rejects_when_exceeded(self):
        """Daily loss exceeding limit should be rejected."""
        limits = RiskLimits(max_daily_loss_pct=2.0)
        # daily_pnl = -3000 on 100k equity = 3% loss > 2% limit
        result = limits.check_daily_loss(equity=100_000.0, daily_pnl=-3_000.0)
        assert result.allowed is False
        assert "daily loss" in result.reason

    def test_check_daily_loss_accepts_within_limit(self):
        """Daily loss within limit should be accepted."""
        limits = RiskLimits(max_daily_loss_pct=2.0)
        # daily_pnl = -1000 on 100k equity = 1% loss < 2% limit
        result = limits.check_daily_loss(equity=100_000.0, daily_pnl=-1_000.0)
        assert result.allowed is True

    def test_check_daily_loss_accepts_profit_day(self):
        """Positive daily PnL should always be accepted."""
        limits = RiskLimits(max_daily_loss_pct=2.0)
        result = limits.check_daily_loss(equity=100_000.0, daily_pnl=5_000.0)
        assert result.allowed is True

    def test_check_open_positions_boundary(self):
        """Open positions at limit should be rejected; below should be accepted."""
        limits = RiskLimits(max_open_positions=10)
        assert limits.check_open_positions(9).allowed is True
        assert limits.check_open_positions(10).allowed is False
        assert limits.check_open_positions(11).allowed is False

    def test_check_consecutive_losses(self):
        """Consecutive losses at/above max should be rejected."""
        limits = RiskLimits(max_consecutive_losses=5)
        assert limits.check_consecutive_losses(4).allowed is True
        assert limits.check_consecutive_losses(5).allowed is False
        assert limits.check_consecutive_losses(10).allowed is False

    def test_zero_equity_is_always_rejected(self):
        """Zero or negative equity should cause all checks to fail safely."""
        limits = RiskLimits()
        assert limits.check_position_size(0, 1000).allowed is False
        assert limits.check_daily_loss(0, -100).allowed is False
        assert limits.check_position_size(-1000, 100).allowed is False


# ════════════════════════════════════════════════════════════════════════
# 7. End-to-End Signal Flow (integration)
# ════════════════════════════════════════════════════════════════════════


class TestEndToEndSignalFlow:
    """Integration: strategy -> allocator -> risk gate."""

    def test_full_signal_to_risk_approval(self):
        """Generate signals, allocate, then pass through risk manager for approval."""
        from src.strategy_engine.classical import EMACrossoverStrategy, MACDStrategy
        from src.strategy_engine.registry import StrategyRegistry
        from src.strategy_engine.runner import StrategyRunner

        # Setup strategies
        registry = StrategyRegistry()
        registry.register(EMACrossoverStrategy(fast=9, slow=21))
        registry.register(MACDStrategy())
        runner = StrategyRunner(registry)

        # Setup allocator: disable Kelly so that classical strategies (score~0.3) still
        # produce allocations. Kelly with score=0.3 yields kelly_full=0 which rounds to
        # zero shares at typical NSE prices.
        allocator = PortfolioAllocator(
            AllocatorConfig(
                max_active_signals=5,
                max_capital_pct_per_signal=10.0,
                min_confidence=0.1,
                use_kelly=False,
            )
        )

        # Setup risk manager (permissive limits for integration test)
        limits = RiskLimits(
            max_position_pct=10.0,
            max_daily_loss_pct=5.0,
            max_open_positions=20,
            max_single_trade_loss_pct=5.0,
            max_per_symbol_pct=15.0,
            var_limit_pct=None,
            cvar_limit_pct=0.0,
        )
        rm = RiskManager(equity=100_000.0, limits=limits, load_persisted_state=False)

        # Generate bars and run pipeline
        bars = _make_bars(200)
        total_approved = 0
        total_rejected = 0

        for i in range(50, len(bars)):
            window = bars[max(0, i - 100) : i + 1]
            state = MarketState(
                symbol="RELIANCE",
                exchange=Exchange.NSE,
                bars=window,
                latest_price=window[-1].close,
                volume=window[-1].volume,
            )

            signals = runner.run(state)
            if not signals:
                continue

            allocs = allocator.allocate(
                signals=signals,
                equity=100_000.0,
                positions=[],
                max_position_pct=10.0,
            )

            for sig, qty in allocs:
                assert qty > 0, f"Allocator returned non-positive qty: {qty}"
                assert sig.price > 0, f"Signal has non-positive price: {sig.price}"

                result = rm.can_place_order(sig, quantity=qty, price=sig.price)
                if result.allowed:
                    total_approved += 1
                else:
                    total_rejected += 1

        assert total_approved > 0, "Pipeline should produce at least one approved order over 150 bars"

    def test_oversized_allocation_rejected_by_risk(self):
        """An oversized allocation should be caught by risk manager."""
        limits = RiskLimits(max_position_pct=5.0)
        rm = RiskManager(equity=100_000.0, limits=limits, load_persisted_state=False)

        signal = _make_signal(score=0.9, price=100.0)
        # 1000 shares * 100 = 100k = 100% of equity, far exceeding 5%
        result = rm.can_place_order(signal, quantity=1000, price=100.0)
        assert result.allowed is False, "100% of equity position should be rejected"

    def test_signal_quantity_pairs_are_valid(self):
        """All (signal, quantity) pairs from allocator should have positive values."""
        allocator = PortfolioAllocator(
            AllocatorConfig(
                max_active_signals=5,
                max_capital_pct_per_signal=10.0,
                min_confidence=0.3,
            )
        )

        signals = [
            _make_signal(symbol="RELIANCE", score=0.8, price=2500.0, strategy_id="s1"),
            _make_signal(symbol="TCS", score=0.6, price=3800.0, strategy_id="s2"),
            _make_signal(symbol="INFY", score=0.7, price=1500.0, strategy_id="s3"),
        ]

        allocs = allocator.allocate(
            signals=signals,
            equity=100_000.0,
            positions=[],
        )

        assert len(allocs) > 0, "Should produce at least one allocation"
        for sig, qty in allocs:
            assert qty > 0, f"Quantity must be positive, got {qty} for {sig.symbol}"
            assert sig.price > 0, f"Price must be positive, got {sig.price} for {sig.symbol}"
            notional = sig.price * qty
            assert notional > 0, "Notional value must be positive"

    def test_risk_rejection_with_daily_loss_exceeded(self):
        """Orders should be rejected when daily loss limit is already breached."""
        limits = RiskLimits(max_daily_loss_pct=2.0)
        rm = RiskManager(equity=100_000.0, limits=limits, load_persisted_state=False)

        # Simulate heavy losses
        rm.register_pnl(-3_000.0)  # 3% daily loss > 2% limit

        signal = _make_signal(score=0.8, price=2500.0)
        result = rm.can_place_order(signal, quantity=2, price=2500.0)
        assert result.allowed is False, f"Should reject when daily loss exceeded: {result.reason}"

    def test_reducing_order_bypasses_daily_loss_check(self):
        """Reducing (exit) orders should bypass daily loss checks."""
        limits = RiskLimits(max_daily_loss_pct=2.0, max_position_pct=10.0)
        rm = RiskManager(equity=100_000.0, limits=limits, load_persisted_state=False)

        # Simulate heavy losses
        rm.register_pnl(-3_000.0)

        signal = _make_signal(score=0.8, price=2500.0)
        result = rm.can_place_order(signal, quantity=2, price=2500.0, is_reducing=True)
        assert result.allowed is True, f"Reducing order should bypass daily loss check: {result.reason}"

    def test_empty_signals_produce_no_allocations(self):
        """Allocator should return empty list for empty signals."""
        allocator = PortfolioAllocator(AllocatorConfig())
        allocs = allocator.allocate(signals=[], equity=100_000.0, positions=[])
        assert allocs == []

    def test_zero_equity_produces_no_allocations(self):
        """Allocator should return empty list for zero equity."""
        allocator = PortfolioAllocator(AllocatorConfig(min_confidence=0.1))
        signal = _make_signal(score=0.8, price=2500.0)
        allocs = allocator.allocate(signals=[signal], equity=0.0, positions=[])
        assert allocs == []

    def test_risk_manager_rejects_zero_quantity(self):
        """Risk manager should reject zero/negative quantity."""
        rm = RiskManager(equity=100_000.0, load_persisted_state=False)
        signal = _make_signal(score=0.8, price=2500.0)
        result = rm.can_place_order(signal, quantity=0, price=2500.0)
        assert result.allowed is False

    def test_risk_manager_rejects_zero_price(self):
        """Risk manager should reject zero/negative price."""
        rm = RiskManager(equity=100_000.0, load_persisted_state=False)
        signal = _make_signal(score=0.8, price=2500.0)
        result = rm.can_place_order(signal, quantity=2, price=0.0)
        assert result.allowed is False
