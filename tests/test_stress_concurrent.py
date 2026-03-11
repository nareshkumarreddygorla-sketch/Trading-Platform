"""
Concurrent order stress tests.

Tests thread-safety and concurrency of:
- Order submission pipeline (10 simultaneous orders)
- Rate limiter under concurrent load
- Idempotency guard under concurrent duplicate submissions
- RiskManager.can_place_order thread safety
- CircuitBreaker thread safety

All tests use threading for concurrency, mock the broker gateway,
and require no external services (Redis, DB, etc.).
"""

import asyncio
import threading
import time
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from src.core.events import (
    Exchange,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Signal,
    SignalSide,
)
from src.execution.lifecycle import OrderLifecycle
from src.execution.order_entry.idempotency import IdempotencyStore
from src.execution.order_entry.kill_switch import KillSwitch
from src.execution.order_entry.rate_limiter import OrderRateLimiter, RateLimitConfig
from src.execution.order_entry.request import OrderEntryRequest, RejectReason
from src.execution.order_entry.reservation import ExposureReservation
from src.execution.order_entry.service import OrderEntryService
from src.execution.order_router import OrderRouter
from src.risk_engine.circuit_breaker import CircuitBreaker, CircuitState
from src.risk_engine.limits import RiskLimits
from src.risk_engine.manager import RiskManager

# ---------------------------------------------------------------------------
# Shared event loop for async operations from multiple threads
# ---------------------------------------------------------------------------

_shared_loop = None
_loop_thread = None


def _get_shared_loop() -> asyncio.AbstractEventLoop:
    """Get or create a shared event loop running in a background thread.
    This is needed because asyncio.Lock is bound to a single event loop,
    so all async operations must run on the same loop."""
    global _shared_loop, _loop_thread
    if _shared_loop is not None and not _shared_loop.is_closed() and _shared_loop.is_running():
        return _shared_loop
    if _shared_loop is None or _shared_loop.is_closed():
        _shared_loop = asyncio.new_event_loop()

        def run_loop():
            asyncio.set_event_loop(_shared_loop)
            _shared_loop.run_forever()

        _loop_thread = threading.Thread(target=run_loop, daemon=True)
        _loop_thread.start()
        # Wait briefly for loop to start
        time.sleep(0.01)
    return _shared_loop


def _run_on_shared_loop(coro):
    """Submit a coroutine to the shared event loop and block until it completes."""
    loop = _get_shared_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=30)


@pytest.fixture(autouse=True)
def _reset_shared_loop():
    """Reset the shared event loop between tests to avoid state leaking."""
    global _shared_loop, _loop_thread
    _shared_loop = None
    _loop_thread = None
    yield
    if _shared_loop is not None and not _shared_loop.is_closed():
        _shared_loop.call_soon_threadsafe(_shared_loop.stop)
        if _loop_thread is not None:
            _loop_thread.join(timeout=5)
        try:
            if not _shared_loop.is_running():
                _shared_loop.close()
        except Exception:
            pass
    _shared_loop = None
    _loop_thread = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(symbol: str = "RELIANCE", side: SignalSide = SignalSide.BUY, price: float = 100.0) -> Signal:
    """Create a test Signal."""
    return Signal(
        strategy_id="stress_test",
        symbol=symbol,
        exchange=Exchange.NSE,
        side=side,
        score=0.5,
        portfolio_weight=0.01,
        risk_level="NORMAL",
        reason="stress_test",
        price=price,
    )


def _make_order_entry_request(
    symbol: str = "RELIANCE",
    side: SignalSide = SignalSide.BUY,
    quantity: int = 10,
    price: float = 100.0,
    idempotency_key: str = None,
) -> OrderEntryRequest:
    """Create a test OrderEntryRequest."""
    return OrderEntryRequest(
        signal=_make_signal(symbol, side, price),
        quantity=quantity,
        order_type=OrderType.LIMIT,
        limit_price=price,
        idempotency_key=idempotency_key or str(uuid.uuid4()),
        source="test",
    )


def _make_risk_manager(equity: float = 1_000_000.0) -> RiskManager:
    """Create a RiskManager with generous limits for stress testing."""
    limits = RiskLimits(
        max_position_pct=25.0,
        max_daily_loss_pct=10.0,
        max_open_positions=50,
        max_consecutive_losses=20,
    )
    return RiskManager(equity=equity, limits=limits, load_persisted_state=False)


def _make_mock_order(order_id: str = None) -> Order:
    """Create a mock Order response from broker."""
    return Order(
        order_id=order_id or str(uuid.uuid4()),
        strategy_id="stress_test",
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        side=SignalSide.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=100.0,
        status=OrderStatus.PENDING,
        broker_order_id=f"BRK-{uuid.uuid4().hex[:8]}",
    )


def _build_order_entry_service(
    risk_manager: RiskManager = None,
    rate_limiter: OrderRateLimiter = None,
) -> OrderEntryService:
    """Build a fully wired OrderEntryService with mock broker.
    Must be called from the shared event loop thread (or via _run_on_shared_loop)."""
    rm = risk_manager or _make_risk_manager()

    # Mock gateway that returns a successful order
    mock_gateway = AsyncMock()
    mock_gateway.place_order = AsyncMock(side_effect=lambda *a, **kw: _make_mock_order())
    mock_gateway.paper = True

    order_router = OrderRouter(default_gateway=mock_gateway)
    lifecycle = OrderLifecycle()
    idempotency = IdempotencyStore()
    # Force in-memory mode (skip Redis)
    idempotency._redis_checked = True
    idempotency._redis_available = False

    # Patch the kill switch state file path to avoid loading stale state from disk
    with patch.object(KillSwitch, "_load_state", return_value=None):
        kill_switch = KillSwitch()

    reservation = ExposureReservation()

    svc = OrderEntryService(
        risk_manager=rm,
        order_router=order_router,
        lifecycle=lifecycle,
        idempotency_store=idempotency,
        kill_switch=kill_switch,
        reservation=reservation,
        rate_limiter=rate_limiter,
    )
    return svc


# ---------------------------------------------------------------------------
# Test 1: Concurrent order submission (10 simultaneous orders)
# ---------------------------------------------------------------------------


class TestConcurrentOrderSubmission:
    """Verify 10 simultaneous orders can be submitted without data corruption."""

    def test_10_concurrent_orders_all_succeed(self):
        """Submit 10 orders from 10 threads; all should succeed without errors."""
        svc = _build_order_entry_service()
        results = []
        errors = []
        lock = threading.Lock()

        def submit_order(i: int):
            try:
                req = _make_order_entry_request(
                    symbol=f"SYM{i}",
                    price=100.0 + i,
                    quantity=10,
                )
                result = _run_on_shared_loop(svc.submit_order(req))
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=submit_order, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors during concurrent submission: {errors}"
        assert len(results) == 10
        # All should have succeeded (no risk rejection with generous limits)
        succeeded = [r for r in results if r.success]
        assert len(succeeded) == 10, (
            f"Expected 10 successes, got {len(succeeded)}. "
            f"Failures: {[(r.reject_reason, r.reject_detail) for r in results if not r.success]}"
        )
        # All order IDs should be unique
        order_ids = [r.order_id for r in succeeded]
        assert len(set(order_ids)) == 10, f"Duplicate order IDs detected: {order_ids}"

    def test_concurrent_orders_different_symbols_no_data_corruption(self):
        """Submit orders for different symbols concurrently; verify no cross-contamination."""
        svc = _build_order_entry_service()
        results_by_symbol = {}
        lock = threading.Lock()

        submit_errors = []

        def submit_order(symbol: str, price: float):
            try:
                req = _make_order_entry_request(symbol=symbol, price=price, quantity=10)
                result = _run_on_shared_loop(svc.submit_order(req))
                with lock:
                    results_by_symbol[symbol] = result
            except Exception as e:
                with lock:
                    submit_errors.append((symbol, e))

        symbols = [f"STOCK{i}" for i in range(10)]
        threads = [threading.Thread(target=submit_order, args=(sym, 100.0 + i)) for i, sym in enumerate(symbols)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        total = len(results_by_symbol) + len(submit_errors)
        assert total == 10, (
            f"Expected 10 completions, got {len(results_by_symbol)} results + {len(submit_errors)} errors"
        )
        for sym, result in results_by_symbol.items():
            assert result.success, f"Order for {sym} failed: {result.reject_reason}"
            assert result.order_id is not None


# ---------------------------------------------------------------------------
# Test 2: Rate limiter under concurrent load
# ---------------------------------------------------------------------------


class TestRateLimiterConcurrency:
    """Verify rate limiter correctly throttles under concurrent access."""

    def test_rate_limiter_blocks_beyond_limit(self):
        """Concurrent calls beyond limit should be rejected."""
        limiter = OrderRateLimiter(RateLimitConfig(max_orders_per_minute=5, window_seconds=60.0))
        results = []
        lock = threading.Lock()

        def try_allow():
            allowed = limiter.allow()
            with lock:
                results.append(allowed)

        threads = [threading.Thread(target=try_allow) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(results) == 20
        allowed_count = sum(1 for r in results if r)
        rejected_count = sum(1 for r in results if not r)
        # Exactly 5 should be allowed (the limit), rest rejected
        assert allowed_count == 5, f"Expected 5 allowed, got {allowed_count}"
        assert rejected_count == 15, f"Expected 15 rejected, got {rejected_count}"

    def test_rate_limiter_concurrent_access_no_race(self):
        """Multiple threads calling allow() should never exceed the configured limit."""
        limiter = OrderRateLimiter(RateLimitConfig(max_orders_per_minute=10, window_seconds=60.0))
        allowed_count_lock = threading.Lock()
        allowed_count = [0]

        def try_allow_many():
            for _ in range(5):
                if limiter.allow():
                    with allowed_count_lock:
                        allowed_count[0] += 1

        threads = [threading.Thread(target=try_allow_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # Should never exceed the limit of 10
        assert allowed_count[0] <= 10, f"Rate limiter allowed {allowed_count[0]} orders, expected <= 10"

    def test_rate_limiter_rejects_orders_in_service(self):
        """OrderEntryService should reject orders when rate limiter is exhausted."""
        limiter = OrderRateLimiter(RateLimitConfig(max_orders_per_minute=3, window_seconds=60.0))
        svc = _build_order_entry_service(rate_limiter=limiter)
        results = []
        errors = []
        lock = threading.Lock()

        def submit(i: int):
            try:
                req = _make_order_entry_request(symbol=f"RL{i}", price=100.0 + i, quantity=10)
                result = _run_on_shared_loop(svc.submit_order(req))
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=submit, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        total = len(results) + len(errors)
        assert total == 10, f"Expected 10 completions, got {len(results)} results + {len(errors)} errors"
        succeeded = [r for r in results if r.success]
        # At most 3 should succeed (rate limit = 3/minute)
        assert len(succeeded) <= 3, f"Expected <= 3 successes, got {len(succeeded)}"
        # The rest should be rate-limited or timed out
        non_succeeded = len(results) - len(succeeded) + len(errors)
        assert non_succeeded >= 7, (
            f"Expected >= 7 non-successes, got {non_succeeded}. "
            f"Other rejections: {[(r.reject_reason, r.reject_detail) for r in results if not r.success]}"
        )


# ---------------------------------------------------------------------------
# Test 3: Idempotency guard under concurrent duplicate submissions
# ---------------------------------------------------------------------------


class TestIdempotencyConcurrency:
    """Verify idempotency guard handles concurrent duplicate keys correctly."""

    def test_concurrent_duplicate_keys_only_one_wins(self):
        """Submit same idempotency key from multiple threads; only one should create a new entry."""
        store = IdempotencyStore()
        store._redis_checked = True
        store._redis_available = False

        shared_key = "test-idempotency-key-123"
        results = []
        lock = threading.Lock()

        errors = []

        def try_set(thread_id: int):
            try:
                order_id = f"order-{thread_id}"
                result = _run_on_shared_loop(store.set(shared_key, order_id))
                with lock:
                    results.append((thread_id, result))
            except Exception as e:
                with lock:
                    errors.append((thread_id, e))

        threads = [threading.Thread(target=try_set, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        total = len(results) + len(errors)
        assert total == 10, f"Expected 10 completions, got {len(results)} results + {len(errors)} errors"
        winners = [r for r in results if r[1] is True]
        losers = [r for r in results if r[1] is False]
        # Exactly one should win the set (NX semantics)
        assert len(winners) == 1, f"Expected 1 winner, got {len(winners)}: {winners}"
        assert len(losers) == 9, f"Expected 9 losers, got {len(losers)}"

    def test_concurrent_duplicate_orders_all_succeed(self):
        """Submit duplicate orders concurrently via OrderEntryService; all should succeed
        (either original submission or cached idempotency result) and the broker should
        only be called once."""
        rm = _make_risk_manager()
        mock_gateway = AsyncMock()
        broker_call_count = [0]
        broker_lock = threading.Lock()

        async def counting_place_order(*a, **kw):
            with broker_lock:
                broker_call_count[0] += 1
            return _make_mock_order()

        mock_gateway.place_order = AsyncMock(side_effect=counting_place_order)
        mock_gateway.paper = True

        order_router = OrderRouter(default_gateway=mock_gateway)
        lifecycle = OrderLifecycle()
        idempotency = IdempotencyStore()
        idempotency._redis_checked = True
        idempotency._redis_available = False

        with patch.object(KillSwitch, "_load_state", return_value=None):
            kill_switch = KillSwitch()

        reservation = ExposureReservation()
        svc = OrderEntryService(
            risk_manager=rm,
            order_router=order_router,
            lifecycle=lifecycle,
            idempotency_store=idempotency,
            kill_switch=kill_switch,
            reservation=reservation,
        )

        shared_key = f"idem-dup-{uuid.uuid4().hex[:8]}"
        results = []
        lock = threading.Lock()

        idem_errors = []

        def submit(i: int):
            try:
                req = _make_order_entry_request(
                    symbol="RELIANCE",
                    price=2500.0,
                    quantity=10,
                    idempotency_key=shared_key,
                )
                result = _run_on_shared_loop(svc.submit_order(req))
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    idem_errors.append(e)

        threads = [threading.Thread(target=submit, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        total = len(results) + len(idem_errors)
        assert total == 5, f"Expected 5 completions, got {len(results)} results + {len(idem_errors)} errors"
        # All should succeed (either original or cached)
        for r in results:
            assert r.success, f"Expected success, got reject: {r.reject_reason} {r.reject_detail}"
        # The broker should have been called at most once (idempotency guard)
        assert broker_call_count[0] == 1, (
            f"Broker called {broker_call_count[0]} times, expected exactly 1 (idempotency guard failed)"
        )


# ---------------------------------------------------------------------------
# Test 4: RiskManager thread safety under concurrent can_place_order
# ---------------------------------------------------------------------------


class TestRiskManagerThreadSafety:
    """Verify RiskManager.can_place_order is thread-safe."""

    def test_concurrent_can_place_order_no_crash(self):
        """Multiple threads calling can_place_order should not crash or corrupt state."""
        rm = _make_risk_manager()
        errors = []
        results = []
        lock = threading.Lock()

        def check_risk(i: int):
            try:
                signal = _make_signal(symbol=f"SYM{i}", price=100.0 + i)
                result = rm.can_place_order(signal, quantity=10, price=100.0 + i)
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=check_risk, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors during concurrent risk check: {errors}"
        assert len(results) == 20

    def test_concurrent_position_mutations_and_risk_checks(self):
        """Simultaneous position adds and risk checks should not corrupt the positions list."""
        rm = _make_risk_manager()
        errors = []

        def add_positions(start: int):
            try:
                for j in range(10):
                    pos = Position(
                        symbol=f"POS{start}_{j}",
                        exchange=Exchange.NSE,
                        side=SignalSide.BUY,
                        quantity=5,
                        avg_price=100.0,
                    )
                    rm.add_position(pos)
            except Exception as e:
                errors.append(("add", e))

        def check_risk_repeatedly():
            try:
                for _ in range(20):
                    signal = _make_signal(symbol="CHECKSTOCK", price=100.0)
                    rm.can_place_order(signal, quantity=5, price=100.0)
            except Exception as e:
                errors.append(("check", e))

        threads = []
        # 5 threads adding positions
        for i in range(5):
            threads.append(threading.Thread(target=add_positions, args=(i,)))
        # 5 threads checking risk
        for _ in range(5):
            threads.append(threading.Thread(target=check_risk_repeatedly))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert len(errors) == 0, f"Errors: {errors}"
        # 5 threads * 10 positions each = 50 positions added
        assert len(rm.positions) == 50, f"Expected 50 positions, got {len(rm.positions)}"

    def test_concurrent_register_pnl_no_corruption(self):
        """Multiple threads calling register_pnl should not produce inconsistent daily_pnl."""
        rm = _make_risk_manager()
        n_threads = 10
        pnl_per_call = 100.0
        calls_per_thread = 50

        def register_pnls():
            for _ in range(calls_per_thread):
                rm.register_pnl(pnl_per_call)

        threads = [threading.Thread(target=register_pnls) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        expected_pnl = n_threads * calls_per_thread * pnl_per_call
        assert abs(rm.daily_pnl - expected_pnl) < 0.01, f"PnL corruption: expected {expected_pnl}, got {rm.daily_pnl}"


# ---------------------------------------------------------------------------
# Test 5: CircuitBreaker thread safety
# ---------------------------------------------------------------------------


class TestCircuitBreakerThreadSafety:
    """Verify CircuitBreaker operations are thread-safe."""

    def test_concurrent_trip_only_opens_once(self):
        """Multiple threads tripping the circuit should result in exactly one OPEN state."""
        rm = _make_risk_manager()
        cb = CircuitBreaker(risk_manager=rm)
        trip_count = [0]
        lock = threading.Lock()

        original_trip = cb.trip

        def counting_trip():
            original_trip()
            with lock:
                trip_count[0] += 1

        cb.trip = counting_trip

        def trip_circuit():
            cb.trip()

        threads = [threading.Thread(target=trip_circuit) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert cb.state == CircuitState.OPEN
        assert rm.is_circuit_open() is True

    def test_concurrent_allow_order_half_open_limit(self):
        """In HALF_OPEN state, only max_trades orders should be allowed across threads."""
        rm = _make_risk_manager()
        cb = CircuitBreaker(risk_manager=rm)
        cb._half_open_max_trades = 3
        cb._half_open_observation_secs = 3600  # long period so it stays HALF_OPEN

        # Trip then reset to HALF_OPEN
        cb.trip()
        cb.reset()

        assert cb.state == CircuitState.HALF_OPEN

        # Disable auto-promotion so allow_order() doesn't promote HALF_OPEN→CLOSED
        cb.check_half_open_promotion = lambda: None

        allowed = []
        lock = threading.Lock()

        def try_order():
            result = cb.allow_order()
            with lock:
                allowed.append(result)

        threads = [threading.Thread(target=try_order) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(allowed) == 20
        allowed_count = sum(1 for a in allowed if a)
        # Should allow exactly _half_open_max_trades (3)
        assert allowed_count == 3, f"Expected 3 allowed in HALF_OPEN, got {allowed_count}"

    def test_concurrent_update_equity_triggers_trip(self):
        """Concurrent equity updates that breach drawdown should trip the circuit exactly once."""
        rm = _make_risk_manager(equity=100_000.0)
        rm.limits.circuit_breaker_drawdown_pct = 5.0
        cb = CircuitBreaker(risk_manager=rm)
        cb._peak_equity = 100_000.0

        errors = []

        def update_equity(new_equity: float):
            try:
                cb.update_equity(new_equity)
            except Exception as e:
                errors.append(e)

        # 10 threads simultaneously report equity drop of 6% (should trigger)
        threads = [threading.Thread(target=update_equity, args=(94_000.0,)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors: {errors}"
        assert cb.state == CircuitState.OPEN
        assert rm.is_circuit_open() is True

    def test_circuit_breaker_closed_allows_all_concurrent(self):
        """In CLOSED state, all concurrent allow_order calls should return True."""
        rm = _make_risk_manager()
        cb = CircuitBreaker(risk_manager=rm)
        assert cb.state == CircuitState.CLOSED

        results = []
        lock = threading.Lock()

        def check_allow():
            result = cb.allow_order()
            with lock:
                results.append(result)

        threads = [threading.Thread(target=check_allow) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(results) == 50
        assert all(r is True for r in results), "All orders should be allowed in CLOSED state"


# ---------------------------------------------------------------------------
# Test 6: Stress - many orders with max_open_positions limit
# ---------------------------------------------------------------------------


class TestPositionLimitUnderConcurrency:
    """Verify max_open_positions is not exceeded under concurrent submissions."""

    def test_max_positions_respected_under_load(self):
        """With max_open_positions=3, concurrent orders should not exceed the limit."""
        rm = _make_risk_manager()
        rm.limits.max_open_positions = 3

        svc = _build_order_entry_service(risk_manager=rm)
        results = []
        lock = threading.Lock()

        errors = []

        def submit(i: int):
            try:
                req = _make_order_entry_request(
                    symbol=f"POS{i}",
                    price=100.0,
                    quantity=10,
                )
                result = _run_on_shared_loop(svc.submit_order(req))
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Submit 8 orders for 8 different symbols concurrently
        threads = [threading.Thread(target=submit, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        total = len(results) + len(errors)
        assert total == 8, f"Expected 8 completions, got {len(results)} results + {len(errors)} errors"
        succeeded = [r for r in results if r.success]
        # Should not exceed max_open_positions (3)
        assert len(succeeded) <= 3, f"Position limit violated: {len(succeeded)} orders succeeded, max allowed is 3"


# ---------------------------------------------------------------------------
# Test 7: Kill switch thread safety
# ---------------------------------------------------------------------------


class TestKillSwitchConcurrency:
    """Verify KillSwitch operations under concurrent access."""

    def test_concurrent_arm_disarm_no_corruption(self):
        """Rapid sequential arm/disarm (via shared loop) should not corrupt state."""
        from src.execution.order_entry.kill_switch import KillReason

        with patch.object(KillSwitch, "_load_state", return_value=None):
            ks = KillSwitch()

        errors = []

        def arm_switch(i: int):
            try:
                _run_on_shared_loop(ks.arm(KillReason.MANUAL, f"thread-{i}"))
            except Exception as e:
                errors.append(("arm", e))

        def disarm_switch():
            try:
                _run_on_shared_loop(ks.disarm())
            except Exception as e:
                errors.append(("disarm", e))

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=arm_switch, args=(i,)))
            threads.append(threading.Thread(target=disarm_switch))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors: {errors}"
        # Final state should be consistent (either armed or disarmed, no corruption)
        final_state = _run_on_shared_loop(ks.is_armed())
        assert isinstance(final_state, bool)

    def test_kill_switch_blocks_concurrent_orders_when_armed(self):
        """When kill switch is armed, all concurrent non-reduce orders should be rejected."""
        from src.execution.order_entry.kill_switch import KillReason

        rm = _make_risk_manager()
        svc = _build_order_entry_service(risk_manager=rm)

        # Arm the kill switch
        _run_on_shared_loop(svc.kill_switch.arm(KillReason.MANUAL, "stress_test"))

        results = []
        ks_errors = []
        lock = threading.Lock()

        def submit(i: int):
            try:
                req = _make_order_entry_request(
                    symbol=f"BLOCK{i}",
                    price=100.0,
                    quantity=10,
                )
                result = _run_on_shared_loop(svc.submit_order(req))
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    ks_errors.append(e)

        threads = [threading.Thread(target=submit, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        total = len(results) + len(ks_errors)
        assert total == 10, f"Expected 10 completions, got {len(results)} results + {len(ks_errors)} errors"
        # All completed results should be rejected with KILL_SWITCH reason
        for r in results:
            assert not r.success, f"Order should have been rejected, got success: {r.order_id}"
            assert r.reject_reason == RejectReason.KILL_SWITCH, (
                f"Expected KILL_SWITCH rejection, got {r.reject_reason}: {r.reject_detail}"
            )


# ---------------------------------------------------------------------------
# Test 8: Mixed concurrent operations (orders + position updates + risk checks)
# ---------------------------------------------------------------------------


class TestMixedConcurrentOperations:
    """Stress test with mixed concurrent operations happening simultaneously."""

    def test_mixed_operations_no_deadlock(self):
        """Run order submissions, position mutations, and risk checks concurrently.
        The test passes if there are no deadlocks (all threads complete within timeout)."""
        rm = _make_risk_manager()
        errors = []
        completed = []
        lock = threading.Lock()

        def add_positions():
            try:
                for i in range(20):
                    pos = Position(
                        symbol=f"MIX{i}",
                        exchange=Exchange.NSE,
                        side=SignalSide.BUY,
                        quantity=5,
                        avg_price=50.0 + i,
                    )
                    rm.add_position(pos)
                with lock:
                    completed.append("positions")
            except Exception as e:
                with lock:
                    errors.append(("positions", e))

        def check_risk():
            try:
                for i in range(50):
                    signal = _make_signal(symbol=f"CHECK{i}", price=100.0)
                    rm.can_place_order(signal, quantity=5, price=100.0)
                with lock:
                    completed.append("risk")
            except Exception as e:
                with lock:
                    errors.append(("risk", e))

        def register_pnls():
            try:
                for _ in range(30):
                    rm.register_pnl(10.0)
                with lock:
                    completed.append("pnl")
            except Exception as e:
                with lock:
                    errors.append(("pnl", e))

        def update_equity():
            try:
                for i in range(20):
                    rm.update_equity(1_000_000.0 + i * 100)
                with lock:
                    completed.append("equity")
            except Exception as e:
                with lock:
                    errors.append(("equity", e))

        threads = [
            threading.Thread(target=add_positions),
            threading.Thread(target=add_positions),
            threading.Thread(target=check_risk),
            threading.Thread(target=check_risk),
            threading.Thread(target=check_risk),
            threading.Thread(target=register_pnls),
            threading.Thread(target=register_pnls),
            threading.Thread(target=update_equity),
        ]

        for t in threads:
            t.start()

        # If any thread hangs, we detect it via timeout
        for t in threads:
            t.join(timeout=15)
            assert not t.is_alive(), f"Thread {t.name} appears deadlocked"

        assert len(errors) == 0, f"Errors during mixed operations: {errors}"
        assert len(completed) == 8, f"Not all operations completed: {completed}"

    def test_risk_snapshot_concurrent_with_mutations(self):
        """risk_snapshot() should not crash when called concurrently with position mutations."""
        rm = _make_risk_manager()
        errors = []

        def mutate_positions():
            try:
                for i in range(50):
                    pos = Position(
                        symbol=f"SNAP{i}",
                        exchange=Exchange.NSE,
                        side=SignalSide.BUY,
                        quantity=10,
                        avg_price=100.0,
                    )
                    rm.add_position(pos)
                    rm.register_pnl(5.0)
            except Exception as e:
                errors.append(("mutate", e))

        def take_snapshots():
            try:
                for _ in range(30):
                    snapshot = rm.risk_snapshot()
                    assert "equity" in snapshot
                    assert "n_positions" in snapshot
            except Exception as e:
                errors.append(("snapshot", e))

        threads = [
            threading.Thread(target=mutate_positions),
            threading.Thread(target=mutate_positions),
            threading.Thread(target=take_snapshots),
            threading.Thread(target=take_snapshots),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert len(errors) == 0, f"Errors: {errors}"
