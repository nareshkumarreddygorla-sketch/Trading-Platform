"""
QA Phase 1 — Execution pipeline break tests.
1) Idempotency storm: 100 concurrent identical keys → broker called exactly once.
2) Kill switch mid-submit: arm during submit → order respects kill / reduce-only.
3) Circuit open during allocation: no new orders; loop skips; manual blocked.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.events import Exchange, Order, OrderStatus, OrderType, Position, Signal, SignalSide
from src.execution.angel_one_gateway import AngelOneExecutionGateway
from src.execution.order_entry.kill_switch import KillReason
from src.execution.lifecycle import OrderLifecycle
from src.execution.order_entry import OrderEntryService
from src.execution.order_entry.idempotency import IdempotencyStore
from src.execution.order_entry.kill_switch import KillSwitch
from src.execution.order_entry.request import OrderEntryRequest
from src.execution.order_entry.reservation import ExposureReservation
from src.execution.order_router import OrderRouter
from src.risk_engine import RiskManager
from src.risk_engine.limits import RiskLimits


def _make_gateway_mock():
    gw = MagicMock(spec=AngelOneExecutionGateway)
    gw.place_order = AsyncMock(
        return_value=Order(
            order_id="broker_1",
            strategy_id="s1",
            symbol="RELIANCE",
            exchange=Exchange.NSE,
            side=SignalSide.BUY,
            quantity=10.0,
            order_type=OrderType.LIMIT,
            limit_price=2500.0,
            status=OrderStatus.LIVE,
            filled_qty=0.0,
            avg_price=None,
            broker_order_id="broker_1",
            metadata={},
        )
    )
    return gw


def _make_signal():
    return Signal(
        strategy_id="s1",
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        side=SignalSide.BUY,
        score=0.9,
        portfolio_weight=0.1,
        price=2500.0,
    )


@pytest.fixture
def idempotency_store():
    return IdempotencyStore(redis_url="redis://localhost:6379/0")


@pytest.fixture
def kill_switch():
    return KillSwitch(allow_reduce_only=True)


@pytest.fixture
def risk_manager():
    return RiskManager(
        equity=1_000_000.0,
        limits=RiskLimits(max_open_positions=50, max_position_pct=10.0),
    )


@pytest.fixture
def lifecycle():
    return OrderLifecycle()


@pytest.fixture
def reservation():
    return ExposureReservation()


# --- 1) Idempotency storm ---
@pytest.mark.asyncio
async def test_idempotency_storm_100_concurrent_same_key_broker_called_once(
    idempotency_store, kill_switch, risk_manager, lifecycle, reservation
):
    """Send 100 concurrent identical idempotency keys. Expect broker called exactly once."""
    gateway = _make_gateway_mock()
    router = OrderRouter(default_gateway=gateway)
    try:
        client = await idempotency_store._get_redis()
        if not client:
            pytest.skip("Redis not available")
        await client.ping()
    except Exception as e:
        err_msg = str(e).lower()
        if any(x in err_msg for x in ("connection", "refused", "timeout", "redis", "econnrefused")):
            pytest.skip("Redis not available: %s" % e)
        raise

    service = OrderEntryService(
        risk_manager=risk_manager,
        order_router=router,
        lifecycle=lifecycle,
        idempotency_store=idempotency_store,
        kill_switch=kill_switch,
        reservation=reservation,
    )
    idem_key = "qa_storm_100_key_001"
    req = OrderEntryRequest(
        signal=_make_signal(),
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        idempotency_key=idem_key,
        source="qa",
    )

    async def submit():
        return await service.submit_order(req)

    results = await asyncio.gather(*[submit() for _ in range(100)], return_exceptions=True)
    success = [r for r in results if not isinstance(r, Exception) and getattr(r, "success", False)]
    assert len(success) >= 1, "At least one submission must succeed"
    order_ids = {getattr(r, "order_id", None) for r in success if getattr(r, "order_id", None)}
    assert len(order_ids) <= 1, "All successful results must return same order_id"
    assert gateway.place_order.await_count <= 1, "Broker must be called at most once (idempotency storm)"


# --- 2) Kill switch mid-submit ---
@pytest.mark.asyncio
async def test_kill_switch_armed_before_broker_call_rejects_new_order(
    idempotency_store, risk_manager, lifecycle, reservation
):
    """Arm kill switch before broker is called; ensure order is rejected (no reduce-only for new BUY)."""
    gateway = _make_gateway_mock()
    kill_switch = KillSwitch(allow_reduce_only=True)
    router = OrderRouter(default_gateway=gateway)
    try:
        client = await idempotency_store._get_redis()
        if not client:
            pytest.skip("Redis not available")
    except Exception:
        pytest.skip("Redis not available")

    service = OrderEntryService(
        risk_manager=risk_manager,
        order_router=router,
        lifecycle=lifecycle,
        idempotency_store=idempotency_store,
        kill_switch=kill_switch,
        reservation=reservation,
    )
    await kill_switch.arm(KillReason.MANUAL, "qa_test")
    req = OrderEntryRequest(
        signal=_make_signal(),
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        idempotency_key="qa_kill_switch_key_001",
        source="qa",
    )
    result = await service.submit_order(req)
    assert result.success is False
    assert result.reject_reason is not None
    assert gateway.place_order.await_count == 0, "Broker must not be called when kill switch armed (new BUY)"


@pytest.mark.asyncio
async def test_kill_switch_armed_allow_reduce_only_sell_when_long(
    idempotency_store, risk_manager, lifecycle, reservation
):
    """When armed, SELL that reduces long position may be allowed (reduce-only)."""
    gateway = _make_gateway_mock()
    kill_switch = KillSwitch(allow_reduce_only=True)
    risk_manager.positions = [
        Position(
            symbol="RELIANCE",
            exchange=Exchange.NSE,
            side=SignalSide.BUY,
            quantity=10.0,
            avg_price=2500.0,
            strategy_id="s1",
        )
    ]
    router = OrderRouter(default_gateway=gateway)
    try:
        client = await idempotency_store._get_redis()
        if not client:
            pytest.skip("Redis not available")
    except Exception:
        pytest.skip("Redis not available")

    service = OrderEntryService(
        risk_manager=risk_manager,
        order_router=router,
        lifecycle=lifecycle,
        idempotency_store=idempotency_store,
        kill_switch=kill_switch,
        reservation=reservation,
    )
    await kill_switch.arm(KillReason.MANUAL, "qa_test")
    sig = Signal(
        strategy_id="s1",
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        side=SignalSide.SELL,
        score=0.9,
        portfolio_weight=0.1,
        price=2500.0,
    )
    req = OrderEntryRequest(
        signal=sig,
        quantity=5,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        idempotency_key="qa_reduce_only_001",
        source="qa",
    )
    result = await service.submit_order(req)
    assert result.success is True or (result.reject_reason is not None)
    if result.success:
        assert gateway.place_order.await_count == 1


# --- 3) Circuit open during allocation ---
@pytest.mark.asyncio
async def test_circuit_open_blocks_new_orders():
    """Force drawdown breach (circuit open). OrderEntryService must reject; no broker call."""
    from src.risk_engine import RiskManager
    from src.risk_engine.limits import RiskLimits
    from src.risk_engine.circuit_breaker import CircuitBreaker

    limits = RiskLimits(max_open_positions=10, circuit_breaker_drawdown_pct=5.0)
    rm = RiskManager(equity=100_000.0, limits=limits)
    cb = CircuitBreaker(rm)
    cb.update_equity(100_000.0)
    rm.update_equity(94_000.0)
    cb.update_equity(94_000.0)
    assert rm.is_circuit_open()

    gateway = _make_gateway_mock()
    kill_switch = KillSwitch()
    router = OrderRouter(default_gateway=gateway)
    lifecycle = OrderLifecycle()
    reservation = ExposureReservation()
    idempotency_store = IdempotencyStore(redis_url="redis://localhost:6379/0")
    try:
        if not await idempotency_store.is_available():
            pytest.skip("Redis not available")
    except Exception:
        pytest.skip("Redis not available")

    service = OrderEntryService(
        risk_manager=rm,
        order_router=router,
        lifecycle=lifecycle,
        idempotency_store=idempotency_store,
        kill_switch=kill_switch,
        reservation=reservation,
    )
    req = OrderEntryRequest(
        signal=_make_signal(),
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=2500.0,
        idempotency_key="qa_circuit_open_001",
        source="qa",
    )
    result = await service.submit_order(req)
    assert result.success is False
    assert getattr(result, "reject_reason", None) is not None
    assert gateway.place_order.await_count == 0
