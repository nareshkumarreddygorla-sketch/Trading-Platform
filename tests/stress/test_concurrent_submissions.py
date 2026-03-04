"""
Stress tests: many concurrent submit_order calls; idempotency and lock prevent duplicate broker calls.
Requires Redis for distributed lock and cluster reservation (skips if unavailable).
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.events import Exchange, Order, OrderStatus, OrderType, Signal, SignalSide
from src.execution.angel_one_gateway import AngelOneExecutionGateway
from src.execution.lifecycle import OrderLifecycle
from src.execution.order_entry import OrderEntryService
from src.execution.order_entry.idempotency import IdempotencyStore
from src.execution.order_entry.reservation import ExposureReservation
from src.execution.order_router import OrderRouter
from src.risk_engine import RiskManager
from src.risk_engine.limits import RiskLimits


@pytest.fixture
def risk_manager():
    return RiskManager(
        equity=1_000_000.0,
        limits=RiskLimits(max_open_positions=100, max_position_pct=10.0),
    )


@pytest.fixture
def lifecycle():
    return OrderLifecycle()


@pytest.fixture
def idempotency_store():
    return IdempotencyStore(redis_url="redis://localhost:6379/0")


@pytest.fixture
def reservation():
    return ExposureReservation()


@pytest.mark.asyncio
async def test_concurrent_submissions_same_idempotency_key(risk_manager, lifecycle, idempotency_store, reservation):
    """Many concurrent submissions with same idempotency key: only one should succeed to broker."""
    gateway = MagicMock(spec=AngelOneExecutionGateway)
    gateway.place_order = AsyncMock(
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
    router = OrderRouter(default_gateway=gateway)
    kill_switch = MagicMock()
    kill_switch.is_armed = AsyncMock(return_value=False)
    kill_switch.get_state = AsyncMock(return_value={"armed": False})

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

    idem_key = "stress_same_key_001"
    signal = Signal(strategy_id="s1", symbol="RELIANCE", exchange=Exchange.NSE, side=SignalSide.BUY, score=0.9, portfolio_weight=0.05, price=2500.0)
    from src.execution.order_entry.request import OrderEntryRequest

    async def submit():
        return await service.submit_order(
            OrderEntryRequest(signal=signal, quantity=10, order_type=OrderType.LIMIT, limit_price=2500.0, idempotency_key=idem_key, source="stress")
        )

    results = await asyncio.gather(*[submit() for _ in range(50)], return_exceptions=True)
    success = [r for r in results if not isinstance(r, Exception) and getattr(r, "success", False)]
    # With same idempotency key, only first wins; rest are idempotency hits (return existing order_id or reject)
    assert len(success) >= 1
    # Broker should be called at most once (first submission)
    assert gateway.place_order.await_count <= 1


@pytest.mark.asyncio
async def test_concurrent_submissions_different_keys_capped_by_reservation(lifecycle, idempotency_store, reservation):
    """Many concurrent submissions with different keys: risk max_open_positions caps how many pass."""
    risk_manager = RiskManager(
        equity=1_000_000.0,
        limits=RiskLimits(max_open_positions=50, max_position_pct=10.0),
    )
    gateway = MagicMock(spec=AngelOneExecutionGateway)
    gateway.place_order = AsyncMock(
        return_value=Order(
            order_id="broker_x",
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
            broker_order_id="broker_x",
            metadata={},
        )
    )
    router = OrderRouter(default_gateway=gateway)
    kill_switch = MagicMock()
    kill_switch.is_armed = AsyncMock(return_value=False)
    kill_switch.get_state = AsyncMock(return_value={"armed": False})

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

    signal = Signal(strategy_id="s1", symbol="RELIANCE", exchange=Exchange.NSE, side=SignalSide.BUY, score=0.9, portfolio_weight=0.05, price=2500.0)
    from src.execution.order_entry.request import OrderEntryRequest

    async def submit(i):
        return await service.submit_order(
            OrderEntryRequest(signal=signal, quantity=10, order_type=OrderType.LIMIT, limit_price=2500.0, idempotency_key=f"stress_key_{i}", source="stress")
        )

    results = await asyncio.gather(*[submit(i) for i in range(60)], return_exceptions=True)
    success = [r for r in results if not isinstance(r, Exception) and getattr(r, "success", False)]
    # Risk max_open_positions=50: at most 50 can pass risk; rest get RISK_REJECTED
    assert gateway.place_order.await_count <= 50
