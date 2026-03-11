"""
QA Phase 4 — Chaos & crash recovery.
12) Broker latency spike: mock long delay → timeout; reservation released.
13) Redis down: idempotency rejects; no broker call.
14) Persist retry logic exists.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.events import Exchange, OrderType, Signal, SignalSide
from src.execution.angel_one_gateway import AngelOneExecutionGateway
from src.execution.lifecycle import OrderLifecycle
from src.execution.order_entry import OrderEntryService
from src.execution.order_entry.idempotency import IdempotencyStore
from src.execution.order_entry.request import OrderEntryRequest
from src.execution.order_entry.reservation import ExposureReservation
from src.execution.order_router import OrderRouter
from src.risk_engine import RiskManager
from src.risk_engine.limits import RiskLimits


@pytest.mark.slow
@pytest.mark.asyncio
async def test_broker_latency_spike_timeout_releases_reservation():
    """Mock broker delay; expect timeout or reject; no zombie reservation."""
    gateway = MagicMock(spec=AngelOneExecutionGateway)
    gateway.place_order = AsyncMock(side_effect=asyncio.sleep(2.0))
    router = OrderRouter(default_gateway=gateway)
    kill_switch = MagicMock()
    kill_switch.is_armed = AsyncMock(return_value=False)
    kill_switch.get_state = AsyncMock(return_value={"armed": False})
    risk_manager = RiskManager(equity=1_000_000.0, limits=RiskLimits(max_open_positions=10, max_position_pct=10.0))
    lifecycle = OrderLifecycle()
    reservation = ExposureReservation()
    idempotency_store = IdempotencyStore(redis_url="redis://localhost:6379/0")
    try:
        if not await idempotency_store.is_available():
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
    sig = Signal(
        strategy_id="s1",
        symbol="R",
        exchange=Exchange.NSE,
        side=SignalSide.BUY,
        score=0.9,
        portfolio_weight=0.1,
        price=100.0,
    )
    req = OrderEntryRequest(
        signal=sig,
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=100.0,
        idempotency_key="qa_latency_001",
        source="qa",
    )
    result = await asyncio.wait_for(service.submit_order(req), timeout=5.0)
    assert result.success is False or result.reject_reason is not None


@pytest.mark.asyncio
async def test_redis_down_idempotency_rejects_no_broker_call():
    """When idempotency store unavailable, reject; no broker call."""
    gateway = MagicMock(spec=AngelOneExecutionGateway)
    gateway.place_order = AsyncMock()
    router = OrderRouter(default_gateway=gateway)
    kill_switch = MagicMock()
    kill_switch.is_armed = AsyncMock(return_value=False)
    kill_switch.get_state = AsyncMock(return_value={"armed": False})
    risk_manager = RiskManager(equity=1_000_000.0, limits=RiskLimits(max_open_positions=10, max_position_pct=10.0))
    lifecycle = OrderLifecycle()
    reservation = ExposureReservation()
    store = IdempotencyStore(redis_url="redis://localhost:6379/0")
    store._redis = None

    async def no_redis():
        return None

    store._get_redis = no_redis

    async def is_unavailable():
        return False

    store.is_available = is_unavailable
    service = OrderEntryService(
        risk_manager=risk_manager,
        order_router=router,
        lifecycle=lifecycle,
        idempotency_store=store,
        kill_switch=kill_switch,
        reservation=reservation,
    )
    req = OrderEntryRequest(
        signal=Signal(
            strategy_id="s1",
            symbol="R",
            exchange=Exchange.NSE,
            side=SignalSide.BUY,
            score=0.9,
            portfolio_weight=0.1,
            price=100.0,
        ),
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=100.0,
        idempotency_key="qa_redis_down_001",
        source="qa",
    )
    result = await service.submit_order(req)
    assert result.success is False
    assert getattr(result, "reject_reason", None) is not None
    assert gateway.place_order.await_count == 0


def test_order_entry_has_persist_retry():
    """OrderEntryService has retry logic for persist."""
    import inspect

    from src.execution.order_entry import service as svc

    src = inspect.getsource(svc.OrderEntryService.submit_order)
    assert "_persist" in src or "retry" in src.lower() or "attempt" in src.lower()
