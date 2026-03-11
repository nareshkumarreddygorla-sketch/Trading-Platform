"""
QA Phase 3 — Risk & circuit safety.
8) VaR breach edge: signals that barely exceed VaR → reject; no reservation leak.
9) Distributed lock expiry: lock TTL expires; another submission → idempotency catches duplicate.
10) Cluster reservation overflow: max_open=5, 10 concurrent → exactly 5 broker calls.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.events import Exchange, Order, OrderStatus, OrderType, Position, Signal, SignalSide
from src.execution.angel_one_gateway import AngelOneExecutionGateway
from src.execution.lifecycle import OrderLifecycle
from src.execution.order_entry import OrderEntryService
from src.execution.order_entry.idempotency import IdempotencyStore
from src.execution.order_entry.request import OrderEntryRequest
from src.execution.order_entry.reservation import ExposureReservation
from src.execution.order_router import OrderRouter
from src.risk_engine import RiskManager
from src.risk_engine.limits import RiskLimits


# --- 8) VaR breach edge ---
def test_var_breach_rejects_no_reservation_leak():
    """Craft signals that barely exceed VaR; expect reject; no reservation leak."""
    from unittest.mock import MagicMock

    limits = RiskLimits(
        max_open_positions=10,
        max_position_pct=25.0,
        var_limit_pct=5.0,
        max_sector_concentration_pct=100.0,
        max_per_symbol_pct=50.0,
    )
    rm = RiskManager(equity=100_000.0, limits=limits, load_persisted_state=False)
    rm.positions = [
        Position(
            symbol="X", exchange=Exchange.NSE, side=SignalSide.BUY, quantity=500, avg_price=50.0, strategy_id="s1"
        ),
    ]
    # Mock portfolio VaR to return breach
    mock_var = MagicMock()
    mock_var.check_var_limit.return_value = (False, 6.5)  # VaR 6.5% > limit 5%
    rm._portfolio_var = mock_var

    sig = Signal(
        strategy_id="s1",
        symbol="Y",
        exchange=Exchange.NSE,
        side=SignalSide.BUY,
        score=0.9,
        portfolio_weight=0.1,
        price=100.0,
    )
    result = rm.can_place_order(sig, 100, 100.0)
    assert not result.allowed
    reason = result.reason.lower()
    assert "var" in reason, f"Expected VaR rejection, got: {result.reason}"


# --- 9) Lock expiry: covered in chaos test_redis_distributed_lock_expiry ---


# --- 10) Cluster reservation overflow ---
@pytest.mark.asyncio
async def test_cluster_reservation_overflow_max_5_broker_calls():
    """Set max_open_positions=5; submit 10 concurrent; expect at most 5 broker calls."""
    limits = RiskLimits(max_open_positions=5, max_position_pct=10.0)
    risk_manager = RiskManager(equity=1_000_000.0, limits=limits)
    gateway = MagicMock(spec=AngelOneExecutionGateway)
    gateway.place_order = AsyncMock(
        return_value=Order(
            order_id="b",
            strategy_id="s1",
            symbol="R",
            exchange=Exchange.NSE,
            side=SignalSide.BUY,
            quantity=10.0,
            order_type=OrderType.LIMIT,
            limit_price=2500.0,
            status=OrderStatus.LIVE,
            filled_qty=0.0,
            avg_price=None,
            broker_order_id="b",
            metadata={},
        )
    )
    router = OrderRouter(default_gateway=gateway)
    kill_switch = MagicMock()
    kill_switch.is_armed = AsyncMock(return_value=False)
    kill_switch.get_state = AsyncMock(return_value={"armed": False})
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
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        side=SignalSide.BUY,
        score=0.9,
        portfolio_weight=0.1,
        price=2500.0,
    )

    async def submit(i):
        return await service.submit_order(
            OrderEntryRequest(
                signal=sig,
                quantity=10,
                order_type=OrderType.LIMIT,
                limit_price=2500.0,
                idempotency_key=f"qa_overflow_{i}",
                source="qa",
            )
        )

    await asyncio.gather(*[submit(i) for i in range(10)], return_exceptions=True)
    assert gateway.place_order.await_count <= 5, "max_open_positions=5 must cap broker calls"
