"""
Fill delivery pipeline tests: dedup, out-of-order, partial aggregation.
Invariant: no fill may corrupt position state under concurrency.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.events import Exchange, Order, OrderStatus, OrderType, SignalSide
from src.execution.fill_handler import FillHandler, FillListener
from src.execution.lifecycle import OrderLifecycle
from src.risk_engine import RiskManager
from src.risk_engine.limits import RiskLimits


def _make_order(oid: str, symbol: str, filled_qty: float, avg_price: float, status: OrderStatus):
    return Order(
        order_id=oid,
        strategy_id="s1",
        symbol=symbol,
        exchange=Exchange.NSE,
        side=SignalSide.BUY,
        quantity=10.0,
        order_type=OrderType.MARKET,
        status=status,
        filled_qty=filled_qty,
        avg_price=avg_price,
    )


@pytest.fixture
def risk_manager():
    return RiskManager(equity=100_000.0, limits=RiskLimits())


@pytest.fixture
def lifecycle():
    return OrderLifecycle()


@pytest.fixture
def fill_handler(risk_manager, lifecycle):
    return FillHandler(risk_manager=risk_manager, lifecycle=lifecycle, order_lock=None)


@pytest.mark.asyncio
async def test_duplicate_fill_flood_dedup(fill_handler, lifecycle):
    """Same fill reported multiple times: only first application updates position; rest are deduplicated."""
    gateway = MagicMock()
    order = _make_order("o1", "INFY", 5.0, 100.0, OrderStatus.PARTIALLY_FILLED)
    await lifecycle.register(order)
    orders_return = [order]
    gateway.get_orders = AsyncMock(return_value=orders_return)
    listener = FillListener(gateway, fill_handler, poll_interval_seconds=60.0)
    listener._running = True

    await listener._process_orders(orders_return)
    assert (await lifecycle.get_order("o1")) is not None
    positions = [p for p in fill_handler.risk_manager.positions if p.symbol == "INFY" and p.side == SignalSide.BUY]
    assert len(positions) == 1
    assert positions[0].quantity == 5.0

    # Same state again: should not double position (dedup / no delta)
    await listener._process_orders(orders_return)
    positions2 = [p for p in fill_handler.risk_manager.positions if p.symbol == "INFY" and p.side == SignalSide.BUY]
    assert len(positions2) == 1 and positions2[0].quantity == 5.0


@pytest.mark.asyncio
async def test_out_of_order_fill_no_double_apply(fill_handler):
    """Newer fill (10) then older (5): we already applied 10 so older is skipped."""
    gateway = MagicMock()
    listener = FillListener(gateway, fill_handler, poll_interval_seconds=60.0)
    listener._last_applied["o1"] = (10.0, 100.0, "complete")

    orders_new_then_old = [
        _make_order("o1", "INFY", 5.0, 100.0, OrderStatus.PARTIALLY_FILLED),
    ]
    await listener._process_orders(orders_new_then_old)
    # No new delta (5 <= 10), so no new position added
    positions = [p for p in fill_handler.risk_manager.positions if p.symbol == "INFY"]
    assert len(positions) == 0


@pytest.mark.asyncio
async def test_partial_fill_aggregation(fill_handler, lifecycle):
    """5 then 10 filled: two events with delta 5 and 5; position = 10."""
    gateway = MagicMock()
    listener = FillListener(gateway, fill_handler, poll_interval_seconds=60.0)
    o = _make_order("o1", "INFY", 0.0, None, OrderStatus.LIVE)
    await lifecycle.register(o)

    await listener._process_orders(
        [
            _make_order("o1", "INFY", 5.0, 99.0, OrderStatus.PARTIALLY_FILLED),
        ]
    )
    positions = [p for p in fill_handler.risk_manager.positions if p.symbol == "INFY" and p.side == SignalSide.BUY]
    assert len(positions) == 1 and positions[0].quantity == 5.0

    await listener._process_orders(
        [
            _make_order("o1", "INFY", 10.0, 99.5, OrderStatus.FILLED),
        ]
    )
    positions2 = [p for p in fill_handler.risk_manager.positions if p.symbol == "INFY" and p.side == SignalSide.BUY]
    assert len(positions2) == 1 and positions2[0].quantity == 10.0


@pytest.mark.asyncio
async def test_reject_and_cancel_only_once(fill_handler, lifecycle):
    """Reject/cancel status: event emitted once per order, not on every poll."""
    gateway = MagicMock()
    listener = FillListener(gateway, fill_handler, poll_interval_seconds=60.0)
    o2 = _make_order("o2", "RELIANCE", 0.0, None, OrderStatus.REJECTED)
    await lifecycle.register(
        Order(
            order_id="o2",
            strategy_id="s1",
            symbol="RELIANCE",
            exchange=Exchange.NSE,
            side=SignalSide.BUY,
            quantity=10.0,
            order_type=OrderType.MARKET,
            status=OrderStatus.LIVE,
        )
    )
    orders_rejected = [o2]
    await listener._process_orders(orders_rejected)
    await listener._process_orders(orders_rejected)
    # Lifecycle should have o2 as REJECTED; no position
    positions = [p for p in fill_handler.risk_manager.positions if p.symbol == "RELIANCE"]
    assert len(positions) == 0
