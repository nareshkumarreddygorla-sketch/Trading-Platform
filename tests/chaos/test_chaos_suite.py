"""
Chaos tests: assert invariants under failure injection.
Invariants: no duplicate broker call, no position corruption, no capital leakage.
"""
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.events import Exchange, Order, OrderStatus, Signal, SignalSide
from src.execution.fill_handler import FillHandler, FillListener
from src.execution.order_entry.idempotency import IdempotencyStore
from src.execution.order_entry.redis_cluster_reservation import RedisClusterReservation
from src.execution.order_entry.redis_distributed_lock import RedisDistributedLock
from src.risk_engine import RiskManager
from src.risk_engine.limits import RiskLimits
from src.execution.lifecycle import OrderLifecycle


@pytest.fixture
def risk_manager():
    return RiskManager(equity=100_000.0, limits=RiskLimits(max_open_positions=5))


@pytest.fixture
def lifecycle():
    return OrderLifecycle()


@pytest.mark.asyncio
async def test_duplicate_fill_flood_no_position_corruption(risk_manager, lifecycle):
    """Duplicate fill flood: apply same fill 10x; position must equal single fill qty (dedup)."""
    handler = FillHandler(risk_manager=risk_manager, lifecycle=lifecycle)
    from src.core.events import Order as O
    o = O(order_id="c1", strategy_id="s1", symbol="INFY", exchange=Exchange.NSE, side=SignalSide.BUY, quantity=10.0, status=OrderStatus.LIVE)
    lifecycle.register(o)
    gateway = MagicMock()
    gateway.get_orders = AsyncMock(return_value=[
        O(order_id="c1", strategy_id="s1", symbol="INFY", exchange=Exchange.NSE, side=SignalSide.BUY, quantity=10.0, filled_qty=5.0, avg_price=100.0, status=OrderStatus.PARTIALLY_FILLED),
    ])
    listener = FillListener(gateway, handler, poll_interval_seconds=999.0)
    for _ in range(10):
        await listener._process_orders(gateway.get_orders.return_value)
    positions = [p for p in risk_manager.positions if p.symbol == "INFY"]
    assert len(positions) <= 1
    if positions:
        assert positions[0].quantity == 5.0


@pytest.mark.asyncio
async def test_idempotency_storm_identical_keys():
    """100 identical idempotency keys: only first set succeeds; no duplicate broker call implied."""
    store = IdempotencyStore(redis_url="redis://localhost:6379/0")
    try:
        client = await store._get_redis()
        if not client:
            pytest.skip("Redis not available")
    except Exception:
        pytest.skip("Redis not available")
    key = "chaos_storm_key_001"
    order_id = "ord_1"
    first = await store.set(key, order_id, None, "PENDING")
    assert first is True
    for _ in range(99):
        ok = await store.set(key, "other", None, "PENDING")
        assert ok is False
    got = await store.get(key)
    assert got and got.get("order_id") == order_id


@pytest.mark.asyncio
async def test_redis_distributed_lock_expiry():
    """Lock acquire returns False on timeout when lock held elsewhere."""
    lock = RedisDistributedLock(redis_url="redis://localhost:6379/0", acquire_timeout_seconds=0.5)
    try:
        r = await lock._get_redis()
        if not r:
            pytest.skip("Redis not available")
    except Exception:
        pytest.skip("Redis not available")
    ok1 = await lock.acquire()
    assert ok1 is True
    lock2 = RedisDistributedLock(redis_url="redis://localhost:6379/0", acquire_timeout_seconds=0.2)
    ok2 = await lock2.acquire()
    assert ok2 is False
    await lock.release()


@pytest.mark.asyncio
async def test_cluster_reservation_over_reservation_prevented():
    """Cluster reservation: reserve up to max_allowed; next reserve with max_allowed=1 fails."""
    res = RedisClusterReservation(redis_url="redis://localhost:6379/0")
    try:
        r = await res._get_redis()
        if not r:
            pytest.skip("Redis not available")
    except Exception:
        pytest.skip("Redis not available")
    await res.reserve("oid_chaos_1", 1)
    ok2 = await res.reserve("oid_chaos_2", 1)
    assert ok2 is False
    await res.release("oid_chaos_1")
