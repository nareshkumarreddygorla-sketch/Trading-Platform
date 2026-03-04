"""Async persistence service: symbol-level locking + executor for throughput and no lost updates."""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from src.core.events import Order, OrderStatus

from .order_repo import OrderRepository
from .position_repo import PositionRepository, PositionConcurrentUpdateError
from .risk_snapshot_repo import RiskSnapshotRepository

logger = logging.getLogger(__name__)

# Bounded lock pool: hash(symbol, exchange, side) % N so different symbols can persist in parallel
POSITION_LOCK_POOL_SIZE = 256
# OCC retries: configurable; second conflict raises after this many attempts
OCC_RETRY_COUNT = 3


class PersistenceService:
    """
    Async facade over OrderRepository and PositionRepository.
    Symbol-level locking: persist_fill for (symbol, exchange, side) acquires a lock from a bounded pool,
    then runs in a multi-threaded executor. Same symbol serialized; different symbols scale.
    """

    def __init__(
        self,
        order_repo: Optional[OrderRepository] = None,
        position_repo: Optional[PositionRepository] = None,
        risk_snapshot_repo: Optional[RiskSnapshotRepository] = None,
        _write_executor: Optional[ThreadPoolExecutor] = None,
    ):
        self._order_repo = order_repo or OrderRepository()
        self._position_repo = position_repo or PositionRepository()
        self._risk_snapshot_repo = risk_snapshot_repo or RiskSnapshotRepository()
        self._write_executor = _write_executor or ThreadPoolExecutor(max_workers=4, thread_name_prefix="persist")
        self._position_locks: list[asyncio.Lock] = [asyncio.Lock() for _ in range(POSITION_LOCK_POOL_SIZE)]

    def _position_lock_for(self, symbol: str, exchange: str, side: str) -> asyncio.Lock:
        """Return the lock for this (symbol, exchange, side) from bounded pool."""
        key = hash((symbol, exchange, side)) % POSITION_LOCK_POOL_SIZE
        return self._position_locks[key]

    def _run_write_sync(self, sync_fn: Callable[[], Any]) -> Any:
        """Run sync DB write in executor."""
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(self._write_executor, sync_fn)

    async def persist_order(self, order: Order, idempotency_key: Optional[str] = None, initial_status: str = "NEW") -> None:
        """Persist order. initial_status NEW or SUBMITTING (write-ahead before broker)."""
        def _do():
            self._order_repo.create_order(order, idempotency_key=idempotency_key, initial_status=initial_status)

        await self._run_write_sync(_do)

    async def persist_order_submitting(self, order: Order, idempotency_key: Optional[str] = None) -> None:
        """Write-ahead: persist order with status SUBMITTING before broker call."""
        await self.persist_order(order, idempotency_key=idempotency_key, initial_status="SUBMITTING")

    def update_order_after_broker_ack_sync(self, order_id: str, broker_order_id: Optional[str]) -> bool:
        """Transition SUBMITTING -> NEW after broker accepts. Returns True if updated."""
        return self._order_repo.update_order_after_broker_ack(order_id, broker_order_id, new_status="NEW")

    def list_submitting_orders_sync(self):
        """Return orders in SUBMITTING state (for recovery reconcile)."""
        return self._order_repo.list_submitting_orders()

    def update_order_status_sync(self, order_id: str, status: OrderStatus, filled_qty: float = 0.0, avg_price: Optional[float] = None) -> bool:
        """Sync update order status (e.g. SUBMITTING -> REJECTED on broker failure). Returns True if updated."""
        return self._order_repo.update_order_status(order_id, status, filled_qty=filled_qty, avg_price=avg_price)

    async def persist_fill(
        self,
        order_id: str,
        status: OrderStatus,
        filled_qty: float,
        avg_price: Optional[float],
        symbol: str,
        exchange: str,
        side: str,
        strategy_id: Optional[str] = None,
    ) -> None:
        """Update order status, record event, and upsert position in one transaction. Symbol-level lock for position."""
        from .database import session_scope

        def _do():
            with session_scope() as session:
                self._order_repo.update_order_status(
                    order_id, status, filled_qty=filled_qty, avg_price=avg_price, session=session
                )
                if filled_qty > 0 and avg_price is not None and status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                    self._position_repo.upsert_from_fill(
                        symbol, exchange, side, filled_qty, avg_price, strategy_id, session=session
                    )
            if status == OrderStatus.FILLED:
                try:
                    from src.monitoring.metrics import track_orders_filled_total
                    track_orders_filled_total()
                except Exception:
                    pass

        lock = self._position_lock_for(symbol, exchange, side)
        async with lock:
            last_err = None
            for attempt in range(OCC_RETRY_COUNT):
                try:
                    await self._run_write_sync(_do)
                    return
                except PositionConcurrentUpdateError as e:
                    last_err = e
                    logger.warning("Position OCC conflict (attempt %s/%s): %s", attempt + 1, OCC_RETRY_COUNT, e)
            if last_err:
                try:
                    from src.monitoring.metrics import track_orders_fill_persist_failed_total
                    track_orders_fill_persist_failed_total()
                except Exception:
                    pass
                raise last_err

    def get_order_sync(self, order_id: str):
        return self._order_repo.get_by_order_id(order_id)

    def list_orders_paginated_sync(self, limit: int = 50, offset: int = 0, status: Optional[str] = None, strategy_id: Optional[str] = None):
        return self._order_repo.list_orders_paginated(limit=limit, offset=offset, status=status, strategy_id=strategy_id)

    def list_positions_sync(self):
        return self._position_repo.list_positions()

    def get_risk_snapshot_sync(self) -> Optional[tuple]:
        """Return (equity, daily_pnl) for cold start restore, or None."""
        return self._risk_snapshot_repo.get_latest()

    def save_risk_snapshot_sync(self, equity: float, daily_pnl: float) -> None:
        """Persist current equity and daily_pnl for cold start restore."""
        self._risk_snapshot_repo.save(equity, daily_pnl)
