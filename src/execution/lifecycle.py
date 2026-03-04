"""Trade lifecycle: persist order state, handle fills/rejections, retries, latency metrics."""
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional

from src.core.events import Order, OrderStatus
from src.execution.lifecycle_transitions import is_allowed_transition_domain

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = (OrderStatus.PENDING, OrderStatus.LIVE, OrderStatus.PARTIALLY_FILLED)


class OrderLifecycle:
    """
    Track order state transitions; log for audit; measure latency.
    Thread-safe via asyncio.Lock for concurrent access from OrderEntryService and FillHandler.
    """

    def __init__(self):
        self._orders: dict[str, Order] = {}
        self._placed_at: dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def register(self, order: Order) -> bool:
        """Register order. Returns False if order_id is empty (caller should handle)."""
        oid = order.order_id
        if not oid or not str(oid).strip():
            logger.error("OrderLifecycle.register: order_id is empty; refusing to register")
            return False
        async with self._lock:
            self._orders[str(oid)] = order
            self._placed_at[str(oid)] = datetime.now(timezone.utc)
        return True

    async def update_status(self, order_id: str, status: OrderStatus, filled_qty: float = 0.0, avg_price: Optional[float] = None) -> None:
        async with self._lock:
            if order_id not in self._orders:
                return
            current = self._orders[order_id].status
            if not is_allowed_transition_domain(current, status):
                logger.warning("Order %s: illegal lifecycle transition %s -> %s; skipping", order_id, current.value, status.value)
                return
            self._orders[order_id].status = status
            self._orders[order_id].filled_qty = filled_qty
            if avg_price is not None:
                self._orders[order_id].avg_price = avg_price
        logger.info("Order %s -> %s filled=%s avg_price=%s", order_id, status.value, filled_qty, avg_price)

    async def get_order(self, order_id: str) -> Optional[Order]:
        async with self._lock:
            return self._orders.get(order_id)

    async def count_active(self) -> int:
        """Count orders that are not terminal (PENDING, LIVE, PARTIALLY_FILLED). Used for reservation slot count."""
        async with self._lock:
            return sum(1 for o in self._orders.values() if getattr(o, "status", None) in ACTIVE_STATUSES)

    def latency_ms(self, order_id: str) -> Optional[float]:
        """Time from place to now (or to fill time if stored)."""
        if order_id not in self._placed_at:
            return None
        return (datetime.now(timezone.utc) - self._placed_at[order_id]).total_seconds() * 1000

    def load_for_recovery(self, orders: List[Order]) -> None:
        """
        Cold start only: repopulate lifecycle from persisted active orders.
        Uses order.ts as placed_at. Must not be used during normal trading.
        """
        for order in orders:
            oid = order.order_id
            if not oid or not str(oid).strip():
                continue
            self._orders[str(oid)] = order
            self._placed_at[str(oid)] = order.ts if order.ts else datetime.now(timezone.utc)
        logger.info("OrderLifecycle loaded %d active orders for recovery", len(orders))

    def list_recent(self, limit: int = 100) -> List[Order]:
        """Return most recently placed orders (by placement time)."""
        if not self._orders:
            return []
        order_ids_by_time = sorted(
            self._placed_at.keys(),
            key=lambda oid: self._placed_at[oid],
            reverse=True,
        )
        return [self._orders[oid] for oid in order_ids_by_time[:limit]]
