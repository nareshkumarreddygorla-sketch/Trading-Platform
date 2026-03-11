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
        logger.info(
            "Order REGISTERED: id=%s symbol=%s side=%s qty=%s type=%s status=%s price=%s strategy=%s",
            oid, order.symbol,
            order.side.value if hasattr(order.side, "value") else order.side,
            order.quantity,
            order.order_type.value if hasattr(order.order_type, "value") else order.order_type,
            order.status.value if hasattr(order.status, "value") else order.status,
            order.limit_price,
            order.strategy_id or "",
        )
        return True

    async def update_status(self, order_id: str, status: OrderStatus, filled_qty: float = 0.0, avg_price: Optional[float] = None) -> None:
        async with self._lock:
            if order_id not in self._orders:
                logger.warning(
                    "Order TRANSITION FAILED: id=%s not found in lifecycle (target=%s)",
                    order_id, status.value,
                )
                return
            order = self._orders[order_id]
            current = order.status
            if not is_allowed_transition_domain(current, status):
                logger.warning(
                    "Order TRANSITION BLOCKED: id=%s symbol=%s illegal %s -> %s; skipping",
                    order_id, order.symbol, current.value, status.value,
                )
                return
            order.status = status
            order.filled_qty = filled_qty
            if avg_price is not None:
                order.avg_price = avg_price
            # Compute latency for terminal states
            latency_info = ""
            if status in (OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED):
                placed = self._placed_at.get(order_id)
                if placed:
                    lat_ms = (datetime.now(timezone.utc) - placed).total_seconds() * 1000
                    latency_info = f" latency={lat_ms:.0f}ms"
        logger.info(
            "Order TRANSITION: id=%s symbol=%s %s -> %s filled=%s/%s avg_price=%s%s",
            order_id, order.symbol, current.value, status.value,
            filled_qty, order.quantity, avg_price, latency_info,
        )

    async def get_order(self, order_id: str) -> Optional[Order]:
        async with self._lock:
            return self._orders.get(order_id)

    async def count_active(self) -> int:
        """Count orders that are not terminal (PENDING, LIVE, PARTIALLY_FILLED). Used for reservation slot count."""
        async with self._lock:
            return sum(1 for o in self._orders.values() if getattr(o, "status", None) in ACTIVE_STATUSES)

    def latency_ms(self, order_id: str) -> Optional[float]:
        """Time from place to now (or to fill time if stored).

        BUG 58 NOTE: This reads _placed_at without acquiring _lock. This is safe
        under CPython's GIL for simple dict reads (dict.__contains__ and dict.__getitem__
        are atomic at the bytecode level). No code change needed.
        """
        if order_id not in self._placed_at:
            return None
        return (datetime.now(timezone.utc) - self._placed_at[order_id]).total_seconds() * 1000

    async def get_orders_snapshot(self) -> list:
        """Return a snapshot (list copy) of all orders, safely acquiring the lock.

        BUG 32 FIX: Provides a safe way for external callers (e.g. PaperFillSimulator)
        to read orders without directly accessing the internal _orders dict.
        """
        async with self._lock:
            return list(self._orders.values())

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

    async def sweep_stale_orders(self, max_age_seconds: float = 300.0) -> List[str]:
        """Cancel orders that have been PENDING for longer than max_age_seconds.
        Returns list of cancelled order IDs (phantom order protection)."""
        now = datetime.now(timezone.utc)
        stale_ids: List[str] = []
        async with self._lock:
            for oid, order in list(self._orders.items()):
                if order.status not in ACTIVE_STATUSES:
                    continue
                placed = self._placed_at.get(oid)
                if placed is None:
                    continue
                age = (now - placed).total_seconds()
                if age > max_age_seconds:
                    order.status = OrderStatus.CANCELLED
                    stale_ids.append(oid)
                    logger.warning(
                        "Stale order sweep: cancelled %s (age=%.0fs, status=%s, symbol=%s)",
                        oid, age, order.status.value, order.symbol,
                    )
        return stale_ids

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
