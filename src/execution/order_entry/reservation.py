"""
Atomic exposure reservation: reserve before broker call; rollback if broker fails.
Prevents two concurrent orders from both passing risk check and exceeding limits.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class ReservedExposure:
    """One reserved exposure (pending order)."""

    order_id: str
    symbol: str
    exchange: str
    side: str
    quantity: float
    price: float
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


class ExposureReservation:
    """
    Reserve exposure (position value + slot) before calling broker.
    If broker fails or order is rejected, release reservation.
    Thread-safe: use single lock for all reserve/release.
    """

    def __init__(self):
        self._reservations: dict[str, ReservedExposure] = {}  # order_id -> reserved
        self._lock = asyncio.Lock()

    async def reserve(
        self,
        order_id: str,
        symbol: str,
        exchange: str,
        side: str,
        quantity: int,
        price: float,
        current_positions: list,
        max_open_positions: int,
        max_position_pct: float,
        equity: float,
        active_order_count: int = 0,
    ) -> tuple[bool, str]:
        """
        Reserve one slot and position value. Returns (ok, reason).
        Caller must hold risk lock so that current_positions and counts are consistent.
        active_order_count: number of non-terminal orders in lifecycle (consumes slots).
        """
        async with self._lock:
            reserved_count = len(self._reservations)
            open_count = len(current_positions)
            if open_count + reserved_count + active_order_count >= max_open_positions:
                return False, "max_open_positions_exceeded_with_reservations_and_active_orders"

            position_value = quantity * price
            if equity <= 0:
                return False, "zero_equity"
            pct = 100.0 * position_value / equity
            if pct > max_position_pct:
                return False, "position_size_exceeded"

            self._reservations[order_id] = ReservedExposure(
                order_id=order_id,
                symbol=symbol,
                exchange=exchange,
                side=side,
                quantity=float(quantity),
                price=price,
            )
            return True, ""

    async def release(self, order_id: str) -> None:
        """Release reservation (on broker reject or timeout)."""
        async with self._lock:
            self._reservations.pop(order_id, None)
            logger.debug("Reservation released: order_id=%s", order_id)

    async def commit(self, order_id: str) -> None:
        """Order filled; remove from pending reservations (position will be added by FillHandler)."""
        async with self._lock:
            self._reservations.pop(order_id, None)

    def reserved_count(self) -> int:
        """Current number of reserved slots (for risk check). Use under lock in production."""
        return len(self._reservations)
