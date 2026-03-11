"""
Iceberg Order Execution Algorithm.
Shows only a fraction of total quantity to the market.
Replenishes after each partial fill.

Use case: Orders > 10% of ADV. Minimizes market impact.
"""

import asyncio
import logging
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class IcebergConfig:
    """Configuration for iceberg execution."""

    total_quantity: int
    symbol: str
    side: str
    exchange: str = "NSE"
    display_qty: int = 0  # Visible quantity (0 = auto-calculate)
    display_pct: float = 10.0  # % of total to show
    limit_price: float | None = None
    limit_offset_bps: float = 3.0
    replenish_delay_seconds: float = 2.0  # Delay between replenishments
    max_replenish_attempts: int = 100


@dataclass
class IcebergExecution:
    """Tracks iceberg execution state."""

    exec_id: str
    config: IcebergConfig
    status: str = "CREATED"
    display_qty: int = 0
    total_filled: int = 0
    total_submitted: int = 0  # Qty submitted (fills tracked via FillListener)
    total_remaining: int = 0
    replenish_count: int = 0
    avg_fill_price: float = 0.0
    child_orders: list[dict] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None


class IcebergAlgorithm:
    """
    Iceberg order execution: hide true order size from the market.

    Only a small "display quantity" is visible at any time.
    When that slice fills, a new one is automatically placed.
    Random variation in display size prevents detection.
    """

    def __init__(self, submit_order_fn: Callable[..., Coroutine]):
        self._submit = submit_order_fn
        self._active: dict[str, IcebergExecution] = {}

    def create_execution(self, config: IcebergConfig) -> IcebergExecution:
        """Create iceberg execution plan."""
        exec_id = f"ice_{uuid.uuid4().hex[:12]}"

        display_qty = config.display_qty
        if display_qty <= 0:
            display_qty = max(1, int(config.total_quantity * config.display_pct / 100))

        execution = IcebergExecution(
            exec_id=exec_id,
            config=config,
            display_qty=display_qty,
            total_remaining=config.total_quantity,
        )
        self._active[exec_id] = execution

        logger.info(
            "Iceberg created: %s %s total=%d display=%d (%.1f%%)",
            config.side,
            config.symbol,
            config.total_quantity,
            display_qty,
            config.display_pct,
        )
        return execution

    async def execute(
        self,
        execution: IcebergExecution,
        get_market_price: Callable | None = None,
    ) -> IcebergExecution:
        """
        Execute iceberg order with automatic replenishment.

        After each visible slice fills, place a new one until
        total quantity is filled.
        """
        import random

        config = execution.config
        execution.status = "RUNNING"
        execution.start_time = datetime.now(UTC)

        total_cost = 0.0
        total_filled = 0
        total_submitted = 0
        remaining = config.total_quantity

        for attempt in range(config.max_replenish_attempts):
            if execution.status == "CANCELLED" or remaining <= 0:
                break

            # Randomize display size (+/- 20%)
            variation = random.uniform(0.8, 1.2)
            slice_qty = max(1, min(int(execution.display_qty * variation), remaining))

            # Get limit price
            limit_price = config.limit_price
            if limit_price is None and get_market_price:
                try:
                    price = await get_market_price(config.symbol)
                    if price and price > 0:
                        offset = config.limit_offset_bps / 10000
                        if config.side == "BUY":
                            limit_price = round(price * (1 + offset), 2)
                        else:
                            limit_price = round(price * (1 - offset), 2)
                except Exception:
                    pass

            try:
                order_type = "LIMIT" if limit_price else "MARKET"
                order_id = await self._submit(
                    symbol=config.symbol,
                    side=config.side,
                    quantity=slice_qty,
                    order_type=order_type,
                    limit_price=limit_price,
                    exchange=config.exchange,
                )

                # Track child order — mark as SUBMITTED, not FILLED.
                # Actual fill confirmation arrives via FillListener.
                child = {
                    "order_id": order_id,
                    "quantity": slice_qty,
                    "limit_price": limit_price,
                    "status": "SUBMITTED",
                    "attempt": attempt,
                }
                execution.child_orders.append(child)

                # Track submitted qty; do NOT assume fills.
                # total_filled stays 0 — real fills come via FillListener.
                total_submitted += slice_qty
                remaining -= slice_qty
                execution.replenish_count = attempt + 1

                logger.debug(
                    "Iceberg slice %d: %s %s x%d (remaining: %d)",
                    attempt + 1,
                    config.side,
                    config.symbol,
                    slice_qty,
                    remaining,
                )

            except Exception as e:
                logger.error("Iceberg slice failed: %s", e)
                execution.child_orders.append(
                    {
                        "quantity": slice_qty,
                        "status": "FAILED",
                        "error": str(e),
                        "attempt": attempt,
                    }
                )

            # Delay between replenishments (randomized to avoid detection)
            if remaining > 0:
                delay = config.replenish_delay_seconds * random.uniform(0.7, 1.5)
                await asyncio.sleep(delay)

        # total_filled stays 0 — actual fills arrive via FillListener.
        # total_submitted reflects how much was sent to the broker.
        execution.total_filled = total_filled
        execution.total_submitted = total_submitted
        execution.total_remaining = remaining
        execution.avg_fill_price = total_cost / total_filled if total_filled > 0 else 0.0
        execution.status = "SUBMITTED" if remaining == 0 else "PARTIAL"
        execution.end_time = datetime.now(UTC)

        logger.info(
            "Iceberg done: %s %s submitted %d/%d in %d slices (fills pending via FillListener)",
            config.side,
            config.symbol,
            total_submitted,
            config.total_quantity,
            execution.replenish_count,
        )
        return execution

    async def cancel(self, exec_id: str) -> bool:
        """Cancel iceberg execution."""
        execution = self._active.get(exec_id)
        if execution and execution.status == "RUNNING":
            execution.status = "CANCELLED"
            return True
        return False
