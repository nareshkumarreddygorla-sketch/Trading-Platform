"""
VWAP (Volume-Weighted Average Price) Execution Algorithm.
Distributes child orders according to historical intraday volume profile.

Use case: Orders that are 5-10% of ADV. Better than TWAP for liquid stocks.
"""
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Coroutine, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# NSE intraday volume profile (30-minute buckets, percentage of daily volume)
# Source: empirical NSE data analysis
DEFAULT_NSE_VOLUME_PROFILE = {
    "09:15": 0.12,  # Opening auction + first 30 min: high volume
    "09:45": 0.09,
    "10:15": 0.08,
    "10:45": 0.07,
    "11:15": 0.06,
    "11:45": 0.06,
    "12:15": 0.05,  # Lunch lull
    "12:45": 0.05,
    "13:15": 0.06,
    "13:45": 0.07,
    "14:15": 0.08,
    "14:45": 0.10,
    "15:00": 0.11,  # Closing auction: high volume
}


@dataclass
class VWAPConfig:
    """Configuration for VWAP execution."""
    total_quantity: int
    symbol: str
    side: str
    exchange: str = "NSE"
    start_time: str = "09:15"  # HH:MM IST
    end_time: str = "15:15"    # HH:MM IST
    limit_offset_bps: float = 5.0
    max_participation_pct: float = 10.0
    volume_profile: Optional[Dict[str, float]] = None


@dataclass
class VWAPSlice:
    """A single child order in the VWAP schedule."""
    slice_id: str
    bucket_time: str
    quantity: int
    volume_weight: float
    status: str = "PENDING"
    order_id: Optional[str] = None
    fill_price: float = 0.0
    fill_qty: int = 0


@dataclass
class VWAPExecution:
    """Tracks the overall VWAP execution."""
    exec_id: str
    config: VWAPConfig
    slices: List[VWAPSlice] = field(default_factory=list)
    status: str = "CREATED"
    total_filled: int = 0
    total_submitted: int = 0  # Qty submitted (fills tracked via FillListener)
    achieved_vwap: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class VWAPAlgorithm:
    """
    Volume-Weighted Average Price execution algorithm.

    Distributes a parent order according to historical intraday volume patterns.
    Heavier participation during high-volume periods (open/close),
    lighter during low-volume periods (mid-day).
    """

    def __init__(self, submit_order_fn: Callable[..., Coroutine]):
        self._submit = submit_order_fn
        self._active: Dict[str, VWAPExecution] = {}
        self._custom_profiles: Dict[str, Dict[str, float]] = {}

    def set_volume_profile(self, symbol: str, profile: Dict[str, float]) -> None:
        """Set a custom intraday volume profile for a symbol."""
        # Normalize to sum to 1.0
        total = sum(profile.values())
        if total > 0:
            self._custom_profiles[symbol] = {k: v / total for k, v in profile.items()}

    def create_schedule(self, config: VWAPConfig) -> VWAPExecution:
        """Create VWAP execution schedule based on volume profile."""
        exec_id = f"vwap_{uuid.uuid4().hex[:12]}"

        # Get volume profile
        profile = config.volume_profile or self._custom_profiles.get(
            config.symbol, DEFAULT_NSE_VOLUME_PROFILE
        )

        # Filter buckets within start/end time
        active_buckets = {}
        for bucket_time, weight in sorted(profile.items()):
            if config.start_time <= bucket_time <= config.end_time:
                active_buckets[bucket_time] = weight

        if not active_buckets:
            active_buckets = profile  # fallback to full profile

        # Normalize weights
        total_weight = sum(active_buckets.values())
        normalized = {k: v / total_weight for k, v in active_buckets.items()}

        # Create slices
        slices = []
        allocated = 0
        bucket_list = sorted(normalized.items())

        for i, (bucket_time, weight) in enumerate(bucket_list):
            if i == len(bucket_list) - 1:
                qty = config.total_quantity - allocated  # remainder
            else:
                qty = max(1, int(config.total_quantity * weight))
                allocated += qty

            slices.append(VWAPSlice(
                slice_id=f"{exec_id}_v{i:03d}",
                bucket_time=bucket_time,
                quantity=qty,
                volume_weight=weight,
            ))

        execution = VWAPExecution(exec_id=exec_id, config=config, slices=slices)
        self._active[exec_id] = execution

        logger.info(
            "VWAP schedule: %s %s %d shares in %d buckets (%s-%s)",
            config.side, config.symbol, config.total_quantity,
            len(slices), config.start_time, config.end_time,
        )
        return execution

    async def execute(
        self,
        execution: VWAPExecution,
        get_market_price: Optional[Callable] = None,
    ) -> VWAPExecution:
        """Execute the VWAP schedule."""
        config = execution.config
        execution.status = "RUNNING"
        execution.start_time = datetime.now(timezone.utc)

        total_cost = 0.0
        total_filled = 0
        total_submitted = 0

        for i, slice_order in enumerate(execution.slices):
            if execution.status == "CANCELLED":
                break

            try:
                # Get limit price
                limit_price = None
                if get_market_price:
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

                order_type = "LIMIT" if limit_price else "MARKET"
                order_id = await self._submit(
                    symbol=config.symbol,
                    side=config.side,
                    quantity=slice_order.quantity,
                    order_type=order_type,
                    limit_price=limit_price,
                    exchange=config.exchange,
                )

                slice_order.order_id = order_id
                # BUG 12 FIX: LIMIT orders do not fill immediately. Mark as
                # SUBMITTED/PENDING. Real fill confirmation should come from
                # the fill handler (FillListener / PaperFillSimulator).
                slice_order.status = "SUBMITTED"
                total_submitted += slice_order.quantity

                # NOTE: total_filled stays 0 here — actual fills are tracked
                # asynchronously via the FillListener / fill handler pipeline.
                # Consumers should use total_submitted to know how much was sent.

                logger.info(
                    "VWAP bucket %s: %s %s x%d (%.0f%% weight) @ %s",
                    slice_order.bucket_time, config.side, config.symbol,
                    slice_order.quantity, slice_order.volume_weight * 100,
                    f"{limit_price:.2f}" if limit_price else "MARKET",
                )
            except Exception as e:
                slice_order.status = "FAILED"
                logger.error("VWAP bucket %s failed: %s", slice_order.bucket_time, e)

            # Wait until next bucket (roughly 30 min intervals)
            if i < len(execution.slices) - 1:
                # ~30 min between buckets for NSE profile
                await asyncio.sleep(30 * 60 / len(execution.slices))

        # total_filled stays 0 — actual fills arrive via FillListener.
        # total_submitted reflects how much was sent to the broker.
        execution.total_filled = total_filled
        execution.total_submitted = total_submitted
        execution.achieved_vwap = total_cost / total_filled if total_filled > 0 else 0.0
        execution.status = "SUBMITTED"
        execution.end_time = datetime.now(timezone.utc)

        logger.info(
            "VWAP done: %s %s submitted %d/%d (fills pending via FillListener)",
            config.side, config.symbol, total_submitted,
            config.total_quantity,
        )
        return execution

    async def cancel(self, exec_id: str) -> bool:
        """Cancel a running VWAP execution."""
        execution = self._active.get(exec_id)
        if execution and execution.status == "RUNNING":
            execution.status = "CANCELLED"
            return True
        return False
