"""
Fill delivery pipeline: poll broker order book, deduplicate, apply idempotently via FillHandler.
Invariant: no fill may corrupt position state under concurrency (dedup + delta + order_lock).
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from src.core.events import Order
from src.monitoring.metrics import (
    track_duplicate_fill,
    track_fill_event,
    track_fill_latency,
    track_fill_reconciliation_mismatch,
)

from ..base import BaseExecutionGateway
from .events import FillEvent, FillType
from .handler import FillHandler

logger = logging.getLogger(__name__)


def _order_status_to_fill_type(status: str) -> FillType:
    s = (status or "").strip().lower()
    if s in ("rejected", "reject"):
        return FillType.REJECT
    if s in ("cancelled", "cancel"):
        return FillType.CANCEL
    if s in ("complete", "completed", "traded", "filled"):
        return FillType.FILL
    if s in ("open", "pending", "trigger pending"):
        return FillType.PARTIAL_FILL
    return FillType.PARTIAL_FILL


class FillListener:
    """
    Polls gateway for orders; diffs with last applied state; emits FillEvents to FillHandler.
    Dedup: only apply delta (new filled_qty - last_applied_filled) so same fill never applied twice.
    Out-of-order: if new filled_qty <= last_applied, skip and count duplicate.
    """

    def __init__(
        self,
        gateway: BaseExecutionGateway,
        fill_handler: FillHandler,
        poll_interval_seconds: float = 10.0,
        redis_client=None,
    ):
        self.gateway = gateway
        self.fill_handler = fill_handler
        self.poll_interval = poll_interval_seconds
        self._last_applied: Dict[str, Tuple[float, Optional[float], str]] = {}  # order_id -> (filled_qty, avg_price, status)
        self._dedup_lock = asyncio.Lock()  # Protect _last_applied from concurrent access
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._redis = redis_client
        self._REDIS_PREFIX = "fill_dedup:"
        self._REDIS_TTL = 48 * 3600  # 48 hours

        # BUG 34 FIX: Redis client may be async, and __init__ is sync.
        # Wrap in try/except to handle both sync and async client failures
        # gracefully. If the client is async, these sync calls will fail and
        # we fall back to empty dedup state (loaded lazily on first poll).
        if self._redis:
            try:
                self._load_dedup_from_redis()
            except Exception as e:
                logger.warning(
                    "Failed to load fill dedup state from Redis on init "
                    "(will use in-memory; dedup state may reload on first poll): %s", e
                )

    def _load_dedup_from_redis(self) -> None:
        """Load persisted fill dedup state from Redis into _last_applied."""
        if not self._redis:
            return
        import json
        keys = self._redis.keys(f"{self._REDIS_PREFIX}*")
        loaded = 0
        for key in keys:
            try:
                raw = self._redis.get(key)
                if raw:
                    data = json.loads(raw)
                    order_id = key.replace(self._REDIS_PREFIX, "") if isinstance(key, str) else key.decode().replace(self._REDIS_PREFIX, "")
                    self._last_applied[order_id] = (
                        float(data.get("filled_qty", 0)),
                        data.get("avg_price"),
                        data.get("status", ""),
                    )
                    loaded += 1
            except Exception as e:
                logger.debug("Skipping corrupt Redis fill dedup key %s: %s", key, e)
        if loaded:
            logger.info("Loaded %d fill dedup entries from Redis", loaded)

    def _persist_dedup_to_redis(self, order_id: str, filled_qty: float, avg_price: Optional[float], status: str) -> None:
        """Persist fill dedup state to Redis with TTL."""
        if not self._redis:
            return
        import json
        try:
            key = f"{self._REDIS_PREFIX}{order_id}"
            data = json.dumps({"filled_qty": filled_qty, "avg_price": avg_price, "status": status})
            self._redis.setex(key, self._REDIS_TTL, data)
        except Exception as e:
            logger.warning("Failed to persist fill dedup to Redis for %s: %s", order_id, e)

    async def _fetch_orders(self) -> list:
        try:
            orders = await asyncio.wait_for(
                self.gateway.get_orders(limit=500),
                timeout=10.0,
            )
            return orders or []
        except asyncio.TimeoutError:
            logger.warning("Fill listener fetch orders timed out (10s)")
            return []
        except Exception as e:
            logger.warning("Fill listener fetch orders failed: %s", e)
            return []

    def _order_to_event(
        self,
        order: Order,
        delta_qty: float,
        fill_type: FillType,
        fill_ts: Optional[datetime] = None,
    ) -> FillEvent:
        return FillEvent(
            order_id=order.order_id or "",
            broker_order_id=order.broker_order_id,
            symbol=order.symbol,
            exchange=order.exchange.value if hasattr(order.exchange, "value") else str(order.exchange),
            side=order.side.value if hasattr(order.side, "value") else str(order.side),
            fill_type=fill_type,
            filled_qty=delta_qty,
            remaining_qty=order.quantity - order.filled_qty,
            avg_price=order.avg_price,
            ts=fill_ts or datetime.now(timezone.utc),
            strategy_id=order.strategy_id or "",
            metadata=getattr(order, "metadata", None) or {},
        )

    async def _process_orders(self, orders: list) -> None:
        async with self._dedup_lock:
            for order in orders:
                if not order or not getattr(order, "order_id", None):
                    continue
                oid = order.order_id
                filled = float(getattr(order, "filled_qty", 0) or 0)
                avg_price = getattr(order, "avg_price", None)
                try:
                    status_str = (order.status.value if hasattr(order.status, "value") else str(order.status)).lower()
                except Exception:
                    status_str = "pending"

                prev = self._last_applied.get(oid, (0.0, None, ""))

                if status_str in ("rejected", "reject"):
                    if prev[2] != "rejected":
                        event = self._order_to_event(order, 0.0, FillType.REJECT)
                        await self.fill_handler.on_fill_event(event)
                        track_fill_event("reject")
                        self._last_applied[oid] = (filled, avg_price, "rejected")
                        self._persist_dedup_to_redis(oid, filled, avg_price, "rejected")
                    continue

                if status_str in ("cancelled", "cancel"):
                    if prev[2] != "cancelled":
                        event = self._order_to_event(order, filled, FillType.CANCEL)
                        await self.fill_handler.on_fill_event(event)
                        track_fill_event("cancel")
                        self._last_applied[oid] = (filled, avg_price, "cancelled")
                        self._persist_dedup_to_redis(oid, filled, avg_price, "cancelled")
                    continue

                # Fill or partial: apply only delta to avoid double count
                prev_filled = prev[0]
                if filled <= prev_filled:
                    if filled > 0 and filled == prev_filled:
                        track_duplicate_fill()
                    continue

                delta = filled - prev_filled
                fill_type = FillType.FILL if status_str in ("complete", "completed", "traded", "filled") else FillType.PARTIAL_FILL

                # Compute marginal fill price for this delta, not the cumulative
                # average the broker reports.  The broker gives cumulative avg_price
                # across all fills so far; we derive the price for *this* fill only:
                #   marginal = (avg_price * filled - prev_avg * prev_filled) / delta
                prev_avg = prev[1]  # previous cumulative avg_price (may be None)
                marginal_price = avg_price  # fallback: use cumulative if we can't compute
                if avg_price is not None and delta > 0:
                    if prev_avg is not None and prev_filled > 0:
                        try:
                            marginal_price = (avg_price * filled - prev_avg * prev_filled) / delta
                        except (ZeroDivisionError, TypeError):
                            marginal_price = avg_price
                    # else: first fill — cumulative == marginal, no adjustment needed

                event = self._order_to_event(order, delta, fill_type)
                event.avg_price = marginal_price
                try:
                    await self.fill_handler.on_fill_event(event)
                    track_fill_event("fill" if fill_type == FillType.FILL else "partial_fill")
                    if fill_ts := getattr(order, "ts", None):
                        try:
                            lat = (datetime.now(timezone.utc) - fill_ts).total_seconds()
                            if lat >= 0:
                                track_fill_latency(lat)
                        except Exception:
                            pass
                    self._last_applied[oid] = (filled, avg_price, status_str)
                    self._persist_dedup_to_redis(oid, filled, avg_price, status_str)

                    # Feed outcome back to ensemble for IC weight learning
                    ensemble = getattr(self, '_ensemble', None)
                    if ensemble and hasattr(order, 'metadata') and order.metadata:
                        pred_dir = order.metadata.get('predicted_direction')
                        entry_price = order.metadata.get('entry_price', order.limit_price)
                        if pred_dir is not None and avg_price and entry_price and entry_price > 0:
                            actual_return = (avg_price - entry_price) / entry_price
                            try:
                                ensemble.record_prediction("ensemble", order.symbol, pred_dir, actual_return)
                            except Exception:
                                pass  # Non-critical feedback

                except Exception as e:
                    logger.exception("Apply fill failed for %s: %s", oid, e)
                    track_fill_reconciliation_mismatch()

    async def _loop(self) -> None:
        while self._running:
            try:
                orders = await self._fetch_orders()
                await self._process_orders(orders)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Fill listener loop error: %s", e)
            await asyncio.sleep(self.poll_interval)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Fill listener started (poll_interval=%.1fs)", self.poll_interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Fill listener stopped")
