"""
Live market data service: connect to feed, push ticks to TickToBarAggregator,
exponential reconnect, health status. Triggers safe_mode on feed unhealthy.
Does NOT remove REST broker logic. Autonomous loop pauses when feed unhealthy;
manual trading remains available.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable, List, Optional

from src.core.events import Exchange, Tick

logger = logging.getLogger(__name__)

# If no tick received for this many seconds, feed is unhealthy
FEED_STALE_SECONDS = 120
RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY = 60.0


def _normalize_ts(ts) -> datetime:
    if ts is None:
        return datetime.now(timezone.utc)
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return datetime.now(timezone.utc)


class MarketDataService:
    """
    Wraps a market data connector: connect, subscribe, stream ticks into aggregator.
    Exponential reconnect; health via last_tick_ts; optional on_feed_unhealthy callback.
    """

    def __init__(
        self,
        connector,
        bar_cache,
        aggregator,
        symbols: List[str],
        *,
        on_feed_unhealthy: Optional[Callable[[], None]] = None,
        feed_stale_seconds: float = FEED_STALE_SECONDS,
    ):
        self.connector = connector
        self.bar_cache = bar_cache
        self.aggregator = aggregator
        self.symbols = list(symbols)
        self.on_feed_unhealthy = on_feed_unhealthy
        self.feed_stale_seconds = feed_stale_seconds
        self._connected = False
        self._last_tick_ts: Optional[datetime] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._reconnect_delay = RECONNECT_BASE_DELAY
        self._health_check_task: Optional[asyncio.Task] = None

    def is_healthy(self) -> bool:
        if not self._connected:
            return False
        if self._last_tick_ts is None:
            return True  # just connected, give time
        age = (datetime.now(timezone.utc) - self._last_tick_ts).total_seconds()
        return age < self.feed_stale_seconds

    def get_status(self) -> dict:
        return {
            "connected": self._connected,
            "healthy": self.is_healthy(),
            "last_tick_ts": self._last_tick_ts.isoformat() if self._last_tick_ts else None,
            "symbols": list(self.symbols),
            "feed_stale_seconds": self.feed_stale_seconds,
        }

    async def _consume_ticks(self) -> None:
        try:
            stream = self.connector.stream_ticks()
            async for tick in stream:
                self._last_tick_ts = _normalize_ts(getattr(tick, "ts", None)) or datetime.now(timezone.utc)
                if not hasattr(tick, "ts") or tick.ts is None or not hasattr(tick.ts, "timestamp"):
                    tick = Tick(
                        symbol=tick.symbol,
                        exchange=tick.exchange,
                        price=tick.price,
                        size=tick.size,
                        ts=self._last_tick_ts,
                    )
                self.aggregator.push_tick(tick)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Market data tick stream error: %s", e)
            self._connected = False

    async def _health_loop(self) -> None:
        while self._running:
            await asyncio.sleep(30)
            if not self._running:
                break
            if self._connected and not self.is_healthy() and self.on_feed_unhealthy:
                logger.warning("Market feed unhealthy (stale); triggering safe_mode")
                try:
                    self.on_feed_unhealthy()
                except Exception as e:
                    logger.exception("on_feed_unhealthy failed: %s", e)

    async def _run(self) -> None:
        while self._running:
            try:
                await self.connector.connect()
                self._connected = True
                self._reconnect_delay = RECONNECT_BASE_DELAY
                await self.connector.subscribe_ticks(self.symbols)
                await self._consume_ticks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Market data service connection error: %s", e)
                self._connected = False
            if not self._running:
                break
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(RECONNECT_MAX_DELAY, self._reconnect_delay * 2)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())
        self._health_check_task = asyncio.create_task(self._health_loop())
        logger.info("MarketDataService started for symbols=%s", self.symbols)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
        try:
            await self.connector.disconnect()
        except Exception as e:
            logger.debug("Connector disconnect: %s", e)
        self._connected = False
