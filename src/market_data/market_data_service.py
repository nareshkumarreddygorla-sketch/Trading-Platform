"""
Live market data service: connect to feed, push ticks to TickToBarAggregator,
exponential reconnect, health status. Triggers safe_mode on feed unhealthy.
Does NOT remove REST broker logic. Autonomous loop pauses when feed unhealthy;
manual trading remains available.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime

from src.core.events import Tick
from src.data_pipeline.tick_validator import TickValidator
from src.market_data.timestamp import normalize_ts

logger = logging.getLogger(__name__)

# If no tick received for this many seconds, feed is unhealthy
FEED_STALE_SECONDS = 120
RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY = 60.0
# Grace period: allow this many seconds after start before requiring a tick
SILENT_CONNECTION_GRACE_SECONDS = 30.0


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
        symbols: list[str],
        *,
        on_feed_unhealthy: Callable[[], None] | None = None,
        feed_stale_seconds: float = FEED_STALE_SECONDS,
        tick_validator: TickValidator | None = None,
    ):
        self.connector = connector
        self.bar_cache = bar_cache
        self.aggregator = aggregator
        self.symbols = list(symbols)
        self.on_feed_unhealthy = on_feed_unhealthy
        self.feed_stale_seconds = feed_stale_seconds
        self._tick_validator = tick_validator or TickValidator()
        self._connected = False
        self._last_tick_ts: datetime | None = None
        self._running = False
        self._task: asyncio.Task | None = None
        self._reconnect_delay = RECONNECT_BASE_DELAY
        self._health_check_task: asyncio.Task | None = None
        self._started_at: datetime | None = None

    def is_healthy(self) -> bool:
        if not self._connected:
            return False
        if self._last_tick_ts is None:
            # No tick ever received — check if we've exceeded the grace period
            started = self._started_at
            if started is not None:
                elapsed = (datetime.now(UTC) - started).total_seconds()
                if elapsed > SILENT_CONNECTION_GRACE_SECONDS:
                    logger.warning(
                        "Silent connection detected: connected but no ticks received after %.0fs (grace=%ds)",
                        elapsed,
                        int(SILENT_CONNECTION_GRACE_SECONDS),
                    )
                    return False
            return True  # still within grace period
        age = (datetime.now(UTC) - self._last_tick_ts).total_seconds()
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
                self._last_tick_ts = normalize_ts(getattr(tick, "ts", None))
                if not hasattr(tick, "ts") or tick.ts is None or not hasattr(tick.ts, "timestamp"):
                    tick = Tick(
                        symbol=tick.symbol,
                        exchange=tick.exchange,
                        price=tick.price,
                        size=tick.size,
                        ts=self._last_tick_ts,
                    )

                # Validate tick before passing to aggregator
                vr = self._tick_validator.validate_tick(
                    symbol=tick.symbol,
                    price=tick.price,
                    volume=tick.size,
                    timestamp=tick.ts,
                )
                if not vr.is_valid:
                    logger.debug(
                        "Tick rejected for %s: price=%.2f reasons=%s",
                        tick.symbol,
                        tick.price,
                        [r.value for r in vr.reject_reasons],
                    )
                    continue

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
        self._started_at = datetime.now(UTC)
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
