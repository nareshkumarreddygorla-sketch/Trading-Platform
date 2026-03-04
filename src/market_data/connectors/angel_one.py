"""
Angel One SmartAPI connector for Indian markets (NSE/BSE).
WebSocket for live ticks; REST for historical and order placement.
See: https://smartapi.angelone.in/
"""
import asyncio
import logging
from typing import AsyncIterator, List

from src.core.events import Bar, Exchange, Tick
from .base import BaseMarketDataConnector

logger = logging.getLogger(__name__)


class AngelOneConnector(BaseMarketDataConnector):
    """Angel One SmartAPI: real-time and historical. Stub implements interface; fill with SmartAPI SDK."""

    exchange = Exchange.NSE

    def __init__(self, api_key: str, api_secret: str, token: str, feed: str = "wss://smartapis.angelone.in/smart-stream"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.token = token
        self.feed_url = feed
        self._ws = None
        self._tick_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._bar_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._symbols: List[str] = []
        self._interval: str = "1m"

    async def connect(self) -> None:
        # TODO: SmartAPI WebSocket connect; on_message push to _tick_queue
        logger.info("AngelOne connector connect (stub)")
        await asyncio.sleep(0)

    async def disconnect(self) -> None:
        if self._ws:
            await self._ws.close()
        self._ws = None

    async def subscribe_ticks(self, symbols: list[str]) -> None:
        self._symbols = list(symbols)
        # TODO: send subscription message to SmartAPI WebSocket

    async def subscribe_bars(self, symbols: list[str], interval: str) -> None:
        self._symbols = list(symbols)
        self._interval = interval

    def stream_ticks(self) -> AsyncIterator[Tick]:
        async def _gen():
            while True:
                try:
                    raw = await asyncio.wait_for(self._tick_queue.get(), timeout=30.0)
                    # Convert SmartAPI tick to Tick
                    yield Tick(
                        symbol=raw.get("symbol", ""),
                        exchange=Exchange.NSE,
                        price=float(raw.get("ltp", 0)),
                        size=float(raw.get("volume", 0)),
                        ts=raw.get("ts"),  # normalize to datetime in normalizer
                    )
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.exception(e)
                    break
        return _gen()

    def stream_bars(self) -> AsyncIterator[Bar]:
        async def _gen():
            while True:
                try:
                    raw = await asyncio.wait_for(self._bar_queue.get(), timeout=30.0)
                    yield Bar(
                        symbol=raw.get("symbol", ""),
                        exchange=Exchange.NSE,
                        interval=self._interval,
                        open=raw.get("o", 0),
                        high=raw.get("h", 0),
                        low=raw.get("l", 0),
                        close=raw.get("c", 0),
                        volume=raw.get("v", 0),
                        ts=raw.get("ts"),
                        source="angel_one",
                    )
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.exception(e)
                    break
        return _gen()
