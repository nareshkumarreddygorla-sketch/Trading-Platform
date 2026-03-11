"""
Angel One SmartAPI connector for Indian markets (NSE/BSE).
WebSocket for live ticks; REST for historical and order placement.
See: https://smartapi.angelone.in/
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta, timezone

from src.core.events import Bar, Exchange, Tick

from .base import BaseMarketDataConnector

logger = logging.getLogger(__name__)


_IST = timezone(timedelta(hours=5, minutes=30))


def _normalize_tick_ts(ts) -> datetime:
    """Normalize a raw timestamp value to a timezone-aware UTC datetime."""
    if ts is None:
        return datetime.now(UTC)
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=_IST).astimezone(UTC)
        return ts
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=UTC)
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=_IST).astimezone(UTC)
            return dt
        except (ValueError, TypeError):
            pass
    return datetime.now(UTC)


class AngelOneConnector(BaseMarketDataConnector):
    """
    Angel One SmartAPI: real-time and historical market data for NSE/BSE.
    Uses SmartAPI WebSocket V2 for live ticks; falls back gracefully if SDK not installed.
    """

    exchange = Exchange.NSE

    def __init__(
        self, api_key: str, api_secret: str, token: str, feed: str = "wss://smartapis.angelone.in/smart-stream"
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.token = token
        self.feed_url = feed
        self._ws = None
        self._ws_thread = None
        self._connected = False
        self._tick_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._bar_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._symbols: list[str] = []
        self._interval: str = "1m"
        self._loop: asyncio.AbstractEventLoop = None

    def _on_data(self, ws, message):
        """Callback for SmartAPI WebSocket V2 data events."""
        try:
            if isinstance(message, dict):
                tick_data = {
                    "symbol": message.get("tk", message.get("symbol", "")),
                    "ltp": message.get("ltp", 0),
                    "volume": message.get("v", message.get("volume", 0)),
                    "ts": message.get("exchange_timestamp", message.get("ts")),
                }
                if self._loop and not self._loop.is_closed():
                    asyncio.run_coroutine_threadsafe(self._tick_queue.put(tick_data), self._loop)
        except Exception as e:
            logger.warning("AngelOne tick parse error: %s", e)

    def _on_error(self, ws, error):
        """Callback for SmartAPI WebSocket errors."""
        logger.error("AngelOne WebSocket error: %s", error)

    def _on_close(self, ws, close_status_code=None, close_msg=None):
        """Callback for WebSocket close."""
        self._connected = False
        logger.info("AngelOne WebSocket closed (status=%s)", close_status_code)

    def _on_open(self, ws):
        """Callback for WebSocket connection open."""
        self._connected = True
        logger.info("AngelOne WebSocket connected successfully")

    async def connect(self) -> None:
        """Connect to Angel One SmartAPI WebSocket for real-time market data."""
        self._loop = asyncio.get_running_loop()
        try:
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2

            self._ws = SmartWebSocketV2(
                self.token,
                self.api_key,
                client_code=self.api_secret,
                feed_token=self.token,
            )
            self._ws.on_data = self._on_data
            self._ws.on_error = self._on_error
            self._ws.on_close = self._on_close
            self._ws.on_open = self._on_open

            # Connect in a background thread (SmartAPI SDK uses synchronous WebSocket)
            import threading

            self._ws_thread = threading.Thread(target=self._ws.connect, daemon=True, name="angel-one-ws")
            self._ws_thread.start()
            logger.info("AngelOne WebSocket connection initiated (background thread)")

            # Wait briefly for connection
            for _ in range(10):
                if self._connected:
                    break
                await asyncio.sleep(0.5)

            if not self._connected:
                logger.warning("AngelOne WebSocket: connection not confirmed after 5s, proceeding anyway")

        except ImportError:
            logger.warning(
                "SmartApi package not installed; AngelOne live feed unavailable. "
                "Install with: pip install smartapi-python. Using yfinance fallback."
            )
        except Exception as e:
            logger.error("AngelOne WebSocket connect failed: %s (will use fallback data source)", e)

    async def disconnect(self) -> None:
        """Disconnect WebSocket and clean up."""
        self._connected = False
        if self._ws:
            try:
                self._ws.close_connection()
            except Exception as e:
                logger.warning("AngelOne WebSocket close error: %s", e)
        self._ws = None

    async def subscribe_ticks(self, symbols: list[str]) -> None:
        """Subscribe to live tick data for given symbols.

        Resolves human-readable symbols to numeric Angel One tokens via the
        global SymbolTokenMap before passing them to the SmartAPI SDK.
        """
        self._symbols = list(symbols)
        if self._ws and self._connected:
            try:
                # Attempt to use the global SymbolTokenMap for proper token resolution
                stm = None
                try:
                    from src.market_data.symbol_token_map import get_symbol_token_map

                    stm = get_symbol_token_map()
                except Exception:
                    pass

                # SmartAPI V2 token list format: [[exchange_type, token]]
                # exchange_type: 1=NSE, 2=NFO, 3=BSE
                token_list = []
                for sym in symbols:
                    clean = sym.replace(".NS", "").replace("-EQ", "").upper()
                    numeric_token = None
                    if stm is not None and stm.is_loaded:
                        numeric_token = stm.get_token(clean, "NSE")
                    if numeric_token is not None:
                        token_list.append([1, numeric_token])
                    else:
                        # Fallback: pass raw symbol (will likely fail on SmartAPI)
                        logger.warning(
                            "No numeric token for %s — passing raw symbol to SmartAPI",
                            clean,
                        )
                        token_list.append([1, clean])

                self._ws.subscribe(str(uuid.uuid4())[:8], 1, token_list)  # mode 1 = LTP
                logger.info("AngelOne: subscribed to %d symbols for live ticks", len(symbols))
            except Exception as e:
                logger.warning("AngelOne subscribe_ticks failed: %s", e)
        else:
            logger.info("AngelOne: WebSocket not connected; tick subscription queued for %d symbols", len(symbols))

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
                        ts=_normalize_tick_ts(raw.get("ts")),
                    )
                except TimeoutError:
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
                except TimeoutError:
                    continue
                except Exception as e:
                    logger.exception(e)
                    break

        return _gen()
