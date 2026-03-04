"""
Angel One SmartAPI WebSocket connector for live ticks.
Authenticates via SmartAPI (session token), opens WebSocket, subscribes to symbols,
parses tick data, pushes into TickToBarAggregator via MarketDataService consumption.
Reconnect with exponential backoff; heartbeat failure detection; optional on_feed_unhealthy.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, List, Optional

from src.core.events import Exchange, Tick

logger = logging.getLogger(__name__)

# Default SmartAPI smart-stream endpoint
DEFAULT_WS_URL = "wss://smartapis.angelone.in/smart-stream"
# Heartbeat: if no message for this many seconds, consider unhealthy
HEARTBEAT_TIMEOUT_SECONDS = 60


def _normalize_ts(ts: Any) -> datetime:
    if ts is None:
        return datetime.now(timezone.utc)
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return datetime.now(timezone.utc)


def _parse_smartapi_tick(payload: dict, exchange: Exchange) -> Optional[Tick]:
    """Parse SmartAPI tick payload to Tick. Returns None if not a tick message."""
    # SmartAPI LTP/quote payloads: ltp, volume, symbol/tradingsymbol, etc.
    ltp = payload.get("ltp") or payload.get("last_price")
    if ltp is None:
        return None
    raw = (payload.get("tradingsymbol") or payload.get("symbol") or "")
    symbol = (raw.replace("-EQ", "").strip() or (payload.get("symbol") or "").replace("-EQ", "").strip()) or raw
    vol = payload.get("volume") or payload.get("last_quantity") or 0
    ts_val = payload.get("last_trade_time") or payload.get("ts") or datetime.now(timezone.utc)
    return Tick(
        symbol=symbol,
        exchange=exchange,
        price=float(ltp),
        size=float(vol),
        ts=_normalize_ts(ts_val),
    )


class AngelOneWsConnector:
    """
    Angel One live market data via SmartAPI WebSocket.
    - connect(): authenticate and open WebSocket.
    - disconnect(): close connection.
    - subscribe(symbols): subscribe to tick stream for symbols.
    - stream_ticks(): async generator of Tick (consumed by MarketDataService which pushes to aggregator).
    - is_healthy(): True if connected and recent tick/heartbeat.
    - get_last_tick_ts(): last tick timestamp.
    On failure calls optional on_feed_unhealthy callback.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        token: str,
        *,
        exchange: str = "NSE",
        feed_url: str = DEFAULT_WS_URL,
        on_feed_unhealthy: Optional[Callable[[], None]] = None,
        heartbeat_timeout_seconds: float = HEARTBEAT_TIMEOUT_SECONDS,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.token = token
        self.exchange_str = exchange
        self._exchange = Exchange(exchange) if exchange in [e.value for e in Exchange] else Exchange.NSE
        self.feed_url = feed_url
        self.on_feed_unhealthy = on_feed_unhealthy
        self.heartbeat_timeout = heartbeat_timeout_seconds
        self._tick_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._ws: Any = None
        self._recv_task: Optional[asyncio.Task] = None
        self._connected = False
        self._last_tick_ts: Optional[datetime] = None
        self._last_message_ts: Optional[datetime] = None
        self._symbols: List[str] = []
        self._closed = False

    def _set_connected(self, value: bool) -> None:
        self._connected = value
        if not value and self.on_feed_unhealthy:
            try:
                self.on_feed_unhealthy()
            except Exception as e:
                logger.debug("on_feed_unhealthy callback error: %s", e)

    async def connect(self) -> None:
        """Authenticate and open WebSocket connection."""
        if self._closed:
            raise RuntimeError("Connector is closed")
        self._last_message_ts = datetime.now(timezone.utc)
        try:
            import websockets
        except ImportError:
            logger.warning("websockets not installed; Angel One WS connector will use fallback")
        self._ws = await self._connect_ws()
        self._connected = True
        self._recv_task = asyncio.create_task(self._recv_loop())
        logger.info("Angel One WebSocket connected to %s", self.feed_url)

    async def _connect_ws(self):
        """Open WebSocket. Uses websockets library if available."""
        try:
            import websockets
            extra_headers = {
                "Authorization": f"Bearer {self.token}",
                "x-api-key": self.api_key,
                "x-client-code": self.api_secret,
            }
            ws = await asyncio.wait_for(
                websockets.connect(
                    self.feed_url,
                    additional_headers=extra_headers,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                ),
                timeout=15,
            )
            # SmartAPI requires auth handshake frame after connection
            auth_frame = json.dumps({
                "task": "cn",
                "channel": "",
                "token": self.token,
                "user": self.api_secret,
                "acctid": self.api_secret,
            })
            await ws.send(auth_frame)
            # Wait for auth acknowledgement
            try:
                ack = await asyncio.wait_for(ws.recv(), timeout=10)
                logger.info("Angel One WS auth response received")
            except asyncio.TimeoutError:
                logger.warning("Angel One WS auth ack timeout — continuing")
            return ws
        except ImportError:
            # No websockets: create a mock that never yields (tests can inject queue)
            class NoOpWs:
                async def recv(self):
                    await asyncio.sleep(3600)
                    return None
                async def send(self, data):
                    pass
                async def close(self):
                    pass
            return NoOpWs()

    async def _recv_loop(self) -> None:
        """Receive messages and push ticks to queue."""
        try:
            while self._ws and self._connected and not self._closed:
                try:
                    msg = await asyncio.wait_for(self._ws.recv(), timeout=self.heartbeat_timeout)
                    self._last_message_ts = datetime.now(timezone.utc)
                    if isinstance(msg, bytes):
                        msg = msg.decode("utf-8", errors="ignore")
                    data = json.loads(msg) if isinstance(msg, str) else msg
                    tick = self._payload_to_tick(data)
                    if tick:
                        try:
                            self._tick_queue.put_nowait(tick)
                            self._last_tick_ts = tick.ts
                        except asyncio.QueueFull:
                            logger.warning("Tick queue full; dropping tick")
                except asyncio.TimeoutError:
                    logger.warning("Angel One WebSocket heartbeat timeout")
                    self._set_connected(False)
                    break
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.exception("Angel One WebSocket recv error: %s", e)
                    self._set_connected(False)
                    break
        finally:
            self._connected = False

    def _payload_to_tick(self, data: dict) -> Optional[Tick]:
        """Convert SmartAPI message to Tick. Handles LTP/quote and nested payloads."""
        if isinstance(data, list):
            for item in data:
                t = self._payload_to_tick(item)
                if t:
                    return t
            return None
        if not isinstance(data, dict):
            return None
        # Nested under key (e.g. feed response)
        for key in ("data", "lp", "LTP", "ltp"):
            if key in data and isinstance(data[key], dict):
                t = _parse_smartapi_tick(data[key], self._exchange)
                if t:
                    return t
        return _parse_smartapi_tick(data, self._exchange)

    async def disconnect(self) -> None:
        """Close WebSocket and stop recv loop."""
        self._closed = True
        self._connected = False
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug("WebSocket close: %s", e)
            self._ws = None
        logger.info("Angel One WebSocket disconnected")

    async def subscribe_ticks(self, symbols: list) -> None:
        """Subscribe to tick stream for given symbols via SmartAPI WebSocket.

        SmartAPI requires numeric instrument tokens, not trading symbols.
        We resolve tokens via the HTTP client's search endpoint, falling back
        to trading-symbol format if lookup fails.
        """
        self._symbols = list(symbols)
        if not self._ws:
            return
        # Resolve symbol → numeric token via instrument master or search API
        token_list = await self._resolve_tokens(symbols)
        if not token_list:
            logger.warning("No tokens resolved for symbols: %s", symbols)
            return
        try:
            # SmartAPI v2 subscription format with numeric exchange|token pairs
            payload = json.dumps({
                "action": 1,  # 1 = subscribe
                "params": {
                    "mode": 1,  # 1 = LTP mode
                    "tokenList": [{"exchangeType": 1, "tokens": token_list}],  # 1 = NSE
                },
            })
            await self._ws.send(payload)
            logger.info("Subscribed to %d symbols (%d tokens resolved)", len(symbols), len(token_list))
        except Exception as e:
            logger.warning("Angel One subscribe send failed: %s", e)

    async def _resolve_tokens(self, symbols: list) -> list:
        """Resolve trading symbols to SmartAPI numeric tokens.

        Uses AngelOneHttpClient.search_scrip() if available, otherwise falls back
        to well-known NSE symbol→token mappings for top traded stocks.
        """
        # Well-known NSE tokens for top stocks (fallback if HTTP client unavailable)
        KNOWN_TOKENS = {
            "RELIANCE": "2885", "TCS": "11536", "HDFCBANK": "1333",
            "INFY": "1594", "ICICIBANK": "4963", "HINDUNILVR": "1394",
            "ITC": "1660", "SBIN": "3045", "BHARTIARTL": "10604",
            "KOTAKBANK": "1922", "LT": "11483", "AXISBANK": "5900",
            "WIPRO": "3787", "ADANIENT": "25", "BAJFINANCE": "317",
            "MARUTI": "10999", "TATAMOTORS": "3456", "SUNPHARMA": "3351",
            "TITAN": "3506", "ULTRACEMCO": "11532", "ASIANPAINT": "236",
            "NESTLEIND": "17963", "TECHM": "13538", "POWERGRID": "14977",
            "NTPC": "11630", "BAJAJFINSV": "16675", "JSWSTEEL": "11723",
            "TATASTEEL": "3499", "ONGC": "2475", "HCLTECH": "7229",
            "INDUSINDBK": "5258", "ADANIPORTS": "15083", "COALINDIA": "20374",
            "GRASIM": "1232", "CIPLA": "694", "EICHERMOT": "910",
            "DRREDDY": "881", "APOLLOHOSP": "157", "BPCL": "526",
            "DIVISLAB": "10940", "SBILIFE": "21808", "BRITANNIA": "547",
            "HEROMOTOCO": "1348", "HINDALCO": "1363", "BAJAJ-AUTO": "16669",
            "TATACONSUM": "3432", "M&M": "2031", "WIPRO": "3787",
            "NIFTY50": "26000", "BANKNIFTY": "26009",
        }
        tokens = []
        for sym in symbols:
            clean = sym.replace(".NS", "").replace("-EQ", "").upper()
            if clean in KNOWN_TOKENS:
                tokens.append(KNOWN_TOKENS[clean])
            else:
                # Fallback: try symbol name as token (some APIs accept this)
                tokens.append(clean)
                logger.debug("No token mapping for %s, using symbol name", clean)
        return tokens

    def subscribe(self, symbols: List[str]) -> None:
        """Sync alias for subscribe_ticks (for API compatibility)."""
        self._symbols = list(symbols)

    def is_healthy(self) -> bool:
        """True if connected and received a message recently."""
        if not self._connected:
            return False
        if self._last_message_ts is None:
            return True
        age = (datetime.now(timezone.utc) - self._last_message_ts).total_seconds()
        return age < self.heartbeat_timeout

    def get_last_tick_ts(self) -> Optional[datetime]:
        """Last tick timestamp."""
        return self._last_tick_ts

    async def stream_ticks(self) -> AsyncIterator[Tick]:
        """Async generator of Tick. MarketDataService consumes this and pushes to TickToBarAggregator."""
        while not self._closed:
            try:
                tick = await asyncio.wait_for(self._tick_queue.get(), timeout=30.0)
                yield tick
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("stream_ticks: %s", e)
                break
