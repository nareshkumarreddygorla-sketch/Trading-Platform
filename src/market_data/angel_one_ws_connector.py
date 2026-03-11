"""
Angel One SmartAPI WebSocket connector for live ticks.
Authenticates via SmartAPI (session token), opens WebSocket, subscribes to symbols,
parses tick data, pushes into TickToBarAggregator via MarketDataService consumption.
Reconnect with exponential backoff; heartbeat failure detection; optional on_feed_unhealthy.
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, List, Optional

from src.core.events import Exchange, Tick
from src.market_data.timestamp import normalize_ts

logger = logging.getLogger(__name__)

# Default SmartAPI smart-stream endpoint
DEFAULT_WS_URL = "wss://smartapis.angelone.in/smart-stream"
# Heartbeat: if no message for this many seconds, consider unhealthy
HEARTBEAT_TIMEOUT_SECONDS = 60

# Reconnect constants
_BACKOFF_SCHEDULE = [5, 10, 30, 60]  # seconds – cycles at 60 after exhausting list
_MAX_RECONNECT_ATTEMPTS = 20
_PING_INTERVAL_SECONDS = 30
_PONG_TIMEOUT_SECONDS = 60


def _parse_smartapi_tick(
    payload: dict,
    exchange: Exchange,
    token_to_symbol: Optional[dict] = None,
) -> Optional[Tick]:
    """Parse SmartAPI tick payload to Tick. Returns None if not a tick message.

    If *token_to_symbol* is provided, numeric-only symbol values are
    reverse-resolved to human-readable trading symbols.
    """
    # SmartAPI LTP/quote payloads: ltp, volume, symbol/tradingsymbol, etc.
    ltp = payload.get("ltp") or payload.get("last_price")
    if ltp is None:
        return None
    raw = (payload.get("tradingsymbol") or payload.get("symbol") or payload.get("tk") or "")
    symbol = (raw.replace("-EQ", "").strip() or (payload.get("symbol") or "").replace("-EQ", "").strip()) or raw

    # Reverse-resolve numeric token to symbol name
    if token_to_symbol and symbol.isdigit():
        resolved = token_to_symbol.get(symbol)
        if resolved:
            symbol = resolved

    vol = payload.get("volume") or payload.get("last_quantity") or 0
    ts_val = payload.get("last_trade_time") or payload.get("ts") or datetime.now(timezone.utc)
    return Tick(
        symbol=symbol,
        exchange=exchange,
        price=float(ltp),
        size=float(vol),
        ts=normalize_ts(ts_val),
    )


def _backoff_delay(attempt: int) -> float:
    """Return the backoff delay in seconds for the given attempt (0-indexed)."""
    idx = min(attempt, len(_BACKOFF_SCHEDULE) - 1)
    return float(_BACKOFF_SCHEDULE[idx])


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

    Automatic reconnection with exponential backoff (5s, 10s, 30s, 60s max).
    Heartbeat ping every 30s; connection considered dead after 60s of silence.
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
        symbol_token_map=None,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.token = token
        self.exchange_str = exchange
        self._exchange = Exchange(exchange) if exchange in [e.value for e in Exchange] else Exchange.NSE
        self.feed_url = feed_url
        self.on_feed_unhealthy = on_feed_unhealthy
        self.heartbeat_timeout = heartbeat_timeout_seconds
        self._symbol_token_map = symbol_token_map
        self._tick_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._ws: Any = None
        self._recv_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._connected = False
        self._last_tick_ts: Optional[datetime] = None
        self._last_message_ts: Optional[datetime] = None
        self._symbols: List[str] = []
        self._closed = False
        # Track whether stop() was called intentionally vs unexpected disconnect
        self._intentional_stop = False
        # Reverse map: numeric token string -> symbol name (built during subscribe)
        self._token_to_symbol: dict = {}

        # --- NoOp fallback tracking ---
        self._using_noop_ws: bool = False

        # --- Reconnect metrics ---
        self._reconnect_count: int = 0
        self._last_reconnect_at: float = 0.0
        self._total_disconnect_seconds: float = 0.0
        self._disconnect_started_at: Optional[float] = None

    # ------------------------------------------------------------------
    # Connection state helpers
    # ------------------------------------------------------------------

    def _set_connected(self, value: bool) -> None:
        prev = self._connected
        self._connected = value
        if not value and prev:
            # Transition from connected -> disconnected: start tracking downtime
            self._disconnect_started_at = time.monotonic()
        if value and not prev:
            # Transition from disconnected -> connected: accumulate downtime
            started = self._disconnect_started_at
            if started is not None:
                self._total_disconnect_seconds += time.monotonic() - started
                self._disconnect_started_at = None
        if not value and self.on_feed_unhealthy:
            try:
                self.on_feed_unhealthy()
            except Exception as e:
                logger.debug("on_feed_unhealthy callback error: %s", e)

    # ------------------------------------------------------------------
    # Public connect / disconnect
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Authenticate and open WebSocket connection."""
        if self._closed:
            raise RuntimeError("Connector is closed")
        self._intentional_stop = False
        self._last_message_ts = datetime.now(timezone.utc)
        try:
            import websockets  # noqa: F401
        except ImportError:
            logger.warning("websockets not installed; Angel One WS connector will use fallback")
        self._ws = await self._connect_ws()
        self._set_connected(True)
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
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
            logger.critical(
                "websockets package not installed — Angel One WS connector using "
                "no-op stub. No live tick data will be received. "
                "Install with: pip install websockets"
            )
            self._using_noop_ws = True
            class NoOpWs:
                async def recv(self):
                    logger.debug("NoOpWs.recv() sleeping (no websockets installed)")
                    await asyncio.sleep(3600)
                    return None
                async def send(self, data):
                    pass
                async def close(self):
                    pass
                async def ping(self):
                    pass
            return NoOpWs()

    async def disconnect(self) -> None:
        """Close WebSocket and stop recv loop. Alias kept for backward compat."""
        await self.stop()

    async def stop(self) -> None:
        """Intentionally stop the connector — no reconnection will be attempted."""
        self._intentional_stop = True
        self._closed = True
        self._set_connected(False)

        # Cancel reconnect loop first so it cannot restart anything
        reconnect_task = self._reconnect_task
        if reconnect_task is not None and not reconnect_task.done():
            reconnect_task.cancel()
            try:
                await reconnect_task
            except asyncio.CancelledError:
                pass
        self._reconnect_task = None

        # Cancel heartbeat
        heartbeat_task = self._heartbeat_task
        if heartbeat_task is not None and not heartbeat_task.done():
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        self._heartbeat_task = None

        # Cancel recv loop
        recv_task = self._recv_task
        if recv_task is not None and not recv_task.done():
            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass
        self._recv_task = None

        # Close underlying WebSocket
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug("WebSocket close: %s", e)
            self._ws = None

        # Finalize disconnect-time accounting
        if self._disconnect_started_at is not None:
            self._total_disconnect_seconds += time.monotonic() - self._disconnect_started_at
            self._disconnect_started_at = None

        logger.info("Angel One WebSocket disconnected (intentional stop)")

    # ------------------------------------------------------------------
    # Heartbeat / ping loop
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Send a ping every 30s. If no data received for 60s, force reconnect."""
        try:
            while not self._closed and not self._intentional_stop:
                await asyncio.sleep(_PING_INTERVAL_SECONDS)
                if self._closed or self._intentional_stop:
                    break

                # Send application-level ping
                if self._ws and self._connected:
                    try:
                        ping_frame = json.dumps({"task": "hb", "channel": ""})
                        await self._ws.send(ping_frame)
                    except Exception as e:
                        logger.debug("Heartbeat ping send failed: %s", e)

                # Check staleness
                if self._last_message_ts is not None:
                    silence = (datetime.now(timezone.utc) - self._last_message_ts).total_seconds()
                    if silence >= _PONG_TIMEOUT_SECONDS:
                        logger.warning(
                            "No data received for %.0fs — treating connection as dead",
                            silence,
                        )
                        self._set_connected(False)
                        self._trigger_reconnect()
                        break  # exit heartbeat; reconnect loop will start a new one
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Receive loop
    # ------------------------------------------------------------------

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
                    ticks = self._payload_to_tick(data)
                    for tick in ticks:
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
                    return  # exit without triggering reconnect
                except Exception as e:
                    logger.exception("Angel One WebSocket recv error: %s", e)
                    self._set_connected(False)
                    break
        finally:
            self._set_connected(False)
            # If not intentionally stopped, trigger reconnect
            if not self._intentional_stop and not self._closed:
                self._trigger_reconnect()

    def _payload_to_tick(self, data) -> List[Tick]:
        """Convert SmartAPI message to list of Ticks. Handles LTP/quote and nested payloads."""
        if isinstance(data, list):
            ticks: List[Tick] = []
            for item in data:
                ticks.extend(self._payload_to_tick(item))
            return ticks
        if not isinstance(data, dict):
            return []
        # Nested under key (e.g. feed response)
        reverse_map = self._token_to_symbol
        for key in ("data", "lp", "LTP", "ltp"):
            if key in data and isinstance(data[key], dict):
                t = _parse_smartapi_tick(data[key], self._exchange, reverse_map)
                if t:
                    return [t]
        t = _parse_smartapi_tick(data, self._exchange, reverse_map)
        return [t] if t else []

    # ------------------------------------------------------------------
    # Automatic reconnection with exponential backoff
    # ------------------------------------------------------------------

    def _trigger_reconnect(self) -> None:
        """Schedule the reconnect loop if not already running."""
        if self._intentional_stop or self._closed:
            return
        reconnect_task = self._reconnect_task
        if reconnect_task is not None and not reconnect_task.done():
            return  # already running
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Try to reconnect with exponential backoff up to _MAX_RECONNECT_ATTEMPTS."""
        attempt = 0
        while attempt < _MAX_RECONNECT_ATTEMPTS and not self._intentional_stop and not self._closed:
            delay = _backoff_delay(attempt)
            logger.info(
                "Reconnect attempt %d/%d in %.0fs …",
                attempt + 1,
                _MAX_RECONNECT_ATTEMPTS,
                delay,
            )
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                logger.debug("Reconnect loop cancelled during backoff sleep")
                return

            if self._intentional_stop or self._closed:
                return

            # Tear down old WebSocket
            if self._ws:
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = None

            try:
                self._ws = await self._connect_ws()
                self._last_message_ts = datetime.now(timezone.utc)

                # Update reconnect metrics
                self._reconnect_count += 1
                self._last_reconnect_at = time.monotonic()

                logger.info(
                    "Reconnected to Angel One WebSocket (attempt %d, total reconnects: %d)",
                    attempt + 1,
                    self._reconnect_count,
                )

                # Re-subscribe to previously subscribed symbols
                if self._symbols:
                    try:
                        await self.subscribe_ticks(self._symbols)
                        logger.info(
                            "Re-subscribed to %d symbols after reconnect",
                            len(self._symbols),
                        )
                    except Exception as e:
                        logger.warning("Re-subscribe failed after reconnect: %s", e)

                # Mark connected (accumulates downtime)
                self._set_connected(True)

                # Restart recv + heartbeat loops
                self._recv_task = asyncio.create_task(self._recv_loop())
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                return  # success — exit reconnect loop

            except asyncio.CancelledError:
                logger.debug("Reconnect loop cancelled during connect attempt")
                return
            except Exception as e:
                logger.warning("Reconnect attempt %d failed: %s", attempt + 1, e)
                attempt += 1

        # Exhausted all attempts
        if not self._intentional_stop and not self._closed:
            logger.error(
                "Failed to reconnect after %d attempts — giving up. "
                "Total disconnect time: %.1fs",
                _MAX_RECONNECT_ATTEMPTS,
                self._total_disconnect_seconds,
            )
            self._set_connected(False)  # fires on_feed_unhealthy via _set_connected

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    async def subscribe_ticks(self, symbols: list) -> None:
        """Subscribe to tick stream for given symbols via SmartAPI WebSocket.

        SmartAPI requires numeric instrument tokens, not trading symbols.
        We resolve tokens via the HTTP client's search endpoint, falling back
        to trading-symbol format if lookup fails.
        """
        self._symbols = list(symbols)
        if not self._ws:
            return
        # Resolve symbol -> numeric token via instrument master or search API
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

        Uses the injected SymbolTokenMap (full ~70K instrument master) as the
        primary lookup.  Falls back to a small hardcoded dict of well-known
        NSE tokens only when the SymbolTokenMap is unavailable or has no entry.
        """
        # Hardcoded fallback for top NSE stocks — used ONLY when SymbolTokenMap
        # cannot resolve a symbol (e.g. map not yet loaded or symbol missing).
        _FALLBACK_TOKENS = {
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
            "TATACONSUM": "3432", "M&M": "2031",
            "NIFTY50": "26000", "BANKNIFTY": "26009",
        }

        stm = self._symbol_token_map
        use_stm = stm is not None and stm.is_loaded
        if not use_stm:
            logger.warning(
                "SymbolTokenMap not available/loaded — falling back to "
                "hardcoded KNOWN_TOKENS (%d entries only)", len(_FALLBACK_TOKENS),
            )

        tokens = []
        self._token_to_symbol = {}  # rebuild reverse map on each subscribe

        for sym in symbols:
            clean = sym.replace(".NS", "").replace("-EQ", "").upper()
            resolved_token = None

            # Primary: SymbolTokenMap (full instrument master)
            if use_stm:
                resolved_token = stm.get_token(clean, self.exchange_str)

            # Secondary fallback: hardcoded dict
            if resolved_token is None and clean in _FALLBACK_TOKENS:
                resolved_token = _FALLBACK_TOKENS[clean]
                if use_stm:
                    logger.warning(
                        "SymbolTokenMap miss for %s — using hardcoded fallback token %s",
                        clean, resolved_token,
                    )

            if resolved_token is not None:
                tokens.append(resolved_token)
                self._token_to_symbol[resolved_token] = clean
            else:
                logger.warning("No token mapping for %s — skipping subscription", clean)

        return tokens

    def subscribe(self, symbols: List[str]) -> None:
        """Sync alias for subscribe_ticks (for API compatibility)."""
        self._symbols = list(symbols)

    # ------------------------------------------------------------------
    # Health & metrics
    # ------------------------------------------------------------------

    def is_healthy(self) -> bool:
        """True if connected and received a message recently.
        Always returns False when using NoOpWs fallback (websockets not installed)."""
        if self._using_noop_ws:
            return False
        if not self._connected:
            return False
        if self._last_message_ts is None:
            return True
        age = (datetime.now(timezone.utc) - self._last_message_ts).total_seconds()
        return age < self.heartbeat_timeout

    def get_last_tick_ts(self) -> Optional[datetime]:
        """Last tick timestamp."""
        return self._last_tick_ts

    def get_reconnect_metrics(self) -> dict:
        """Return reconnection metrics for observability."""
        current_downtime = self._total_disconnect_seconds
        if self._disconnect_started_at is not None:
            current_downtime += time.monotonic() - self._disconnect_started_at
        return {
            "reconnect_count": self._reconnect_count,
            "last_reconnect_at": self._last_reconnect_at,
            "total_disconnect_seconds": round(current_downtime, 2),
            "is_connected": self._connected,
            "is_healthy": self.is_healthy(),
        }

    # ------------------------------------------------------------------
    # Tick streaming
    # ------------------------------------------------------------------

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
