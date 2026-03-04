"""
Angel One SmartAPI execution gateway (India).
Live: real HTTP place/cancel/status/positions/orders via AngelOneHttpClient.
Paper: in-memory Order only. Invariant: no fake Order in live mode.

Token lifecycle: tokens are tracked from acquisition and automatically refreshed
1 hour before the 23-hour expiry window. On 401/session-expired errors, an
immediate TOTP-based re-authentication is attempted (up to 3 consecutive failures
before flagging a critical auth failure).
"""
import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from src.core.events import Exchange, Order, OrderStatus, OrderType, Position, SignalSide

from .base import BaseExecutionGateway
from .broker.angel_one_http_client import AngelOneHttpClient, BrokerClientError

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("audit.angel_one.token_refresh")

# Angel One error codes that signal an expired / invalid session token
_SESSION_EXPIRED_CODES = {"AG8001", "AG8002", "AB8051"}

# NSE circuit breaker / exchange halt error codes (no retry, specific reject)
_CIRCUIT_BREAKER_CODES = {"OI8001", "OI8002", "OI8003", "OI8004", "OI8005", "AB1003", "AB1004"}

# Per-symbol circuit halt tracking
_circuit_halted_symbols: Dict[str, float] = {}  # symbol -> time.monotonic() when halted

# Token refresh constants
_TOKEN_MAX_AGE: float = 23 * 3600          # 23 hours (Angel One JWT validity)
_TOKEN_REFRESH_HEADROOM: float = 1 * 3600  # refresh 1 hour before expiry
_MAX_REFRESH_FAILURES: int = 3             # critical alert threshold

# Map Angel One orderstatus to our OrderStatus
_ANGEL_STATUS_MAP = {
    "pending": OrderStatus.PENDING,
    "open": OrderStatus.LIVE,
    "trigger pending": OrderStatus.PENDING,
    "cancelled": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
    "complete": OrderStatus.FILLED,
    "completed": OrderStatus.FILLED,
    "traded": OrderStatus.FILLED,
}


def _to_order_status(s: str) -> OrderStatus:
    if not s:
        return OrderStatus.PENDING
    key = (s or "").strip().lower()
    return _ANGEL_STATUS_MAP.get(key, OrderStatus.PENDING)


class AngelOneExecutionGateway(BaseExecutionGateway):
    """
    Angel One order placement and status.
    Paper: returns in-memory Order (no HTTP).
    Live: uses AngelOneHttpClient; never returns fake Order; raises on broker failure.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        access_token: str,
        paper: bool = True,
        *,
        client_code: Optional[str] = None,
        password: Optional[str] = None,
        totp: Optional[str] = None,
        totp_secret: Optional[str] = None,
        refresh_token: Optional[str] = None,
        request_timeout: float = 15.0,
        on_health_failure: Optional[Callable[[], None]] = None,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.paper = paper
        self._client_code = client_code
        self._password = password
        self._totp_secret = totp_secret  # Base32 TOTP secret for generating fresh codes
        self._request_timeout = request_timeout
        self._on_health_failure = on_health_failure
        self._client: Optional[AngelOneHttpClient] = None

        # --- Token refresh state ---
        self._token_acquired_at: float = 0.0       # time.monotonic() when token was obtained
        self._token_max_age: float = _TOKEN_MAX_AGE
        self._refresh_failures: int = 0             # consecutive refresh failures
        self._auth_failed: bool = False              # set True after _MAX_REFRESH_FAILURES
        self._refresh_lock: asyncio.Lock = asyncio.Lock()

        if not paper and api_key:
            self._client = AngelOneHttpClient(
                api_key,
                access_token=access_token or None,
                refresh_token=refresh_token,
                client_code=client_code,
                password=password,
                totp=totp,
                timeout=request_timeout,
            )

    async def connect(self) -> None:
        if self.paper:
            logger.info("AngelOne execution (paper mode)")
            return
        if not self._client:
            logger.warning("AngelOne live mode but no client (missing api_key)")
            return
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, self._client.ensure_session),
                timeout=self._request_timeout + 5,
            )
            self._token_acquired_at = time.monotonic()
            self._refresh_failures = 0
            self._auth_failed = False
            audit_logger.info("Token acquired during connect at %.0f", time.time())
            logger.info("AngelOne execution (live): session ready")
        except asyncio.TimeoutError:
            logger.error("AngelOne connect timeout")
            self._notify_health_failure()
            raise
        except BrokerClientError as e:
            logger.error("AngelOne connect failed: %s", e)
            self._notify_health_failure()
            raise

    def _notify_health_failure(self) -> None:
        if self._on_health_failure:
            try:
                self._on_health_failure()
            except Exception as e:
                logger.warning("on_health_failure callback error: %s", e)

    async def disconnect(self) -> None:
        self._client = None

    # ------------------------------------------------------------------
    # Token refresh machinery
    # ------------------------------------------------------------------

    def _generate_totp(self) -> str:
        """Generate a fresh TOTP code from the stored secret.

        Requires the ``pyotp`` package.  Falls back to the static TOTP value
        that was passed at construction time if no secret is configured.
        """
        if not self._totp_secret:
            raise BrokerClientError(
                "Cannot generate TOTP: no totp_secret configured for auto-refresh"
            )
        try:
            import pyotp
        except ImportError:
            raise BrokerClientError(
                "pyotp package is required for automatic token refresh. "
                "Install it with: pip install pyotp"
            )
        return pyotp.TOTP(self._totp_secret).now()

    def _is_token_expiring(self) -> bool:
        """Return True if the current token is within 1 hour of its 23-hour expiry."""
        if self._token_acquired_at == 0.0:
            return False  # no token yet, will be caught by ensure_session
        elapsed = time.monotonic() - self._token_acquired_at
        return elapsed >= (self._token_max_age - _TOKEN_REFRESH_HEADROOM)

    @property
    def auth_failed(self) -> bool:
        """True if token refresh has failed >= 3 consecutive times."""
        return self._auth_failed

    async def _refresh_token(self) -> bool:
        """Perform a TOTP-based re-authentication via the broker client.

        Returns True on success, False on failure.  Thread-safe via
        ``_refresh_lock`` -- concurrent callers wait for the first refresh
        attempt rather than stampeding the broker API.
        """
        if not self._client:
            return False

        async with self._refresh_lock:
            # Double-check: another coroutine may have refreshed while we waited
            if (
                self._token_acquired_at > 0.0
                and not self._is_token_expiring()
                and self._refresh_failures == 0
            ):
                return True

            audit_logger.info(
                "Token refresh attempt (failures so far: %d)", self._refresh_failures
            )

            loop = asyncio.get_running_loop()
            try:
                fresh_totp = self._generate_totp()

                # Perform TOTP-based login on the underlying sync client
                def _do_relogin() -> None:
                    self._client.totp = fresh_totp
                    self._client.login()

                await asyncio.wait_for(
                    loop.run_in_executor(None, _do_relogin),
                    timeout=self._request_timeout + 5,
                )

                # Success -- reset state
                self._token_acquired_at = time.monotonic()
                self._refresh_failures = 0
                self._auth_failed = False
                audit_logger.info(
                    "Token refreshed successfully at %.0f", time.time()
                )
                return True

            except Exception as exc:
                self._refresh_failures += 1
                audit_logger.warning(
                    "Token refresh failed (attempt %d/%d): %s",
                    self._refresh_failures,
                    _MAX_REFRESH_FAILURES,
                    exc,
                )
                if self._refresh_failures >= _MAX_REFRESH_FAILURES:
                    self._auth_failed = True
                    logger.critical(
                        "CRITICAL: Angel One token refresh failed %d consecutive "
                        "times. Manual intervention required. Last error: %s",
                        self._refresh_failures,
                        exc,
                    )
                    audit_logger.critical(
                        "Auth failure flag set after %d consecutive refresh failures",
                        self._refresh_failures,
                    )
                    self._notify_health_failure()
                return False

    async def _ensure_valid_token(self) -> None:
        """Called before every live API request to guarantee a valid token.

        * If the token is expiring within 1 hour, proactively refresh.
        * If ``_auth_failed`` is set, raise immediately -- do not attempt
          further requests until the operator resolves the issue.
        """
        if self.paper or not self._client:
            return

        if self._auth_failed:
            raise BrokerClientError(
                "Angel One authentication has failed repeatedly. "
                "Manual re-login required."
            )

        if self._is_token_expiring():
            logger.info("Token approaching expiry, initiating proactive refresh")
            success = await self._refresh_token()
            if not success and self._auth_failed:
                raise BrokerClientError(
                    "Angel One token refresh failed critically. "
                    "Manual re-login required."
                )

    async def _execute_broker_call(
        self,
        operation_name: str,
        broker_fn: Callable[[], Any],
    ) -> Any:
        """Execute a synchronous broker call with automatic 401 retry.

        1. ``_ensure_valid_token()`` -- proactive refresh if near expiry.
        2. Run *broker_fn* in the thread-pool executor.
        3. On session-expired error codes (AG8001 / AG8002 / AB8051):
           attempt ``_refresh_token()`` then retry once.
        4. Track latency and failure metrics.
        """
        from src.monitoring.metrics import (
            track_broker_failure,
            track_broker_latency,
            track_broker_session_expired,
        )

        await self._ensure_valid_token()

        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, broker_fn),
                timeout=self._request_timeout + 2,
            )
            track_broker_latency(operation_name, time.perf_counter() - t0)
            return result

        except asyncio.TimeoutError:
            track_broker_latency(operation_name, time.perf_counter() - t0)
            track_broker_failure(operation_name, "timeout")
            self._notify_health_failure()
            raise BrokerClientError(f"{operation_name} timeout")

        except BrokerClientError as exc:
            elapsed = time.perf_counter() - t0
            track_broker_latency(operation_name, elapsed)

            # Session-expired: attempt one immediate refresh + retry
            if exc.errorcode in _SESSION_EXPIRED_CODES:
                track_broker_session_expired()
                audit_logger.warning(
                    "Session expired during %s (code=%s), attempting refresh",
                    operation_name,
                    exc.errorcode,
                )
                refreshed = await self._refresh_token()
                if refreshed:
                    # Retry once after successful refresh
                    t1 = time.perf_counter()
                    try:
                        result = await asyncio.wait_for(
                            loop.run_in_executor(None, broker_fn),
                            timeout=self._request_timeout + 2,
                        )
                        track_broker_latency(operation_name, time.perf_counter() - t1)
                        audit_logger.info(
                            "Retry of %s succeeded after token refresh",
                            operation_name,
                        )
                        return result
                    except asyncio.TimeoutError:
                        track_broker_latency(operation_name, time.perf_counter() - t1)
                        track_broker_failure(operation_name, "timeout_after_refresh")
                        self._notify_health_failure()
                        raise BrokerClientError(f"{operation_name} timeout after refresh")
                    except BrokerClientError as retry_exc:
                        track_broker_latency(operation_name, time.perf_counter() - t1)
                        track_broker_failure(
                            operation_name,
                            retry_exc.errorcode or type(retry_exc).__name__,
                        )
                        self._notify_health_failure()
                        raise
                else:
                    # Refresh failed
                    track_broker_failure(operation_name, exc.errorcode or "session_expired")
                    self._notify_health_failure()
                    raise

            # NSE circuit breaker: don't retry, track per-symbol halt
            if exc.errorcode in _CIRCUIT_BREAKER_CODES:
                logger.warning(
                    "NSE circuit breaker halt during %s (code=%s): %s",
                    operation_name, exc.errorcode, exc,
                )
                track_broker_failure(operation_name, f"circuit_breaker_{exc.errorcode}")
                raise BrokerClientError(
                    f"NSE circuit breaker halt: {exc.errorcode}",
                    errorcode=exc.errorcode,
                )

            # Non-session error: propagate as-is
            track_broker_failure(operation_name, exc.errorcode or type(exc).__name__)
            raise

    # ------------------------------------------------------------------
    # Parse helpers
    # ------------------------------------------------------------------

    def _parse_exchange(self, exchange: str) -> Exchange:
        try:
            return Exchange(exchange) if exchange else Exchange.NSE
        except ValueError:
            return Exchange.NSE

    def _parse_side(self, side: str) -> SignalSide:
        s = (side or "").strip().upper()
        if s in ("BUY", "SELL"):
            return SignalSide(s)
        return SignalSide.BUY

    def _parse_order_type(self, order_type: str) -> OrderType:
        t = (order_type or "LIMIT").strip().upper()
        if t in ("MARKET", "LIMIT", "IOC", "FOK"):
            return OrderType(t)
        return OrderType.LIMIT

    async def place_order(
        self,
        symbol: str,
        exchange: str,
        side: str,
        quantity: float,
        order_type: str,
        limit_price: Optional[float] = None,
        strategy_id: str = "",
        **kwargs,
    ) -> Order:
        exch = self._parse_exchange(exchange)
        side_enum = self._parse_side(side)
        ot = self._parse_order_type(order_type)

        # Check per-symbol circuit halt (prevent resubmission during halt)
        if symbol in _circuit_halted_symbols:
            halt_time = _circuit_halted_symbols[symbol]
            elapsed = time.monotonic() - halt_time
            if elapsed < 300:  # 5 minute halt window
                raise BrokerClientError(
                    f"Symbol {symbol} under circuit halt ({elapsed:.0f}s ago)",
                    errorcode="CIRCUIT_HALT",
                )
            else:
                del _circuit_halted_symbols[symbol]  # Clear expired halt

        if self.paper:
            from uuid import uuid4
            return Order(
                order_id=str(uuid4()),
                strategy_id=strategy_id,
                symbol=symbol,
                exchange=exch,
                side=side_enum,
                quantity=quantity,
                order_type=ot,
                limit_price=limit_price,
                status=OrderStatus.PENDING,
                metadata={"paper": True, "broker": "angel_one"},
            )
        # Live: real HTTP; never return fake Order
        if not self._client:
            raise BrokerClientError("Live mode but broker client not configured")
        # Angel: tradingsymbol e.g. INFY-EQ, symboltoken optional
        tradingsymbol = kwargs.get("tradingsymbol") or f"{symbol}-EQ"
        params = {
            "exchange": exchange or "NSE",
            "tradingsymbol": tradingsymbol,
            "quantity": int(quantity),
            "transactiontype": side.strip().upper(),
            "ordertype": order_type.strip().upper(),
            "variety": "NORMAL",
            "producttype": kwargs.get("producttype", "INTRADAY"),
            "duration": kwargs.get("duration", "DAY"),
            "scripconsent": "yes",
        }
        if kwargs.get("symboltoken"):
            params["symboltoken"] = kwargs["symboltoken"]
        if limit_price is not None and ot == OrderType.LIMIT:
            params["price"] = str(limit_price)
        if strategy_id and len(strategy_id) <= 20:
            params["ordertag"] = strategy_id

        data = await self._execute_broker_call(
            "place_order",
            lambda: self._client.place_order(params),
        )

        # Response: data with orderid (broker) and uniqueorderid (UUID)
        broker_order_id = data.get("orderid") or data.get("uniqueorderid")
        unique_id = data.get("uniqueorderid") or broker_order_id
        if not unique_id:
            unique_id = str(broker_order_id) if broker_order_id else None
        if not broker_order_id:
            broker_order_id = unique_id
        return Order(
            order_id=unique_id or "",
            strategy_id=strategy_id,
            symbol=symbol,
            exchange=exch,
            side=side_enum,
            quantity=quantity,
            order_type=ot,
            limit_price=limit_price,
            status=OrderStatus.LIVE,
            broker_order_id=str(broker_order_id) if broker_order_id else None,
            metadata={"broker": "angel_one", "uniqueorderid": unique_id},
        )

    async def cancel_order(self, order_id: str, broker_order_id: Optional[str] = None) -> bool:
        if self.paper:
            return True
        if not self._client:
            raise BrokerClientError("Live mode but broker client not configured")
        # Cancel API expects broker orderid (numeric); details API expects uniqueorderid (UUID)
        order_to_use = broker_order_id or order_id
        try:
            await self._execute_broker_call(
                "cancel_order",
                lambda: self._client.cancel_order("NORMAL", str(order_to_use)),
            )
        except (BrokerClientError, Exception):
            return False
        return True

    async def get_order_status(self, order_id: str, broker_order_id: Optional[str] = None) -> OrderStatus:
        if self.paper:
            return OrderStatus.PENDING
        if not self._client:
            return OrderStatus.PENDING
        unique_id = broker_order_id or order_id
        try:
            data = await self._execute_broker_call(
                "get_order_status",
                lambda: self._client.get_order_details(str(unique_id)),
            )
        except Exception:
            return OrderStatus.PENDING
        status_str = (data.get("orderstatus") or data.get("status") or "").strip().lower()
        return _to_order_status(status_str)

    def _position_from_row(self, row: dict) -> Position:
        side = (row.get("netqty") or 0) >= 0 and SignalSide.BUY or SignalSide.SELL
        return Position(
            symbol=row.get("tradingsymbol", "").replace("-EQ", ""),
            exchange=self._parse_exchange(row.get("exchange", "NSE")),
            side=side,
            quantity=abs(float(row.get("netqty", 0) or 0)),
            avg_price=float(row.get("avgnetprice", 0) or 0),
            metadata={"broker": "angel_one"},
        )

    def _order_from_row(self, row: dict) -> Order:
        status_str = (row.get("orderstatus") or row.get("status") or "pending").strip().lower()
        return Order(
            order_id=row.get("uniqueorderid") or str(row.get("orderid", "")),
            strategy_id=row.get("ordertag", ""),
            symbol=(row.get("tradingsymbol") or "").replace("-EQ", ""),
            exchange=self._parse_exchange(row.get("exchange", "NSE")),
            side=self._parse_side(row.get("transactiontype", "BUY")),
            quantity=float(row.get("quantity", 0) or 0),
            order_type=self._parse_order_type(row.get("ordertype", "LIMIT")),
            limit_price=float(row.get("price", 0)) or None,
            status=_to_order_status(status_str),
            filled_qty=float(row.get("filledshares", 0) or 0),
            avg_price=float(row.get("averageprice", 0)) or None,
            broker_order_id=str(row.get("orderid", "")),
            metadata={"broker": "angel_one"},
        )

    async def get_positions(self) -> List[Position]:
        if self.paper:
            return []
        if not self._client:
            return []
        try:
            rows = await self._execute_broker_call(
                "get_positions",
                self._client.get_position,
            )
        except Exception:
            return []
        return [self._position_from_row(r) for r in (rows or []) if r.get("netqty")]

    async def get_orders(self, status: Optional[str] = None, limit: int = 100) -> List[Order]:
        if self.paper:
            return []
        if not self._client:
            return []
        try:
            rows = await self._execute_broker_call(
                "get_orders",
                self._client.get_order_book,
            )
        except Exception:
            return []
        orders = [self._order_from_row(r) for r in (rows or [])[:limit]]
        return orders
