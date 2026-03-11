"""
Zerodha Kite Connect execution gateway.

Live: uses the Kite Connect v3 REST API (via ``kiteconnect`` SDK) for order
      placement, cancellation, positions, holdings, and market quotes.
Paper: in-memory order simulation (no HTTP), identical pattern to Angel One.

Token lifecycle
---------------
Kite Connect access tokens expire daily (at ~06:00 IST next day).  The gateway
tracks the token acquisition time and proactively flags expiry.  Since Kite
does not support silent token refresh (the user must complete the OAuth redirect
flow each day), the gateway raises a clear error when the token has expired so
the caller can initiate re-authentication.

Circuit-breaker tracking
------------------------
Kite-specific error codes that indicate exchange-level halts or rate limits are
tracked per-symbol with a configurable cooldown window, preventing order
stampedes during volatile conditions.

References
----------
* Kite Connect API docs : https://kite.trade/docs/connect/v3/
* kiteconnect PyPI pkg  : https://pypi.org/project/kiteconnect/
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from src.core.events import (
    Exchange,
    Order,
    OrderStatus,
    OrderType,
    Position,
    SignalSide,
)

from .broker_interface import BrokerInterface, register_broker

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("audit.zerodha.token")


# ---------------------------------------------------------------------------
# Lazy SDK import
# ---------------------------------------------------------------------------

_kiteconnect_available: Optional[bool] = None


def _import_kite():
    """Lazily import kiteconnect SDK.  Returns the KiteConnect class or None."""
    global _kiteconnect_available
    if _kiteconnect_available is False:
        return None
    try:
        from kiteconnect import KiteConnect
        _kiteconnect_available = True
        return KiteConnect
    except ImportError:
        _kiteconnect_available = False
        logger.warning(
            "kiteconnect package not installed. Live trading unavailable. "
            "Install with: pip install kiteconnect"
        )
        return None


# ---------------------------------------------------------------------------
# Kite-specific error handling
# ---------------------------------------------------------------------------


class KiteGatewayError(Exception):
    """Raised for Kite Connect API errors with optional error code."""

    def __init__(self, message: str, *, errorcode: Optional[str] = None):
        super().__init__(message)
        self.errorcode = errorcode


# Kite Connect exception types (strings, matched by class name)
# These are the exception classes defined in the kiteconnect SDK:
#   TokenException       - expired or invalid access token
#   PermissionException  - insufficient permissions
#   OrderException       - order placement/modification failure
#   InputException       - invalid input parameters
#   DataException        - data fetch failure
#   NetworkException     - connectivity issues
#   GeneralException     - catch-all

# Error codes / exception types that signal token expiry
_TOKEN_EXPIRED_EXCEPTIONS = {"TokenException"}

# Error codes / exception types for exchange circuit-breaker halts
_CIRCUIT_BREAKER_MESSAGES = {
    "circuit breaker",
    "trading halted",
    "upper circuit",
    "lower circuit",
    "market closed",
}

# Per-symbol circuit halt tracking: symbol -> monotonic timestamp
_circuit_halted_symbols: Dict[str, float] = {}
_CIRCUIT_HALT_WINDOW: float = 300.0  # 5 minutes

# Token constants (Kite tokens expire daily at ~06:00 IST next day)
_TOKEN_MAX_AGE: float = 18 * 3600          # conservative 18h (tokens last ~24h)
_TOKEN_REFRESH_HEADROOM: float = 1 * 3600  # warn 1h before assumed expiry
_MAX_CONSECUTIVE_FAILURES: int = 3         # critical alert threshold

# Retry configuration
_MAX_RETRIES: int = 2
_RETRY_BACKOFF_BASE: float = 0.5  # seconds


# ---------------------------------------------------------------------------
# Internal type mappings  (our generic <-> Kite-specific)
# ---------------------------------------------------------------------------

# Order type mapping: internal -> Kite
_ORDER_TYPE_TO_KITE: Dict[str, str] = {
    "MARKET": "MARKET",
    "LIMIT": "LIMIT",
    "SL": "SL",          # stop-loss limit
    "SL-M": "SL-M",      # stop-loss market
}

# Side mapping: internal -> Kite transaction type
_SIDE_TO_KITE: Dict[str, str] = {
    "BUY": "BUY",
    "SELL": "SELL",
}

# Product type mapping: internal -> Kite product
_PRODUCT_TYPE_TO_KITE: Dict[str, str] = {
    "INTRADAY": "MIS",   # Margin Intraday Square-off
    "MIS": "MIS",
    "CNC": "CNC",        # Cash and Carry (delivery)
    "NRML": "NRML",      # Normal (F&O overnight)
    "DELIVERY": "CNC",
}

# Reverse maps: Kite -> internal
_KITE_PRODUCT_MAP: Dict[str, str] = {"MIS": "INTRADAY", "CNC": "CNC", "NRML": "NRML"}

# Kite order status -> our canonical OrderStatus
_KITE_STATUS_MAP: Dict[str, OrderStatus] = {
    "OPEN": OrderStatus.LIVE,
    "COMPLETE": OrderStatus.FILLED,
    "CANCELLED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "TRIGGER PENDING": OrderStatus.PENDING,
    "OPEN PENDING": OrderStatus.PENDING,
    "VALIDATION PENDING": OrderStatus.PENDING,
    "PUT ORDER REQ RECEIVED": OrderStatus.PENDING,
    "MODIFY VALIDATION PENDING": OrderStatus.LIVE,
    "MODIFY ORDER REQ RECEIVED": OrderStatus.LIVE,
    "CANCEL PENDING": OrderStatus.LIVE,
}

# Exchange mapping: our Exchange names -> Kite exchange strings
_EXCHANGE_TO_KITE: Dict[str, str] = {
    "NSE": "NSE",
    "BSE": "BSE",
    "NFO": "NFO",        # NSE F&O
    "BFO": "BFO",        # BSE F&O
    "CDS": "CDS",        # Currency
    "BCD": "BCD",
    "MCX": "MCX",        # Commodity
}


def _to_order_status(kite_status: str) -> OrderStatus:
    """Map a Kite status string to our canonical OrderStatus."""
    key = (kite_status or "").strip().upper()
    return _KITE_STATUS_MAP.get(key, OrderStatus.PENDING)


def _parse_exchange(exchange: str) -> Exchange:
    """Parse exchange string to Exchange enum, defaulting to NSE."""
    try:
        return Exchange(exchange) if exchange else Exchange.NSE
    except ValueError:
        return Exchange.NSE


def _parse_side(side: str) -> SignalSide:
    """Parse side string to SignalSide enum."""
    s = (side or "").strip().upper()
    if s in ("BUY", "SELL"):
        return SignalSide(s)
    return SignalSide.BUY


def _parse_order_type(order_type: str) -> OrderType:
    """Parse order type string to OrderType enum."""
    t = (order_type or "LIMIT").strip().upper()
    if t in ("MARKET", "LIMIT", "IOC", "FOK"):
        return OrderType(t)
    # SL / SL-M map to LIMIT for our domain model
    if t in ("SL", "SL-M"):
        return OrderType.LIMIT
    return OrderType.LIMIT


def _is_circuit_breaker_error(exc: Exception) -> bool:
    """Check if an exception message indicates an exchange circuit breaker."""
    msg = str(exc).lower()
    return any(cb in msg for cb in _CIRCUIT_BREAKER_MESSAGES)


def _is_token_expired_error(exc: Exception) -> bool:
    """Check if an exception is a Kite token expiry error."""
    exc_type = type(exc).__name__
    return exc_type in _TOKEN_EXPIRED_EXCEPTIONS


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------


class ZerodhaGateway(BrokerInterface):
    """
    Zerodha Kite Connect broker gateway.

    In **paper** mode every method works in-memory without any HTTP calls,
    returning simulated responses with proper Order/Fill event models.

    In **live** mode the gateway delegates to the Kite Connect REST API via
    the ``kiteconnect`` SDK.  It includes:
    * Automatic token expiry tracking (daily token rotation)
    * Circuit breaker detection per-symbol
    * Retry with exponential backoff for transient failures
    * Full mapping to platform Order/Position/Fill event models
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        access_token: str = "",
        *,
        paper: bool = True,
        request_token: Optional[str] = None,
        request_timeout: float = 10.0,
        on_health_failure: Optional[Callable[[], None]] = None,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._access_token = access_token
        self._paper = paper
        self._request_token = request_token
        self._request_timeout = request_timeout
        self._on_health_failure = on_health_failure

        # Kite Connect SDK instance (populated on connect())
        self._kite: Any = None
        self._connected = False

        # Token lifecycle tracking
        self._token_acquired_at: float = 0.0
        self._token_max_age: float = _TOKEN_MAX_AGE
        self._consecutive_failures: int = 0
        self._auth_failed: bool = False
        self._refresh_lock: asyncio.Lock = asyncio.Lock()

        # Paper-mode bookkeeping
        self._paper_orders: List[Dict[str, Any]] = []
        self._paper_positions: List[Dict[str, Any]] = []
        self._paper_holdings: List[Dict[str, Any]] = []

    # -- BrokerInterface properties -----------------------------------------

    @property
    def paper(self) -> bool:  # type: ignore[override]
        return self._paper

    @property
    def auth_failed(self) -> bool:
        """True if the gateway has entered a critical auth failure state."""
        return self._auth_failed

    # -- lifecycle ----------------------------------------------------------

    async def connect(self) -> None:
        """
        Establish a Kite Connect session.

        Paper mode: instant success.
        Live mode:
          1. If *access_token* is already provided, validate it.
          2. If *request_token* is provided, exchange it for an access token
             via ``kite.generate_session(request_token, api_secret)``.
          3. Otherwise raise so the caller can redirect the user to the
             Kite login URL.
        """
        if self._paper:
            self._connected = True
            logger.info("Zerodha execution (paper mode)")
            return

        if not self._api_key:
            raise ConnectionError("Zerodha live mode requires api_key")

        KiteConnect = _import_kite()
        if KiteConnect is None:
            raise ConnectionError(
                "kiteconnect package is not installed. "
                "Install with: pip install kiteconnect"
            )

        self._kite = KiteConnect(api_key=self._api_key)

        loop = asyncio.get_running_loop()

        if self._access_token:
            # Validate existing access token
            self._kite.set_access_token(self._access_token)
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, self._kite.profile),
                    timeout=self._request_timeout,
                )
                self._connected = True
                self._token_acquired_at = time.monotonic()
                self._consecutive_failures = 0
                self._auth_failed = False
                audit_logger.info(
                    "Token validated during connect at %.0f", time.time()
                )
                logger.info("Zerodha execution (live): session ready via access_token")
                return
            except Exception as exc:
                if _is_token_expired_error(exc):
                    logger.warning(
                        "Provided access_token is expired/invalid: %s", exc
                    )
                    # Fall through to request_token flow or raise
                else:
                    raise ConnectionError(
                        f"Zerodha session validation failed: {exc}"
                    ) from exc

        if self._request_token and self._api_secret:
            try:
                def _generate_session():
                    return self._kite.generate_session(
                        self._request_token, api_secret=self._api_secret
                    )

                data = await asyncio.wait_for(
                    loop.run_in_executor(None, _generate_session),
                    timeout=self._request_timeout,
                )
                self._access_token = data["access_token"]
                self._kite.set_access_token(self._access_token)
                self._connected = True
                self._token_acquired_at = time.monotonic()
                self._consecutive_failures = 0
                self._auth_failed = False
                audit_logger.info(
                    "Token acquired via request_token at %.0f", time.time()
                )
                logger.info(
                    "Zerodha execution (live): session ready via request_token"
                )
                return
            except asyncio.TimeoutError:
                logger.error("Zerodha session generation timed out")
                self._notify_health_failure()
                raise ConnectionError("Zerodha session generation timed out")
            except Exception as exc:
                logger.error("Zerodha session generation failed: %s", exc)
                self._notify_health_failure()
                raise ConnectionError(
                    f"Zerodha session generation failed: {exc}"
                ) from exc

        # No valid credentials -- tell the caller what to do
        login_url = self._kite.login_url()
        raise ConnectionError(
            f"No valid access_token or request_token. "
            f"Redirect user to: {login_url}"
        )

    async def disconnect(self) -> None:
        """Invalidate the session."""
        if self._kite is not None and self._access_token:
            try:
                loop = asyncio.get_running_loop()
                token = self._access_token

                def _invalidate():
                    self._kite.invalidate_access_token(token)

                await asyncio.wait_for(
                    loop.run_in_executor(None, _invalidate),
                    timeout=self._request_timeout,
                )
                audit_logger.info("Access token invalidated")
            except Exception as exc:
                logger.warning("Failed to invalidate Kite access token: %s", exc)
        self._kite = None
        self._connected = False
        self._access_token = ""
        logger.info("Zerodha session disconnected")

    # -- authentication helpers ---------------------------------------------

    def login_url(self) -> str:
        """
        Return the Kite Connect login URL the user should be redirected to.

        After the user logs in, Kite redirects back with a ``request_token``
        that must be exchanged for an ``access_token`` via :meth:`connect`.
        """
        if not self._api_key:
            raise ValueError("api_key is required to generate login URL")
        return f"https://kite.zerodha.com/connect/login?v=3&api_key={self._api_key}"

    async def refresh_access_token(self, request_token: str) -> str:
        """
        Exchange *request_token* for a new access token.

        This is the second leg of the Kite Connect OAuth flow:
        1. User visits :meth:`login_url` and authenticates.
        2. Kite redirects to your callback URL with ``request_token``.
        3. You call this method to get the ``access_token``.

        Returns the new access token string.
        """
        if self._paper:
            fake_token = f"paper_token_{uuid4().hex[:12]}"
            self._access_token = fake_token
            self._connected = True
            return fake_token

        if not self._api_key or not self._api_secret:
            raise ValueError("api_key and api_secret required for token exchange")

        KiteConnect = _import_kite()
        if KiteConnect is None:
            raise KiteGatewayError(
                "kiteconnect package not installed",
                errorcode="SDK_MISSING",
            )

        if self._kite is None:
            self._kite = KiteConnect(api_key=self._api_key)

        loop = asyncio.get_running_loop()
        try:
            def _generate():
                return self._kite.generate_session(
                    request_token, api_secret=self._api_secret
                )

            data = await asyncio.wait_for(
                loop.run_in_executor(None, _generate),
                timeout=self._request_timeout,
            )
            self._access_token = data["access_token"]
            self._kite.set_access_token(self._access_token)
            self._connected = True
            self._token_acquired_at = time.monotonic()
            self._consecutive_failures = 0
            self._auth_failed = False
            audit_logger.info(
                "Token refreshed via request_token at %.0f", time.time()
            )
            return self._access_token
        except Exception as exc:
            logger.error("Token exchange failed: %s", exc)
            self._notify_health_failure()
            raise KiteGatewayError(
                f"Token exchange failed: {exc}"
            ) from exc

    # -- token lifecycle ----------------------------------------------------

    def _is_token_expiring(self) -> bool:
        """Return True if the current token is within the headroom window."""
        if self._token_acquired_at == 0.0:
            return False
        elapsed = time.monotonic() - self._token_acquired_at
        return elapsed >= (self._token_max_age - _TOKEN_REFRESH_HEADROOM)

    async def _ensure_valid_token(self) -> None:
        """Check token validity before each live API call.

        Since Kite tokens cannot be silently refreshed (user must complete OAuth
        flow), this raises an error when the token has expired so the caller can
        trigger re-authentication.
        """
        if self._paper or self._kite is None:
            return

        if self._auth_failed:
            raise KiteGatewayError(
                "Zerodha authentication has failed repeatedly. "
                "Manual re-login required.",
                errorcode="AUTH_FAILED",
            )

        if self._is_token_expiring():
            logger.warning(
                "Kite access token approaching expiry (acquired %.0fs ago). "
                "User must re-authenticate via login URL.",
                time.monotonic() - self._token_acquired_at,
            )
            # Don't block -- just warn.  The actual token validation will
            # happen when the API call returns a TokenException.

    def _notify_health_failure(self) -> None:
        """Invoke the health failure callback, if registered."""
        if self._on_health_failure:
            try:
                self._on_health_failure()
            except Exception as exc:
                logger.warning("on_health_failure callback error: %s", exc)

    # -- core broker call executor ------------------------------------------

    async def _execute_broker_call(
        self,
        operation_name: str,
        broker_fn: Callable[[], Any],
        *,
        retries: int = _MAX_RETRIES,
        symbol: Optional[str] = None,
    ) -> Any:
        """Execute a synchronous Kite SDK call with retry and error handling.

        1. Ensure token is valid (or warn on approaching expiry).
        2. Run *broker_fn* in a thread-pool executor.
        3. On token expiry: mark auth_failed (Kite requires manual re-login).
        4. On circuit breaker: track per-symbol halt.
        5. On transient errors: retry with exponential backoff.
        """
        try:
            from src.monitoring.metrics import (
                track_broker_failure,
                track_broker_latency,
                track_broker_session_expired,
            )
        except ImportError:
            # Metrics module may not be available in all environments
            def track_broker_failure(op, reason): pass  # noqa: E704
            def track_broker_latency(op, seconds): pass  # noqa: E704
            def track_broker_session_expired(): pass  # noqa: E704

        await self._ensure_valid_token()

        loop = asyncio.get_running_loop()
        last_exc: Optional[Exception] = None

        for attempt in range(max(retries, 1)):
            t0 = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, broker_fn),
                    timeout=self._request_timeout + 2,
                )
                elapsed = time.perf_counter() - t0
                track_broker_latency(operation_name, elapsed)
                # Reset failure counter on success
                self._consecutive_failures = 0
                return result

            except asyncio.TimeoutError:
                elapsed = time.perf_counter() - t0
                track_broker_latency(operation_name, elapsed)
                track_broker_failure(operation_name, "timeout")
                logger.warning(
                    "Zerodha %s timeout (attempt %d/%d, %.2fs)",
                    operation_name, attempt + 1, retries, elapsed,
                )
                last_exc = KiteGatewayError(
                    f"{operation_name} timeout after {elapsed:.2f}s",
                    errorcode="TIMEOUT",
                )

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                track_broker_latency(operation_name, elapsed)

                # Token expired -- Kite requires manual re-auth
                if _is_token_expired_error(exc):
                    track_broker_session_expired()
                    self._consecutive_failures += 1
                    audit_logger.warning(
                        "Token expired during %s (attempt %d): %s",
                        operation_name, attempt + 1, exc,
                    )
                    if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                        self._auth_failed = True
                        logger.critical(
                            "CRITICAL: Zerodha token has expired %d consecutive "
                            "times. User must re-authenticate. Login URL: %s",
                            self._consecutive_failures,
                            self.login_url() if self._api_key else "(no api_key)",
                        )
                        self._notify_health_failure()
                    raise KiteGatewayError(
                        f"Kite access token expired. Re-authenticate at: "
                        f"{self.login_url() if self._api_key else '(configure api_key)'}",
                        errorcode="TOKEN_EXPIRED",
                    ) from exc

                # Circuit breaker -- no retry, track per-symbol
                if _is_circuit_breaker_error(exc):
                    if symbol:
                        _circuit_halted_symbols[symbol] = time.monotonic()
                    track_broker_failure(
                        operation_name, "circuit_breaker"
                    )
                    logger.warning(
                        "Exchange circuit breaker during %s for %s: %s",
                        operation_name, symbol or "unknown", exc,
                    )
                    raise KiteGatewayError(
                        f"Exchange circuit breaker halt: {exc}",
                        errorcode="CIRCUIT_HALT",
                    ) from exc

                # Order-specific errors (OrderException) -- don't retry
                if type(exc).__name__ == "OrderException":
                    track_broker_failure(operation_name, "order_rejected")
                    raise KiteGatewayError(
                        f"Order rejected by Kite: {exc}",
                        errorcode="ORDER_REJECTED",
                    ) from exc

                # Input validation errors -- don't retry
                if type(exc).__name__ == "InputException":
                    track_broker_failure(operation_name, "input_error")
                    raise KiteGatewayError(
                        f"Invalid input: {exc}",
                        errorcode="INPUT_ERROR",
                    ) from exc

                # Transient / network errors -- retry with backoff
                track_broker_failure(
                    operation_name, type(exc).__name__
                )
                logger.warning(
                    "Zerodha %s failed (attempt %d/%d): %s",
                    operation_name, attempt + 1, retries, exc,
                )
                last_exc = exc

            # Exponential backoff before retry
            if attempt < retries - 1:
                backoff = _RETRY_BACKOFF_BASE * (2 ** attempt)
                await asyncio.sleep(backoff)

        # All retries exhausted
        self._consecutive_failures += 1
        self._notify_health_failure()
        if isinstance(last_exc, KiteGatewayError):
            raise last_exc
        raise KiteGatewayError(
            f"{operation_name} failed after {retries} attempts: {last_exc}",
            errorcode="RETRIES_EXHAUSTED",
        ) from last_exc

    # -- internal helpers ---------------------------------------------------

    def _require_live_session(self) -> None:
        """Raise if live mode but no active session."""
        if not self._connected or self._kite is None:
            raise ConnectionError(
                "Zerodha gateway is not connected. Call connect() first."
            )

    def _check_circuit_halt(self, symbol: str) -> None:
        """Raise if the symbol is under a circuit halt cooldown."""
        if symbol in _circuit_halted_symbols:
            halt_time = _circuit_halted_symbols[symbol]
            elapsed = time.monotonic() - halt_time
            if elapsed < _CIRCUIT_HALT_WINDOW:
                raise KiteGatewayError(
                    f"Symbol {symbol} under circuit halt ({elapsed:.0f}s ago, "
                    f"cooldown {_CIRCUIT_HALT_WINDOW:.0f}s)",
                    errorcode="CIRCUIT_HALT",
                )
        # BUG 29 FIX: Don't delete during iteration. Snapshot keys first,
        # then delete expired entries safely.
        symbols_to_clear = [
            k for k, v in _circuit_halted_symbols.items()
            if time.monotonic() - v >= _CIRCUIT_HALT_WINDOW
        ]
        for k in symbols_to_clear:
            del _circuit_halted_symbols[k]

    # -- parse helpers (Kite response -> domain models) ---------------------

    def _order_from_kite(self, row: Dict[str, Any], strategy_id: str = "") -> Order:
        """Convert a Kite order dict to our Order domain model."""
        status_str = row.get("status", "")
        filled_qty = float(row.get("filled_quantity", 0) or 0)
        order_status = _to_order_status(status_str)

        # If partially filled, update status
        total_qty = float(row.get("quantity", 0) or 0)
        if (
            filled_qty > 0
            and filled_qty < total_qty
            and order_status == OrderStatus.LIVE
        ):
            order_status = OrderStatus.PARTIALLY_FILLED

        return Order(
            order_id=str(row.get("order_id", "")),
            strategy_id=strategy_id or row.get("tag", ""),
            symbol=row.get("tradingsymbol", ""),
            exchange=_parse_exchange(row.get("exchange", "NSE")),
            side=_parse_side(row.get("transaction_type", "BUY")),
            quantity=total_qty,
            order_type=_parse_order_type(row.get("order_type", "LIMIT")),
            limit_price=float(row.get("price", 0) or 0) or None,
            status=order_status,
            filled_qty=filled_qty,
            avg_price=float(row.get("average_price", 0) or 0) or None,
            broker_order_id=str(row.get("order_id", "")),
            metadata={
                "broker": "zerodha",
                "product": row.get("product", ""),
                "variety": row.get("variety", ""),
                "validity": row.get("validity", ""),
                "trigger_price": row.get("trigger_price", 0),
                "status_message": row.get("status_message", ""),
                "exchange_order_id": row.get("exchange_order_id", ""),
            },
        )

    def _position_from_kite(self, row: Dict[str, Any]) -> Position:
        """Convert a Kite position dict to our Position domain model."""
        net_qty = int(row.get("quantity", 0) or 0)
        side = SignalSide.BUY if net_qty >= 0 else SignalSide.SELL
        avg_price = float(row.get("average_price", 0) or 0)
        ltp = float(row.get("last_price", 0) or 0)
        pnl = float(row.get("pnl", 0) or 0)

        return Position(
            symbol=row.get("tradingsymbol", ""),
            exchange=_parse_exchange(row.get("exchange", "NSE")),
            side=side,
            quantity=abs(net_qty),
            avg_price=avg_price,
            unrealized_pnl=pnl,
            metadata={
                "broker": "zerodha",
                "product": row.get("product", ""),
                "buy_quantity": row.get("buy_quantity", 0),
                "sell_quantity": row.get("sell_quantity", 0),
                "buy_price": row.get("buy_price", 0),
                "sell_price": row.get("sell_price", 0),
                "ltp": ltp,
                "multiplier": row.get("multiplier", 1),
            },
        )

    # -- orders -------------------------------------------------------------

    @staticmethod
    def _dict_to_order(record: Dict[str, Any], kwargs: Dict[str, Any] = {}) -> Order:
        """Convert a raw order dict (paper or live) to an Order object."""
        _KITE_SIDE_MAP = {"BUY": SignalSide.BUY, "SELL": SignalSide.SELL}
        _KITE_EXCHANGE_MAP = {
            "NSE": Exchange.NSE, "BSE": Exchange.BSE,
            "NFO": Exchange.NFO, "BFO": Exchange.BFO,
            "CDS": Exchange.CDS, "MCX": Exchange.MCX,
        }
        _KITE_ORDER_TYPE_MAP = {
            "MARKET": OrderType.MARKET, "LIMIT": OrderType.LIMIT,
        }
        _KITE_STATUS_MAP = {
            "PENDING": OrderStatus.PENDING, "COMPLETE": OrderStatus.FILLED,
            "CANCELLED": OrderStatus.CANCELLED, "REJECTED": OrderStatus.REJECTED,
        }
        raw_exchange = record.get("exchange", "NSE")
        return Order(
            order_id=str(record.get("order_id", "")),
            strategy_id=record.get("tag", "") or kwargs.get("strategy_id", ""),
            symbol=record.get("tradingsymbol", ""),
            exchange=_KITE_EXCHANGE_MAP.get(raw_exchange, Exchange.NSE),
            side=_KITE_SIDE_MAP.get(record.get("transaction_type", "BUY"), SignalSide.BUY),
            quantity=float(record.get("quantity", 0)),
            order_type=_KITE_ORDER_TYPE_MAP.get(record.get("order_type", "MARKET"), OrderType.MARKET),
            limit_price=record.get("price") or None,
            status=_KITE_STATUS_MAP.get(record.get("status", "PENDING"), OrderStatus.PENDING),
            filled_qty=float(record.get("filled_quantity", 0)),
            avg_price=record.get("average_price"),
            broker_order_id=str(record.get("order_id", "")),
        )

    async def place_order(
        self,
        symbol: str,
        exchange: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float] = None,
        product_type: str = "INTRADAY",
        **kwargs: Any,
    ) -> Order:
        kite_exchange = _EXCHANGE_TO_KITE.get(exchange.upper(), exchange.upper())
        kite_side = _SIDE_TO_KITE.get(side.upper(), side.upper())
        kite_order_type = _ORDER_TYPE_TO_KITE.get(
            order_type.upper(), order_type.upper()
        )
        kite_product = _PRODUCT_TYPE_TO_KITE.get(
            product_type.upper(), product_type.upper()
        )

        # Check circuit halt before placing
        self._check_circuit_halt(symbol)

        # --- paper mode ---
        if self._paper:
            order_id = str(uuid4())
            fill_price = price or 0.0
            order_record = {
                "order_id": order_id,
                "status": "PENDING",
                "tradingsymbol": symbol,
                "exchange": kite_exchange,
                "transaction_type": kite_side,
                "quantity": int(quantity),
                "order_type": kite_order_type,
                "product": kite_product,
                "price": fill_price,
                "trigger_price": kwargs.get("trigger_price", 0.0),
                "tag": kwargs.get("strategy_id", ""),
                "paper": True,
                "broker": "zerodha",
            }
            self._paper_orders.append(order_record)

            # Simulate immediate fill for MARKET orders in paper mode
            if kite_order_type == "MARKET":
                order_record["status"] = "COMPLETE"
                order_record["filled_quantity"] = int(quantity)
                order_record["average_price"] = fill_price
                # Track in paper positions
                self._update_paper_position(
                    symbol, kite_exchange, kite_side, int(quantity), fill_price,
                    kite_product,
                )

            logger.info(
                "Paper order placed: %s %s %s x%d @ %s [%s]",
                kite_side, symbol, kite_order_type, int(quantity),
                fill_price or "MKT", order_id[:8],
            )
            return self._dict_to_order(order_record, kwargs)

        # --- live mode ---
        self._require_live_session()

        # Build Kite Connect place_order params
        # https://kite.trade/docs/connect/v3/orders/#regular-order-parameters
        variety = kwargs.get("variety", "regular")
        params: Dict[str, Any] = {
            "variety": variety,
            "exchange": kite_exchange,
            "tradingsymbol": symbol,
            "transaction_type": kite_side,
            "quantity": int(quantity),
            "order_type": kite_order_type,
            "product": kite_product,
            "validity": kwargs.get("validity", "DAY"),
        }
        if price is not None and kite_order_type in ("LIMIT", "SL"):
            params["price"] = price
        if kwargs.get("trigger_price") is not None:
            params["trigger_price"] = kwargs["trigger_price"]
        if kwargs.get("disclosed_quantity") is not None:
            params["disclosed_quantity"] = kwargs["disclosed_quantity"]
        tag = kwargs.get("strategy_id", "")
        if tag and len(tag) <= 20:
            params["tag"] = tag

        logger.info(
            "Placing live order: %s %s %s x%d @ %s (product=%s, variety=%s)",
            kite_side, symbol, kite_order_type, int(quantity),
            price or "MKT", kite_product, variety,
        )

        order_id = await self._execute_broker_call(
            "place_order",
            lambda: self._kite.place_order(**params),
            symbol=symbol,
        )

        result = {
            "order_id": str(order_id),
            "status": "PENDING",
            "tradingsymbol": symbol,
            "exchange": kite_exchange,
            "transaction_type": kite_side,
            "quantity": int(quantity),
            "order_type": kite_order_type,
            "product": kite_product,
            "price": price or 0.0,
            "broker": "zerodha",
        }
        logger.info(
            "Live order placed: %s (kite order_id=%s)", symbol, order_id
        )
        return self._dict_to_order(result, kwargs)

    async def cancel_order(self, order_id: str, **kwargs: Any) -> bool:
        if self._paper:
            for o in self._paper_orders:
                if o["order_id"] == order_id:
                    if o["status"] not in ("COMPLETE", "CANCELLED", "REJECTED"):
                        o["status"] = "CANCELLED"
                        logger.info("Paper order cancelled: %s", order_id[:8])
                        return True
                    return False
            logger.warning("Paper cancel: order %s not found", order_id[:8])
            return False

        self._require_live_session()
        variety = kwargs.get("variety", "regular")

        logger.info(
            "Cancelling live order: %s (variety=%s)", order_id, variety
        )

        try:
            await self._execute_broker_call(
                "cancel_order",
                lambda: self._kite.cancel_order(
                    variety=variety, order_id=order_id,
                ),
            )
            logger.info("Live order cancelled: %s", order_id)
            return True
        except KiteGatewayError as exc:
            logger.warning("Cancel order %s failed: %s", order_id, exc)
            return False

    # -- read-only queries --------------------------------------------------

    async def get_positions(self) -> list:
        if self._paper:
            return list(self._paper_positions)

        self._require_live_session()

        raw = await self._execute_broker_call(
            "get_positions",
            lambda: self._kite.positions(),
        )

        # Kite returns {"net": [...], "day": [...]}
        net_positions = raw.get("net", []) if isinstance(raw, dict) else raw
        positions = []
        for row in (net_positions or []):
            qty = int(row.get("quantity", 0) or 0)
            if qty != 0:
                positions.append(self._position_from_kite(row))

        logger.debug("Fetched %d net positions from Kite", len(positions))
        return positions

    async def get_orders(self, limit: int = 50) -> list:
        if self._paper:
            return list(reversed(self._paper_orders))[:limit]

        self._require_live_session()

        raw_orders = await self._execute_broker_call(
            "get_orders",
            lambda: self._kite.orders(),
        )

        orders = []
        for row in (raw_orders or [])[:limit]:
            orders.append(self._order_from_kite(row))

        logger.debug("Fetched %d orders from Kite", len(orders))
        return orders

    async def get_order_status(
        self, order_id: str, broker_order_id: Optional[str] = None
    ) -> OrderStatus:
        """Check the status of a specific order.

        In paper mode, looks up the in-memory order book.
        In live mode, fetches order history from Kite and returns the latest
        status.
        """
        if self._paper:
            for o in self._paper_orders:
                if o["order_id"] == order_id:
                    return _to_order_status(o.get("status", "PENDING"))
            return OrderStatus.PENDING

        self._require_live_session()

        target_id = broker_order_id or order_id

        try:
            raw_history = await self._execute_broker_call(
                "get_order_status",
                lambda: self._kite.order_history(order_id=target_id),
            )
            if not raw_history:
                return OrderStatus.PENDING

            # order_history returns a list of status updates; last is most recent
            latest = raw_history[-1] if isinstance(raw_history, list) else raw_history
            status_str = latest.get("status", "")
            return _to_order_status(status_str)

        except KiteGatewayError:
            logger.warning(
                "Failed to get order status for %s", target_id
            )
            return OrderStatus.PENDING

    async def health_check(self) -> Dict[str, object]:
        """Verify Zerodha gateway connectivity.

        Paper mode: always healthy.
        Live mode: attempt to fetch the profile (lightweight API call)
        to confirm the session is alive and the broker API is reachable.
        """
        if self._paper:
            return {"healthy": True, "broker": "zerodha", "mode": "paper", "detail": "ok"}
        if not self._connected or self._kite is None:
            return {"healthy": False, "broker": "zerodha", "mode": "live", "detail": "not_connected"}
        if self._auth_failed:
            return {"healthy": False, "broker": "zerodha", "mode": "live", "detail": "auth_failed"}
        try:
            await self._execute_broker_call(
                "health_check",
                lambda: self._kite.profile(),
            )
            return {"healthy": True, "broker": "zerodha", "mode": "live", "detail": "ok"}
        except Exception as e:
            logger.warning("Zerodha health check failed: %s", e)
            return {"healthy": False, "broker": "zerodha", "mode": "live", "detail": str(e)}

    async def get_quote(self, symbol: str, exchange: str) -> dict:
        kite_exchange = _EXCHANGE_TO_KITE.get(exchange.upper(), exchange.upper())
        instrument = f"{kite_exchange}:{symbol}"

        if self._paper:
            return {
                "ltp": 0.0,
                "instrument": instrument,
                "paper": True,
            }

        self._require_live_session()

        data = await self._execute_broker_call(
            "get_quote",
            lambda: self._kite.quote([instrument]),
        )

        q = data.get(instrument, {})
        depth = q.get("depth", {})
        buy_depth = depth.get("buy", [{}])
        sell_depth = depth.get("sell", [{}])
        ohlc = q.get("ohlc", {})

        return {
            "ltp": q.get("last_price", 0.0),
            "bid": buy_depth[0].get("price", 0.0) if buy_depth else 0.0,
            "ask": sell_depth[0].get("price", 0.0) if sell_depth else 0.0,
            "volume": q.get("volume", 0),
            "open": ohlc.get("open", 0.0),
            "high": ohlc.get("high", 0.0),
            "low": ohlc.get("low", 0.0),
            "close": ohlc.get("close", 0.0),
            "timestamp": q.get("timestamp"),
            "instrument": instrument,
        }

    async def get_holdings(self) -> list:
        if self._paper:
            return list(self._paper_holdings)

        self._require_live_session()

        raw_holdings = await self._execute_broker_call(
            "get_holdings",
            lambda: self._kite.holdings(),
        )

        holdings = []
        for row in (raw_holdings or []):
            qty = int(row.get("quantity", 0) or 0)
            if qty != 0:
                holdings.append({
                    "symbol": row.get("tradingsymbol", ""),
                    "exchange": row.get("exchange", ""),
                    "quantity": qty,
                    "avg_price": float(row.get("average_price", 0) or 0),
                    "ltp": float(row.get("last_price", 0) or 0),
                    "pnl": float(row.get("pnl", 0) or 0),
                    "isin": row.get("isin", ""),
                    "product": row.get("product", ""),
                    "broker": "zerodha",
                })

        logger.debug("Fetched %d holdings from Kite", len(holdings))
        return holdings

    # -- status -------------------------------------------------------------

    def is_connected(self) -> bool:
        return self._connected

    # -- paper-mode helpers -------------------------------------------------

    def _update_paper_position(
        self,
        symbol: str,
        exchange: str,
        side: str,
        quantity: int,
        price: float,
        product: str,
    ) -> None:
        """Update paper-mode positions after a simulated fill."""
        signed_qty = quantity if side == "BUY" else -quantity

        for pos in self._paper_positions:
            if pos["tradingsymbol"] == symbol and pos["exchange"] == exchange:
                old_qty = pos.get("quantity", 0)
                old_avg = pos.get("average_price", 0.0)

                new_qty = old_qty + signed_qty
                if new_qty == 0:
                    self._paper_positions.remove(pos)
                    return

                # Weighted average price for same-direction additions
                if (old_qty >= 0 and signed_qty > 0) or (old_qty <= 0 and signed_qty < 0):
                    total_cost = (abs(old_qty) * old_avg) + (abs(signed_qty) * price)
                    pos["average_price"] = total_cost / abs(new_qty) if new_qty != 0 else 0.0

                pos["quantity"] = new_qty
                return

        # New position
        self._paper_positions.append({
            "tradingsymbol": symbol,
            "exchange": exchange,
            "quantity": signed_qty,
            "average_price": price,
            "product": product,
            "last_price": price,
            "pnl": 0.0,
            "broker": "zerodha",
            "paper": True,
        })

    def paper_fill_order(
        self,
        order_id: str,
        fill_price: float,
        fill_qty: Optional[int] = None,
    ) -> bool:
        """
        Simulate a fill on a paper order (for use by the paper-mode engine).

        Returns True if the order was found and filled.
        """
        for o in self._paper_orders:
            if o["order_id"] == order_id and o["status"] in ("PENDING", "OPEN"):
                qty = fill_qty or o["quantity"]
                o["status"] = "COMPLETE"
                o["filled_quantity"] = qty
                o["average_price"] = fill_price
                self._update_paper_position(
                    o["tradingsymbol"],
                    o["exchange"],
                    o["transaction_type"],
                    qty,
                    fill_price,
                    o.get("product", "MIS"),
                )
                logger.info(
                    "Paper fill: %s x%d @ %.2f [%s]",
                    o["tradingsymbol"], qty, fill_price, order_id[:8],
                )
                return True
        return False


# ---------------------------------------------------------------------------
# Auto-register with the broker factory
# ---------------------------------------------------------------------------
register_broker("zerodha", ZerodhaGateway)
register_broker("kite", ZerodhaGateway)  # alias
