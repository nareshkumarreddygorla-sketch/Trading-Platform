"""
Zerodha Kite Connect execution gateway.

Live: uses the Kite Connect v3 REST API for order placement, positions,
      holdings, and market quotes.
Paper: in-memory order simulation (no HTTP), identical pattern to Angel One.

This module is structured around the real Kite Connect API but raises
``NotImplementedError`` for methods that require a live API key / session.
The correct endpoint paths, payload shapes, and header conventions are all
in place so that wiring up the ``kiteconnect`` Python SDK (or raw HTTP) is
a drop-in change.

References
----------
* Kite Connect API docs : https://kite.trade/docs/connect/v3/
* kiteconnect PyPI pkg  : https://pypi.org/project/kiteconnect/
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .broker_interface import BrokerInterface, register_broker

logger = logging.getLogger(__name__)


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
_KITE_ORDER_TYPE_MAP: Dict[str, str] = {v: k for k, v in _ORDER_TYPE_TO_KITE.items()}
_KITE_PRODUCT_MAP: Dict[str, str] = {"MIS": "INTRADAY", "CNC": "CNC", "NRML": "NRML"}

# Kite order status -> our canonical status strings
_KITE_STATUS_MAP: Dict[str, str] = {
    "OPEN": "LIVE",
    "COMPLETE": "FILLED",
    "CANCELLED": "CANCELLED",
    "REJECTED": "REJECTED",
    "TRIGGER PENDING": "PENDING",
    "OPEN PENDING": "PENDING",
    "VALIDATION PENDING": "PENDING",
    "PUT ORDER REQ RECEIVED": "PENDING",
    "MODIFY VALIDATION PENDING": "LIVE",
    "MODIFY ORDER REQ RECEIVED": "LIVE",
    "CANCEL PENDING": "LIVE",
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


def _to_our_status(kite_status: str) -> str:
    """Map a Kite status string to our canonical status."""
    key = (kite_status or "").strip().upper()
    return _KITE_STATUS_MAP.get(key, "PENDING")


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------


class ZerodhaGateway(BrokerInterface):
    """
    Zerodha Kite Connect broker gateway.

    In **paper** mode every method works in-memory without any HTTP calls,
    returning simulated responses.

    In **live** mode the gateway delegates to the Kite Connect REST API.
    Methods that require a real API session raise ``NotImplementedError``
    until the ``kiteconnect`` SDK is wired in and valid credentials are
    provided.

    Constructor kwargs match what ``BrokerFactory.create("zerodha", ...)``
    would forward.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        access_token: str = "",
        *,
        paper: bool = True,
        request_token: Optional[str] = None,
        request_timeout: float = 15.0,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._access_token = access_token
        self._paper = paper
        self._request_token = request_token
        self._request_timeout = request_timeout

        # Will hold the kiteconnect.KiteConnect instance in live mode
        self._kite: Any = None
        self._connected = False

        # Paper-mode bookkeeping
        self._paper_orders: List[Dict[str, Any]] = []
        self._paper_positions: List[Dict[str, Any]] = []
        self._paper_holdings: List[Dict[str, Any]] = []

    # -- BrokerInterface properties -----------------------------------------

    @property
    def paper(self) -> bool:  # type: ignore[override]
        return self._paper

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
            raise ConnectionError(
                "Zerodha live mode requires api_key"
            )

        # ----- real Kite Connect integration point -----
        # In production, uncomment the block below and install kiteconnect:
        #
        #   from kiteconnect import KiteConnect
        #   self._kite = KiteConnect(api_key=self._api_key)
        #
        #   if self._access_token:
        #       self._kite.set_access_token(self._access_token)
        #   elif self._request_token and self._api_secret:
        #       data = self._kite.generate_session(
        #           self._request_token, api_secret=self._api_secret
        #       )
        #       self._access_token = data["access_token"]
        #       self._kite.set_access_token(self._access_token)
        #   else:
        #       login_url = self._kite.login_url()
        #       raise ConnectionError(
        #           f"No access_token or request_token. "
        #           f"Redirect user to: {login_url}"
        #       )
        #
        #   # Quick validation
        #   self._kite.profile()
        #   self._connected = True
        #   logger.info("Zerodha execution (live): session ready")
        #   return

        raise NotImplementedError(
            "Zerodha live mode requires the kiteconnect SDK and valid API "
            "credentials. Install kiteconnect and provide api_key + "
            "(access_token | request_token + api_secret)."
        )

    async def disconnect(self) -> None:
        """Invalidate the session."""
        if self._kite is not None and self._access_token:
            # self._kite.invalidate_access_token(self._access_token)
            pass
        self._kite = None
        self._connected = False
        self._access_token = ""
        logger.info("Zerodha session disconnected")

    # -- orders -------------------------------------------------------------

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
    ) -> dict:
        kite_exchange = _EXCHANGE_TO_KITE.get(exchange.upper(), exchange.upper())
        kite_side = _SIDE_TO_KITE.get(side.upper(), side.upper())
        kite_order_type = _ORDER_TYPE_TO_KITE.get(order_type.upper(), order_type.upper())
        kite_product = _PRODUCT_TYPE_TO_KITE.get(product_type.upper(), product_type.upper())

        # --- paper mode ---
        if self._paper:
            order_id = str(uuid4())
            order_record = {
                "order_id": order_id,
                "status": "PENDING",
                "tradingsymbol": symbol,
                "exchange": kite_exchange,
                "transaction_type": kite_side,
                "quantity": int(quantity),
                "order_type": kite_order_type,
                "product": kite_product,
                "price": price or 0.0,
                "trigger_price": kwargs.get("trigger_price", 0.0),
                "tag": kwargs.get("strategy_id", ""),
                "paper": True,
                "broker": "zerodha",
            }
            self._paper_orders.append(order_record)
            return order_record

        # --- live mode ---
        self._require_live_session()

        # Kite Connect place_order params
        # https://kite.trade/docs/connect/v3/orders/#regular-order-parameters
        params: Dict[str, Any] = {
            "variety": kwargs.get("variety", "regular"),
            "exchange": kite_exchange,
            "tradingsymbol": symbol,
            "transaction_type": kite_side,
            "quantity": int(quantity),
            "order_type": kite_order_type,
            "product": kite_product,
            "validity": kwargs.get("validity", "DAY"),
        }
        if price is not None and kite_order_type == "LIMIT":
            params["price"] = price
        if kwargs.get("trigger_price") is not None:
            params["trigger_price"] = kwargs["trigger_price"]
        if kwargs.get("disclosed_quantity") is not None:
            params["disclosed_quantity"] = kwargs["disclosed_quantity"]
        tag = kwargs.get("strategy_id", "")
        if tag and len(tag) <= 20:
            params["tag"] = tag

        # ----- real call -----
        # order_id = self._kite.place_order(**params)
        # return {"order_id": str(order_id), "status": "PENDING", "broker": "zerodha"}
        raise NotImplementedError("Zerodha live place_order requires kiteconnect SDK")

    async def cancel_order(self, order_id: str, **kwargs: Any) -> bool:
        if self._paper:
            for o in self._paper_orders:
                if o["order_id"] == order_id:
                    o["status"] = "CANCELLED"
                    return True
            return False

        self._require_live_session()
        variety = kwargs.get("variety", "regular")

        # ----- real call -----
        # self._kite.cancel_order(variety=variety, order_id=order_id)
        # return True
        raise NotImplementedError("Zerodha live cancel_order requires kiteconnect SDK")

    # -- read-only queries --------------------------------------------------

    async def get_positions(self) -> list:
        if self._paper:
            return list(self._paper_positions)

        self._require_live_session()

        # ----- real call -----
        # positions = self._kite.positions()
        # # positions = {"net": [...], "day": [...]}
        # return positions.get("net", [])
        raise NotImplementedError("Zerodha live get_positions requires kiteconnect SDK")

    async def get_orders(self, limit: int = 50) -> list:
        if self._paper:
            return list(reversed(self._paper_orders))[:limit]

        self._require_live_session()

        # ----- real call -----
        # orders = self._kite.orders()
        # for o in orders:
        #     o["status"] = _to_our_status(o.get("status", ""))
        # return orders[:limit]
        raise NotImplementedError("Zerodha live get_orders requires kiteconnect SDK")

    async def get_quote(self, symbol: str, exchange: str) -> dict:
        kite_exchange = _EXCHANGE_TO_KITE.get(exchange.upper(), exchange.upper())
        instrument = f"{kite_exchange}:{symbol}"

        if self._paper:
            # Paper mode returns a zero-quote placeholder
            return {
                "ltp": 0.0,
                "instrument": instrument,
                "paper": True,
            }

        self._require_live_session()

        # ----- real call -----
        # data = self._kite.quote([instrument])
        # q = data.get(instrument, {})
        # return {
        #     "ltp": q.get("last_price", 0.0),
        #     "bid": q.get("depth", {}).get("buy", [{}])[0].get("price", 0.0),
        #     "ask": q.get("depth", {}).get("sell", [{}])[0].get("price", 0.0),
        #     "volume": q.get("volume", 0),
        #     "open": q.get("ohlc", {}).get("open", 0.0),
        #     "high": q.get("ohlc", {}).get("high", 0.0),
        #     "low": q.get("ohlc", {}).get("low", 0.0),
        #     "close": q.get("ohlc", {}).get("close", 0.0),
        #     "timestamp": q.get("timestamp"),
        #     "instrument": instrument,
        # }
        raise NotImplementedError("Zerodha live get_quote requires kiteconnect SDK")

    async def get_holdings(self) -> list:
        if self._paper:
            return list(self._paper_holdings)

        self._require_live_session()

        # ----- real call -----
        # return self._kite.holdings()
        raise NotImplementedError("Zerodha live get_holdings requires kiteconnect SDK")

    # -- status -------------------------------------------------------------

    def is_connected(self) -> bool:
        return self._connected

    # -- authentication helpers ---------------------------------------------

    def login_url(self) -> str:
        """
        Return the Kite Connect login URL the user should be redirected to.

        After the user logs in, Kite redirects back with a ``request_token``
        that must be exchanged for an ``access_token`` via
        :meth:`connect`.
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

        Returns
        -------
        str
            The new access token.
        """
        if self._paper:
            fake_token = f"paper_token_{uuid4().hex[:12]}"
            self._access_token = fake_token
            self._connected = True
            return fake_token

        if not self._api_key or not self._api_secret:
            raise ValueError("api_key and api_secret required for token exchange")

        # ----- real call -----
        # from kiteconnect import KiteConnect
        # kite = KiteConnect(api_key=self._api_key)
        # data = kite.generate_session(request_token, api_secret=self._api_secret)
        # self._access_token = data["access_token"]
        # self._kite = kite
        # self._kite.set_access_token(self._access_token)
        # self._connected = True
        # return self._access_token
        raise NotImplementedError(
            "Zerodha live token exchange requires kiteconnect SDK"
        )

    # -- internal helpers ---------------------------------------------------

    def _require_live_session(self) -> None:
        """Raise if live mode but no active session."""
        if not self._connected or self._kite is None:
            raise ConnectionError(
                "Zerodha gateway is not connected. Call connect() first."
            )


# ---------------------------------------------------------------------------
# Auto-register with the broker factory
# ---------------------------------------------------------------------------
register_broker("zerodha", ZerodhaGateway)
register_broker("kite", ZerodhaGateway)  # alias
