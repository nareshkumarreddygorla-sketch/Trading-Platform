"""
Multi-broker abstraction layer.

BrokerInterface: abstract base class that all broker gateways must implement.
BrokerFactory: registry-based factory to create broker instances by name.

Design rationale
----------------
The existing ``BaseExecutionGateway`` (in ``base.py``) couples the execution
layer to a single broker.  ``BrokerInterface`` keeps the same method shapes
(so ``AngelOneExecutionGateway`` already satisfies it) while adding:
  * ``get_quote`` -- live market data (LTP / bid-ask)
  * ``get_holdings`` -- delivery holdings (distinct from intraday positions)
  * ``is_connected`` -- connectivity check for health monitors
  * ``paper`` property -- first-class paper/live flag

Method signatures deliberately use plain ``str`` / ``float`` / ``dict`` so
that callers need not import broker-specific types.  Internal gateways map
these to broker-native enums (e.g. Angel "INTRADAY" vs Kite "MIS").
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class BrokerInterface(ABC):
    """
    Unified async interface every broker gateway must implement.

    All methods accept / return plain Python types (``dict``, ``list``,
    ``bool``, ``str``, ``float``) to keep the interface broker-agnostic.
    Individual gateways are free to return richer domain objects (e.g.
    ``Order``, ``Position``) since they are dict-like Pydantic models.
    """

    # -- lifecycle ----------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Establish session / authenticate with the broker."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Tear down session / clean up resources."""
        ...

    # -- orders -------------------------------------------------------------

    @abstractmethod
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
        """
        Place an order and return broker-specific result dict.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g. ``"INFY"``, ``"RELIANCE"``).
        exchange : str
            Exchange segment (``"NSE"``, ``"BSE"``, ``"NFO"`` ...).
        side : str
            ``"BUY"`` or ``"SELL"``.
        quantity : float
            Number of shares / lots.
        order_type : str
            ``"MARKET"``, ``"LIMIT"``, etc.
        price : float, optional
            Limit price (required when *order_type* is ``LIMIT``).
        product_type : str
            ``"INTRADAY"`` (MIS), ``"CNC"`` (delivery), ``"NRML"`` etc.
        **kwargs
            Broker-specific extras (``strategy_id``, ``symboltoken`` ...).

        Returns
        -------
        dict
            Must contain at least ``{"order_id": str, "status": str}``.
        """
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str, **kwargs: Any) -> bool:
        """
        Cancel a pending / open order.

        Returns ``True`` if the cancellation was accepted by the broker.
        """
        ...

    # -- read-only queries --------------------------------------------------

    @abstractmethod
    async def get_positions(self) -> list:
        """Return current (day) positions as a list of dicts / domain objects."""
        ...

    @abstractmethod
    async def get_orders(self, limit: int = 50) -> list:
        """Return recent order book entries (newest first)."""
        ...

    @abstractmethod
    async def get_quote(self, symbol: str, exchange: str) -> dict:
        """
        Get a live quote / LTP for *symbol* on *exchange*.

        Returns
        -------
        dict
            At minimum ``{"ltp": float}``.  May also include
            ``bid``, ``ask``, ``volume``, ``open``, ``high``, ``low``,
            ``close``, ``timestamp``.
        """
        ...

    @abstractmethod
    async def get_holdings(self) -> list:
        """Return delivery holdings (long-term portfolio positions)."""
        ...

    # -- status -------------------------------------------------------------

    @abstractmethod
    def is_connected(self) -> bool:
        """Return ``True`` if the broker session is live and usable."""
        ...

    @property
    @abstractmethod
    def paper(self) -> bool:
        """``True`` when the gateway is running in paper / simulation mode."""
        ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


# Internal registry: broker_name (lower) -> gateway class
_BROKER_REGISTRY: Dict[str, Type[BrokerInterface]] = {}


def register_broker(name: str, cls: Type[BrokerInterface]) -> None:
    """Register a broker gateway class under *name* (case-insensitive)."""
    key = name.strip().lower()
    if key in _BROKER_REGISTRY:
        logger.warning("Overwriting broker registry entry %r", key)
    _BROKER_REGISTRY[key] = cls


class BrokerFactory:
    """
    Create broker gateway instances by name.

    Usage::

        factory = BrokerFactory()
        gw = factory.create("zerodha", api_key="...", paper=True)
        await gw.connect()
        result = await gw.place_order("INFY", "NSE", "BUY", 10, "MARKET")

    Brokers are discovered via :func:`register_broker` which each gateway
    module calls at import time, **or** via explicit
    :meth:`BrokerFactory.register` calls.
    """

    def __init__(self) -> None:
        # Instance-level overrides (take precedence over module-level registry)
        self._local: Dict[str, Type[BrokerInterface]] = {}

    # -- registration -------------------------------------------------------

    def register(self, name: str, cls: Type[BrokerInterface]) -> None:
        """Register a gateway class on this factory instance."""
        self._local[name.strip().lower()] = cls

    @staticmethod
    def register_global(name: str, cls: Type[BrokerInterface]) -> None:
        """Register a gateway class globally (all factory instances see it)."""
        register_broker(name, cls)

    # -- creation -----------------------------------------------------------

    def create(self, broker_name: str, **config: Any) -> BrokerInterface:
        """
        Instantiate and return a ``BrokerInterface`` for *broker_name*.

        Parameters
        ----------
        broker_name : str
            Case-insensitive broker identifier (``"angel_one"``,
            ``"zerodha"``, ``"paper"`` ...).
        **config
            Keyword arguments forwarded to the gateway constructor.

        Raises
        ------
        ValueError
            If no gateway is registered for *broker_name*.
        """
        key = broker_name.strip().lower()
        cls = self._local.get(key) or _BROKER_REGISTRY.get(key)
        if cls is None:
            available = sorted(set(self._local) | set(_BROKER_REGISTRY))
            raise ValueError(
                f"Unknown broker {broker_name!r}. "
                f"Available: {', '.join(available) or '(none registered)'}"
            )
        return cls(**config)

    # -- introspection ------------------------------------------------------

    def available_brokers(self) -> List[str]:
        """Return sorted list of registered broker names."""
        return sorted(set(self._local) | set(_BROKER_REGISTRY))
