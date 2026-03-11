"""Abstract base gateway for all broker integrations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BrokerType(str, Enum):
    ANGEL_ONE = "angel_one"
    ZERODHA = "zerodha"
    IBKR = "ibkr"
    ALPACA = "alpaca"
    FIVEPAISA = "fivepaisa"
    UPSTOX = "upstox"
    PAPER = "paper"


class GatewayStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


@dataclass
class BrokerHealth:
    broker: BrokerType
    status: GatewayStatus
    latency_ms: float = 0.0
    last_heartbeat: str | None = None
    error_count: int = 0
    orders_today: int = 0
    rate_limit_remaining: int | None = None
    supported_exchanges: list[str] = field(default_factory=list)
    supported_order_types: list[str] = field(default_factory=list)
    paper_mode: bool = True


@dataclass
class BrokerOrder:
    broker_order_id: str | None
    symbol: str
    exchange: str
    side: str
    quantity: int
    order_type: str
    limit_price: float | None
    status: str
    filled_qty: int = 0
    avg_price: float = 0.0
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrokerPosition:
    symbol: str
    exchange: str
    quantity: int
    avg_price: float
    current_price: float = 0.0
    pnl: float = 0.0
    side: str = "BUY"


class BaseBrokerGateway(ABC):
    """Abstract base class for all broker gateways."""

    broker_type: BrokerType = BrokerType.PAPER
    paper: bool = True

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to broker. Returns True on success."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker."""
        ...

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        exchange: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: float | None = None,
        **kwargs,
    ) -> BrokerOrder:
        """Place an order. Returns BrokerOrder with broker_order_id."""
        ...

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an order. Returns True on success."""
        ...

    @abstractmethod
    async def get_order_status(self, broker_order_id: str) -> BrokerOrder:
        """Get current order status."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[BrokerPosition]:
        """Get all open positions."""
        ...

    @abstractmethod
    def health(self) -> BrokerHealth:
        """Get current gateway health status."""
        ...

    @abstractmethod
    def supported_exchanges(self) -> list[str]:
        """Return list of supported exchange codes."""
        ...
