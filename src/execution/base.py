"""Execution gateway contract: place/cancel orders, fetch positions/order status."""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from src.core.events import Order, OrderStatus, Position

logger = logging.getLogger(__name__)


class BaseExecutionGateway(ABC):
    """Abstract broker gateway: REST/WebSocket or FIX."""

    @abstractmethod
    async def connect(self) -> None:
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        ...

    @abstractmethod
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
        """Place order; return Order with broker_order_id and status."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str, broker_order_id: Optional[str] = None) -> bool:
        ...

    @abstractmethod
    async def get_order_status(self, order_id: str, broker_order_id: Optional[str] = None) -> OrderStatus:
        ...

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        ...

    @abstractmethod
    async def get_orders(self, status: Optional[str] = None, limit: int = 100) -> List[Order]:
        ...

    async def health_check(self) -> Dict[str, object]:
        """Verify gateway connectivity and return health status.

        Returns a dict with at minimum:
            {"healthy": bool, "broker": str, "detail": str}

        Subclasses should override to perform a real connectivity check
        (e.g. fetch profile or order book). The default implementation
        returns a basic status based on whether connect() has been called.
        """
        return {
            "healthy": False,
            "broker": self.__class__.__name__,
            "detail": "health_check not implemented by subclass",
        }
