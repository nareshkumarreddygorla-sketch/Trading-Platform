"""Execution gateway contract: place/cancel orders, fetch positions/order status."""
from abc import ABC, abstractmethod
from typing import List, Optional

from src.core.events import Order, OrderStatus, Position


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
