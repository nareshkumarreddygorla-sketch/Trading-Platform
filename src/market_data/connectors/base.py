"""Base contract for exchange/market data connectors. Resilient: auto-reconnect, backpressure."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from src.core.events import Bar, Exchange, Tick


class BaseMarketDataConnector(ABC):
    """Abstract connector: stream ticks and/or bars; normalize to core events."""

    exchange: Exchange

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection (WebSocket/REST). Retry with backoff on failure."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Graceful disconnect."""
        ...

    @abstractmethod
    async def subscribe_ticks(self, symbols: list[str]) -> None:
        """Subscribe to tick stream for given symbols."""
        ...

    @abstractmethod
    async def subscribe_bars(self, symbols: list[str], interval: str) -> None:
        """Subscribe to bar stream (e.g. 1m, 5m)."""
        ...

    @abstractmethod
    def stream_ticks(self) -> AsyncIterator[Tick]:
        """Async generator of normalized ticks. Handle backpressure internally."""
        ...

    @abstractmethod
    def stream_bars(self) -> AsyncIterator[Bar]:
        """Async generator of normalized bars."""
        ...

    async def get_historical_bars(self, symbol: str, interval: str, start: str, end: str) -> list[Bar]:
        """Fetch historical bars (REST). Return empty list if not supported."""
        return []
