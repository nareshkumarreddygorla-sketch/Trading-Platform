"""Strategy plugin contract: scored signals, portfolio weight, risk level."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.core.events import Bar, Exchange, Signal


@dataclass
class MarketState:
    """Current market context passed to strategies."""

    symbol: str
    exchange: Exchange
    bars: list[Bar]  # recent OHLCV (e.g. last 100 bars)
    latest_price: float
    volume: float
    order_book: Any = None  # optional L2 snapshot
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class StrategyBase(ABC):
    """Plugin base: each strategy returns scored signals with weight and risk_level."""

    strategy_id: str
    description: str = ""

    @abstractmethod
    def generate_signals(self, state: MarketState) -> list[Signal]:
        """
        Return list of signals. Each signal must have:
        - strategy_id, symbol, side, score (0-1), portfolio_weight (0-1), risk_level.
        """
        ...

    def warm(self, state: MarketState) -> bool:
        """Return True if enough data to generate (e.g. min bars)."""
        return len(state.bars) >= 20
