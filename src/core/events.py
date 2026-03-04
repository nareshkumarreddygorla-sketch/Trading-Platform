"""
Domain events and DTOs for market data, signals, and orders.
All timestamps are UTC. Used across market_data, strategy_engine, risk_engine, execution.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Exchange(str, Enum):
    NSE = "NSE"
    BSE = "BSE"
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    LSE = "LSE"
    FX = "FX"


class Bar(BaseModel):
    """OHLCV bar (normalized)."""
    symbol: str
    exchange: Exchange
    interval: str  # 1m, 5m, 1h, 1d
    open: float
    high: float
    low: float
    close: float
    volume: float
    ts: datetime
    source: str = ""


class Tick(BaseModel):
    """Single tick (last trade)."""
    symbol: str
    exchange: Exchange
    price: float
    size: float
    ts: datetime
    side: Optional[str] = None


class OrderBookSnapshot(BaseModel):
    """Order book snapshot (best bid/ask and optional depth)."""
    symbol: str
    exchange: Exchange
    ts: datetime
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    levels: list[dict[str, Any]] = Field(default_factory=list)


# --- Strategy & risk ---

class SignalSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Signal(BaseModel):
    """Scored signal from strategy engine."""
    strategy_id: str
    symbol: str
    exchange: Exchange
    side: SignalSide
    score: float = Field(ge=0.0, le=1.0)
    portfolio_weight: float = Field(ge=0.0, le=1.0)
    risk_level: str = "NORMAL"  # LOW, NORMAL, HIGH
    reason: str = ""
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    IOC = "IOC"
    FOK = "FOK"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    LIVE = "LIVE"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Order(BaseModel):
    """Order (request or from broker)."""
    order_id: Optional[str] = None
    strategy_id: str
    symbol: str
    exchange: Exchange
    side: SignalSide
    quantity: float
    order_type: OrderType = OrderType.LIMIT
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_price: Optional[float] = None
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    broker_order_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Position(BaseModel):
    """Open position."""
    symbol: str
    exchange: Exchange
    side: SignalSide
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0
    strategy_id: Optional[str] = None
