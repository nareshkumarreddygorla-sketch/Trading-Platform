"""
Domain events and DTOs for market data, signals, and orders.
All timestamps are UTC. Used across market_data, strategy_engine, risk_engine, execution.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Set

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
    SUBMITTING = "SUBMITTING"      # Write-ahead: order persisted, awaiting broker ACK
    LIVE = "LIVE"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    TIMEOUT_UNCERTAIN = "TIMEOUT_UNCERTAIN"  # Broker timeout: order may or may not have been submitted
    EXPIRED = "EXPIRED"


class Order(BaseModel):
    """Order (request or from broker)."""
    order_id: Optional[str] = None
    strategy_id: str
    symbol: str
    exchange: Exchange
    side: SignalSide
    quantity: float = Field(gt=0)
    order_type: OrderType = OrderType.LIMIT
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_price: Optional[float] = None
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    broker_order_id: Optional[str] = None
    updated_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Execution quality fields (populated during order entry pipeline)
    lot_size: Optional[int] = None                # NSE lot size used for validation
    lot_size_adjusted: Optional[bool] = None      # True if quantity was rounded to lot size
    market_impact_bps: Optional[float] = None     # Estimated market impact in basis points
    circuit_limit_check: Optional[str] = None     # "PASSED", "SKIPPED", or rejection reason
    arrival_price: Optional[float] = None         # Price at decision time (for shortfall calc)
    segment: Optional[str] = None                 # Market segment: EQ, FO, IDX


# Valid order status transitions
VALID_ORDER_TRANSITIONS: Dict[OrderStatus, Set[OrderStatus]] = {
    OrderStatus.PENDING: {OrderStatus.SUBMITTING, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED},
    OrderStatus.SUBMITTING: {OrderStatus.LIVE, OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.TIMEOUT_UNCERTAIN, OrderStatus.CANCELLED},
    OrderStatus.LIVE: {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED},
    OrderStatus.PARTIALLY_FILLED: {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED},
    OrderStatus.TIMEOUT_UNCERTAIN: {OrderStatus.LIVE, OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED},
    OrderStatus.FILLED: set(),  # Terminal
    OrderStatus.CANCELLED: set(),  # Terminal
    OrderStatus.REJECTED: set(),  # Terminal
    OrderStatus.EXPIRED: set(),  # Terminal
}


def validate_order_transition(from_status: OrderStatus, to_status: OrderStatus) -> bool:
    """Check if a status transition is valid."""
    valid = VALID_ORDER_TRANSITIONS.get(from_status, set())
    allowed = to_status in valid
    if not allowed:
        import logging
        logging.getLogger(__name__).warning(
            "Invalid order transition: %s -> %s (allowed: %s)",
            from_status.value, to_status.value, {s.value for s in valid},
        )
    return allowed


class Position(BaseModel):
    """Open position."""
    symbol: str
    exchange: Exchange
    side: SignalSide
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0
    strategy_id: Optional[str] = None
