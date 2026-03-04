"""DTOs for order entry: request and result. All flows use these."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from src.core.events import Exchange, OrderType, Signal, SignalSide


class RejectReason(str, Enum):
    VALIDATION = "validation"
    IDEMPOTENCY_DUPLICATE = "idempotency_duplicate"
    IDEMPOTENCY_UNAVAILABLE = "idempotency_unavailable"
    KILL_SWITCH = "kill_switch"
    CIRCUIT_BREAKER = "circuit_breaker"
    RISK_REJECTED = "risk_rejected"
    RESERVATION_FAILED = "reservation_failed"
    BROKER_ERROR = "broker_error"
    TIMEOUT = "timeout"


@dataclass
class OrderEntryRequest:
    """Single request type for all order flows (AI, API, internal)."""
    signal: Signal
    quantity: int
    order_type: OrderType = OrderType.LIMIT
    limit_price: Optional[float] = None
    idempotency_key: Optional[str] = None
    source: str = "api"  # api | ai_engine | manual | test
    metadata: dict = field(default_factory=dict)
    force_reduce: bool = False  # True for emergency close-outs: bypass daily loss, circuit breaker, rate limiter

    def validate(self) -> tuple[bool, str]:
        """Validate input. Returns (ok, error_message)."""
        if self.quantity <= 0:
            return False, "quantity must be positive"
        if self.order_type == OrderType.LIMIT and (self.limit_price is None or self.limit_price <= 0):
            if self.signal.price is None or self.signal.price <= 0:
                return False, "limit_price required for LIMIT order"
        if not self.signal.symbol or not self.signal.symbol.strip():
            return False, "symbol required"
        if self.signal.side not in (SignalSide.BUY, SignalSide.SELL):
            return False, "side must be BUY or SELL"
        return True, ""


@dataclass
class OrderEntryResult:
    """Result of submit_order: either order_id or reject reason."""
    success: bool
    order_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    reject_reason: Optional[RejectReason] = None
    reject_detail: str = ""
    latency_ms: Optional[float] = None
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
