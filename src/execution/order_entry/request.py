"""DTOs for order entry: request and result. All flows use these."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

from src.core.events import OrderType, Signal, SignalSide


class RejectReason(str, Enum):
    VALIDATION = "validation"
    LOT_SIZE_INVALID = "lot_size_invalid"
    CIRCUIT_LIMIT = "circuit_limit"
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
    limit_price: float | None = None
    idempotency_key: str | None = None
    source: str = "api"  # api | ai_engine | manual | test
    metadata: dict = field(default_factory=dict)
    force_reduce: bool = False  # True for emergency close-outs: bypass daily loss, circuit breaker, rate limiter

    def validate(self) -> tuple[bool, str]:
        """Validate input. Returns (ok, error_message)."""
        if not isinstance(self.quantity, (int, float)) or self.quantity <= 0:
            return False, "quantity must be a positive number"
        if self.quantity != int(self.quantity):
            return False, "quantity must be a whole number"
        if self.order_type == OrderType.LIMIT and (self.limit_price is None or self.limit_price <= 0):
            if self.signal.price is None or self.signal.price <= 0:
                return False, "limit_price required for LIMIT order"
        # Validate limit_price is not negative (explicit check for all order types)
        if self.limit_price is not None and self.limit_price < 0:
            return False, "limit_price must not be negative"
        if not self.signal.symbol or not self.signal.symbol.strip():
            return False, "symbol required"
        # Symbol sanity: reject obviously invalid symbols (empty after strip, too long, control chars)
        symbol = self.signal.symbol.strip()
        if len(symbol) > 50:
            return False, f"symbol too long ({len(symbol)} chars, max 50)"
        if any(c in symbol for c in ("\n", "\r", "\t", "\0")):
            return False, "symbol contains invalid characters"
        if self.signal.side not in (SignalSide.BUY, SignalSide.SELL):
            return False, "side must be BUY or SELL"
        return True, ""


@dataclass
class OrderEntryResult:
    """Result of submit_order: either order_id or reject reason."""

    success: bool
    order_id: str | None = None
    broker_order_id: str | None = None
    reject_reason: RejectReason | None = None
    reject_detail: str = ""
    latency_ms: float | None = None
    market_impact_bps: float | None = None
    circuit_check_passed: bool | None = None
    lot_size_adjusted: bool | None = None
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))
