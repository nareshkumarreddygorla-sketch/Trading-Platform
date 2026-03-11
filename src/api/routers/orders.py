import asyncio
import math
import re
from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from src.core.events import Exchange, Signal, SignalSide
from src.api.deps import get_order_entry_service, get_kill_switch
from src.api.auth import get_current_user, require_roles
from src.execution.order_entry import OrderEntryRequest, RejectReason
from src.core.events import OrderType

router = APIRouter()

_SYMBOL_RE = re.compile(r"^[A-Z0-9&_-]{1,20}$")


def _validate_symbol(symbol: str) -> str:
    """Validate and sanitize a stock symbol. Raises HTTP 400 on invalid input."""
    symbol = symbol.strip().upper()
    if not _SYMBOL_RE.match(symbol):
        raise HTTPException(400, f"Invalid symbol: must match {_SYMBOL_RE.pattern}")
    return symbol

MAX_IDEMPOTENCY_KEY_LEN = 256
MAX_KILL_SWITCH_REASON_LEN = 64
MAX_KILL_SWITCH_DETAIL_LEN = 512
MAX_ORDER_LIMIT = 1000
MAX_SYMBOL_LEN = 32
MAX_STRATEGY_ID_LEN = 128
MAX_ORDER_QUANTITY = 1_000_000

# DB status <-> API status mapping for filter
_API_TO_DB_STATUS = {"PENDING": "NEW", "LIVE": "ACK", "PARTIALLY_FILLED": "PARTIAL"}


class OrderSideEnum(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderTypeEnum(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    IOC = "IOC"
    FOK = "FOK"


class ExchangeEnum(str, Enum):
    NSE = "NSE"
    BSE = "BSE"
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    LSE = "LSE"
    FX = "FX"


class PlaceOrderRequest(BaseModel):
    symbol: str = Field(..., max_length=MAX_SYMBOL_LEN, pattern=r"^[A-Za-z0-9._&-]{1,32}$")
    exchange: ExchangeEnum = ExchangeEnum.NSE
    side: OrderSideEnum
    quantity: float = Field(..., gt=0, le=MAX_ORDER_QUANTITY)
    order_type: OrderTypeEnum = OrderTypeEnum.LIMIT
    limit_price: Optional[float] = Field(None, gt=0)
    strategy_id: str = Field("", max_length=MAX_STRATEGY_ID_LEN)
    idempotency_key: Optional[str] = Field(None, max_length=MAX_IDEMPOTENCY_KEY_LEN)

    @field_validator("quantity")
    @classmethod
    def quantity_must_be_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("quantity must be finite")
        return v

    @field_validator("limit_price")
    @classmethod
    def limit_price_must_be_finite(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not math.isfinite(v):
            raise ValueError("limit_price must be finite")
        return v


class OrderResponse(BaseModel):
    order_id: Optional[str]
    broker_order_id: Optional[str]
    strategy_id: str
    symbol: str
    exchange: str
    side: str
    quantity: float
    order_type: str
    limit_price: Optional[float]
    status: str
    filled_qty: float
    avg_price: Optional[float]
    ts: Optional[str]


class OrderListResponse(BaseModel):
    orders: List[OrderResponse]
    total: int


class PositionResponse(BaseModel):
    symbol: str
    exchange: str
    side: str
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0
    strategy_id: Optional[str] = None


class PositionsResponse(BaseModel):
    positions: List[PositionResponse]


class CancelOrderResponse(BaseModel):
    order_id: str
    status: str


class PlaceOrderResponse(BaseModel):
    order_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    status: str
    latency_ms: Optional[float] = None


class KillSwitchResponse(BaseModel):
    status: str
    reason: Optional[str] = None
    message: Optional[str] = None


def _order_to_response(o) -> OrderResponse:
    return OrderResponse(
        order_id=getattr(o, "order_id", None),
        broker_order_id=getattr(o, "broker_order_id", None),
        strategy_id=getattr(o, "strategy_id", ""),
        symbol=getattr(o, "symbol", ""),
        exchange=getattr(getattr(o, "exchange", None), "value", "NSE"),
        side=getattr(getattr(o, "side", None), "value", ""),
        quantity=getattr(o, "quantity", 0),
        order_type=getattr(getattr(o, "order_type", None), "value", "LIMIT"),
        limit_price=getattr(o, "limit_price", None),
        status=getattr(getattr(o, "status", None), "value", "PENDING"),
        filled_qty=getattr(o, "filled_qty", 0),
        avg_price=getattr(o, "avg_price", None),
        ts=(lambda t: t.isoformat() if t and hasattr(t, "isoformat") else None)(getattr(o, "ts", None)),
    )


@router.get("/orders", response_model=OrderListResponse)
async def list_orders(
    request: Request,
    status: Optional[str] = None,
    strategy_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
):
    limit = min(max(1, limit), MAX_ORDER_LIMIT)
    offset = max(0, offset)
    persistence = getattr(request.app.state, "persistence_service", None)
    if persistence is not None:
        db_status = _API_TO_DB_STATUS.get(status, status) if status else None
        loop = asyncio.get_running_loop()
        orders, total = await loop.run_in_executor(
            None,
            lambda: persistence.list_orders_paginated_sync(limit=limit, offset=offset, status=db_status, strategy_id=strategy_id),
        )
        return OrderListResponse(orders=[_order_to_response(o) for o in orders], total=total)
    order_entry = get_order_entry_service(request)
    if order_entry is None:
        return OrderListResponse(orders=[], total=0)
    orders = order_entry.lifecycle.list_recent(limit=limit)
    out = [_order_to_response(o) for o in orders]
    if strategy_id:
        out = [o for o in out if o.strategy_id == strategy_id]
    if status:
        out = [o for o in out if o.status == status]
    return OrderListResponse(orders=out, total=len(out))


@router.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(request: Request, order_id: str, current_user: dict = Depends(get_current_user)):
    persistence = getattr(request.app.state, "persistence_service", None)
    if persistence is not None:
        loop = asyncio.get_running_loop()
        order = await loop.run_in_executor(None, lambda: persistence.get_order_sync(order_id))
        if order is None:
            raise HTTPException(404, "Order not found")
        return _order_to_response(order)
    order_entry = get_order_entry_service(request)
    if order_entry is None:
        raise HTTPException(404, "Order not found")
    order = await order_entry.lifecycle.get_order(order_id)
    if order is None:
        raise HTTPException(404, "Order not found")
    return _order_to_response(order)


@router.post("/orders/{order_id}/cancel", response_model=CancelOrderResponse)
async def cancel_order(
    request: Request,
    order_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Cancel an open order. Uses broker_order_id if available."""
    persistence = getattr(request.app.state, "persistence_service", None)
    order_entry = get_order_entry_service(request)
    gateway = getattr(request.app.state, "gateway", None)
    if gateway is None:
        raise HTTPException(503, "Gateway not configured")
    order = None
    if persistence is not None:
        loop = asyncio.get_running_loop()
        order = await loop.run_in_executor(None, lambda: persistence.get_order_sync(order_id))
    if order is None and order_entry is not None:
        order = await order_entry.lifecycle.get_order(order_id)
    if order is None:
        raise HTTPException(404, "Order not found")
    status_val = getattr(getattr(order, "status", None), "value", str(getattr(order, "status", "")))
    if status_val in ("FILLED", "CANCELLED", "REJECTED", "PARTIAL", "PARTIALLY_FILLED"):
        raise HTTPException(400, f"Order already {status_val}")
    broker_order_id = getattr(order, "broker_order_id", None)
    try:
        ok = await gateway.cancel_order(order_id, broker_order_id)
    except Exception as e:
        import logging as _logging
        _logging.getLogger(__name__).exception("Order cancellation failed")
        raise HTTPException(502, "Order cancellation failed")
    if not ok:
        raise HTTPException(502, "Cancel rejected by broker")

    # Audit log cancel action
    audit_repo = getattr(request.app.state, "audit_repo", None)
    if audit_repo:
        try:
            actor = current_user.get("user_id", "api") if current_user else "api"
            audit_repo.append_sync("order_cancel", actor, {"order_id": order_id, "broker_order_id": broker_order_id})
        except Exception:
            pass

    return {"order_id": order_id, "status": "CANCELLED"}


@router.get("/positions", response_model=PositionsResponse)
async def list_positions(request: Request, current_user: dict = Depends(get_current_user)):
    persistence = getattr(request.app.state, "persistence_service", None)
    if persistence is not None:
        loop = asyncio.get_running_loop()
        positions = await loop.run_in_executor(None, persistence.list_positions_sync)
        return PositionsResponse(
            positions=[
                PositionResponse(
                    symbol=p.symbol,
                    exchange=p.exchange.value if hasattr(p.exchange, "value") else str(p.exchange),
                    side=p.side.value if hasattr(p.side, "value") else str(p.side),
                    quantity=p.quantity,
                    avg_price=p.avg_price,
                    unrealized_pnl=getattr(p, "unrealized_pnl", 0.0),
                    strategy_id=getattr(p, "strategy_id", None),
                )
                for p in positions
            ]
        )
    risk_manager = getattr(request.app.state, "risk_manager", None)
    if risk_manager is None:
        return PositionsResponse(positions=[])
    positions = getattr(risk_manager, "positions", [])
    return PositionsResponse(
        positions=[
            PositionResponse(
                symbol=getattr(p, "symbol", ""),
                exchange=getattr(getattr(p, "exchange", None), "value", "NSE"),
                side=getattr(getattr(p, "side", None), "value", ""),
                quantity=getattr(p, "quantity", 0),
                avg_price=getattr(p, "avg_price", 0),
                unrealized_pnl=getattr(p, "unrealized_pnl", 0),
                strategy_id=getattr(p, "strategy_id", None),
            )
            for p in positions
        ]
    )


@router.post("/orders", response_model=PlaceOrderResponse)
async def place_order(
    request: Request,
    body: PlaceOrderRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Single order entry: ALL orders go through OrderEntryService (validate -> idempotency -> kill -> circuit -> risk -> reserve -> router -> lifecycle).
    Pydantic validates: side (BUY/SELL enum), quantity (>0, <=1M, finite), exchange (enum), order_type (enum), symbol (pattern).
    """
    # Validate and sanitize the symbol beyond Pydantic's pattern check
    body.symbol = _validate_symbol(body.symbol)
    side_upper = body.side.value
    qty_int = int(round(body.quantity))
    if qty_int <= 0:
        raise HTTPException(400, "quantity must be at least 1")
    if body.order_type == OrderTypeEnum.LIMIT:
        if body.limit_price is None:
            raise HTTPException(400, "limit_price is required for LIMIT orders")

    order_entry = get_order_entry_service(request)
    if order_entry is None:
        raise HTTPException(503, "Order entry not configured: set app.state.order_entry_service")

    # When live market data + live execution: require CapitalGate.validate() before allowing execution
    gateway = getattr(getattr(order_entry, "order_router", None), "default_gateway", None)
    if (
        getattr(request.app.state, "angel_one_marketdata_enabled", False)
        and gateway is not None
        and not getattr(gateway, "paper", True)
    ):
        capital_gate = getattr(request.app.state, "capital_gate", None)
        if capital_gate:
            validation = await capital_gate.validate()
            if not validation.get("ok"):
                raise HTTPException(
                    503,
                    validation.get("message", "Capital gate not passed; do not enable autonomous live mode"),
                )

    exch = Exchange(body.exchange.value)
    signal = Signal(
        strategy_id=body.strategy_id or "api",
        symbol=body.symbol,
        exchange=exch,
        side=SignalSide(side_upper),
        score=0.5,
        portfolio_weight=0.0,
        risk_level="NORMAL",
        reason="api",
        price=body.limit_price,
    )
    entry_request = OrderEntryRequest(
        signal=signal,
        quantity=qty_int,
        order_type=OrderType(body.order_type.value),
        limit_price=body.limit_price,
        idempotency_key=body.idempotency_key,
        source="api",
    )
    result = await order_entry.submit_order(entry_request)

    audit_repo = getattr(request.app.state, "audit_repo", None)
    actor = current_user.get("user_id", "api") if current_user else "api"
    if audit_repo:
        try:
            if result.success:
                audit_repo.append_sync(
                    "order_submit_success",
                    actor,
                    {"order_id": result.order_id, "broker_order_id": result.broker_order_id, "symbol": body.symbol, "side": body.side.value},
                )
            else:
                audit_repo.append_sync(
                    "order_submit_reject",
                    actor,
                    {"reason": result.reject_reason.value if result.reject_reason else "", "detail": result.reject_detail or ""},
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception("Audit append failed: %s", e)

    if not result.success:
        reason = result.reject_reason
        detail = result.reject_detail or ""
        if reason == RejectReason.KILL_SWITCH:
            raise HTTPException(503, f"Kill switch active: {detail}")
        if reason == RejectReason.CIRCUIT_BREAKER:
            raise HTTPException(503, "Circuit breaker open")
        if reason == RejectReason.IDEMPOTENCY_UNAVAILABLE:
            raise HTTPException(503, "idempotency_unavailable")
        if reason == RejectReason.TIMEOUT:
            raise HTTPException(503, "broker_timeout")
        if reason == RejectReason.RISK_REJECTED:
            raise HTTPException(403, f"Risk rejected: {detail}")
        if reason == RejectReason.VALIDATION:
            raise HTTPException(400, detail)
        raise HTTPException(503, f"Order rejected: {reason.value if reason else 'unknown'}: {detail}")

    return {"order_id": result.order_id, "broker_order_id": result.broker_order_id, "status": "PENDING", "latency_ms": result.latency_ms}


@router.post("/admin/kill_switch/arm", response_model=KillSwitchResponse)
async def kill_switch_arm(
    request: Request,
    reason: str = "manual",
    detail: str = "",
    current_user: dict = Depends(require_roles(["admin"])),
):
    """Arm global kill switch (admin). Prevents new orders; reduce-only allowed if configured."""
    kill_switch = get_kill_switch(request)
    if kill_switch is None:
        raise HTTPException(503, "Kill switch not configured")
    reason = (reason or "manual")[:MAX_KILL_SWITCH_REASON_LEN]
    detail = (detail or "")[:MAX_KILL_SWITCH_DETAIL_LEN]
    from src.execution.order_entry.kill_switch import KillReason
    try:
        r = KillReason(reason)
    except ValueError:
        r = KillReason.MANUAL
    await kill_switch.arm(r, detail)
    audit_repo = getattr(request.app.state, "audit_repo", None)
    if audit_repo:
        try:
            audit_repo.append_sync("kill_switch_arm", current_user.get("user_id", "admin"), {"reason": reason, "detail": detail})
        except Exception:
            pass
    return {"status": "armed", "reason": reason}


@router.post("/admin/kill_switch/disarm", response_model=KillSwitchResponse)
async def kill_switch_disarm(
    request: Request,
    current_user: dict = Depends(require_roles(["admin"])),
):
    kill_switch = get_kill_switch(request)
    if kill_switch is None:
        raise HTTPException(503, "Kill switch not configured")
    await kill_switch.disarm()
    audit_repo = getattr(request.app.state, "audit_repo", None)
    if audit_repo:
        try:
            audit_repo.append_sync("kill_switch_disarm", current_user.get("user_id", "admin"), {})
        except Exception:
            pass
    return {"status": "disarmed"}


@router.post("/admin/safe_mode/clear", response_model=KillSwitchResponse)
async def safe_mode_clear(
    request: Request,
    current_user: dict = Depends(require_roles(["admin"])),
):
    """Clear safe_mode so trading can be re-enabled after broker is back. Audit logged."""
    import logging
    logger = logging.getLogger(__name__)
    if not getattr(request.app.state, "safe_mode", False):
        return {"status": "ok", "message": "safe_mode was not set"}
    request.app.state.safe_mode = False
    logger.warning("safe_mode cleared via POST /admin/safe_mode/clear (trading re-enabled)")
    audit_repo = getattr(request.app.state, "audit_repo", None)
    if audit_repo:
        try:
            audit_repo.append_sync("safe_mode_clear", current_user.get("user_id", "admin"), {})
        except Exception:
            pass
    return {"status": "ok", "message": "safe_mode cleared; trading re-enabled"}
