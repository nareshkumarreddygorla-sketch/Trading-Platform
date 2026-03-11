"""Order and OrderEvent repository. Sync API; use from async via executor."""

import logging
from datetime import UTC, datetime

from sqlalchemy import desc
from sqlalchemy.orm import Session

from src.core.events import Exchange, Order, OrderStatus, OrderType, SignalSide
from src.execution.lifecycle_transitions import is_allowed_transition

from .database import session_scope
from .models import OrderEventModel, OrderModel

logger = logging.getLogger(__name__)


def _status_to_db(status: OrderStatus) -> str:
    m = {
        OrderStatus.PENDING: "NEW",
        OrderStatus.LIVE: "ACK",
        OrderStatus.REJECTED: "REJECTED",
        OrderStatus.FILLED: "FILLED",
        OrderStatus.PARTIALLY_FILLED: "PARTIAL",
        OrderStatus.CANCELLED: "CANCELLED",
    }
    return m.get(status, status.value if status else "NEW")


def _db_to_status(s: str) -> OrderStatus:
    m = {
        "SUBMITTING": OrderStatus.PENDING,
        "NEW": OrderStatus.PENDING,
        "ACK": OrderStatus.LIVE,
        "REJECTED": OrderStatus.REJECTED,
        "FILLED": OrderStatus.FILLED,
        "PARTIAL": OrderStatus.PARTIALLY_FILLED,
        "CANCELLED": OrderStatus.CANCELLED,
    }
    return m.get(s, OrderStatus.PENDING)


def _order_model_to_domain(m: OrderModel) -> Order:
    return Order(
        order_id=m.order_id,
        strategy_id=m.strategy_id or "",
        symbol=m.symbol,
        exchange=Exchange(m.exchange) if m.exchange else Exchange.NSE,
        side=SignalSide(m.side) if m.side else SignalSide.BUY,
        quantity=m.quantity,
        order_type=OrderType(m.order_type) if m.order_type else OrderType.LIMIT,
        limit_price=m.limit_price,
        status=_db_to_status(m.status),
        filled_qty=m.filled_qty or 0.0,
        avg_price=m.avg_price,
        ts=m.created_at or datetime.now(UTC),
        broker_order_id=m.broker_order_id,
        metadata={},
    )


class OrderRepository:
    """Sync repository for orders and order events."""

    def __init__(self, session_factory=None):
        from .database import get_session_factory

        self._session_factory = session_factory or get_session_factory()

    def create_order(
        self,
        order: Order,
        idempotency_key: str | None = None,
        initial_status: str = "NEW",
    ) -> None:
        """Insert Order. initial_status NEW or SUBMITTING (write-ahead). Fails if order_id already exists."""
        if not (order.order_id and str(order.order_id).strip()):
            logger.warning("create_order skipped: order_id is empty")
            return
        if initial_status not in ("NEW", "SUBMITTING"):
            initial_status = "NEW"
        with session_scope() as session:
            existing = session.query(OrderModel).filter(OrderModel.order_id == order.order_id).first()
            if existing:
                return
            m = OrderModel(
                order_id=order.order_id or "",
                broker_order_id=order.broker_order_id,
                idempotency_key=idempotency_key,
                strategy_id=order.strategy_id or "",
                symbol=order.symbol,
                exchange=order.exchange.value if order.exchange else "NSE",
                side=order.side.value if order.side else "BUY",
                quantity=order.quantity,
                order_type=order.order_type.value if order.order_type else "LIMIT",
                limit_price=order.limit_price,
                status=initial_status,
                filled_qty=order.filled_qty or 0.0,
                avg_price=order.avg_price,
            )
            session.add(m)
            session.flush()
            ev = OrderEventModel(
                order_id_ref=m.order_id,
                event_type=initial_status,
                from_status=None,
                to_status=initial_status,
                filled_qty=m.filled_qty,
                avg_price=m.avg_price,
            )
            session.add(ev)

    def update_order_after_broker_ack(
        self,
        order_id: str,
        broker_order_id: str | None,
        new_status: str = "NEW",
        session: Session | None = None,
    ) -> bool:
        """Transition SUBMITTING -> NEW after broker accepts. Sets broker_order_id."""

        def _up(sess: Session) -> bool:
            m = sess.query(OrderModel).filter(OrderModel.order_id == order_id).first()
            if not m or m.status != "SUBMITTING":
                return False
            if not is_allowed_transition("SUBMITTING", new_status):
                return False
            m.status = new_status
            m.broker_order_id = broker_order_id
            m.updated_at = datetime.now(UTC)
            ev = OrderEventModel(
                order_id_ref=m.order_id,
                event_type=new_status,
                from_status="SUBMITTING",
                to_status=new_status,
                filled_qty=m.filled_qty,
                avg_price=m.avg_price,
            )
            sess.add(ev)
            return True

        if session is not None:
            return _up(session)
        with session_scope() as sess:
            return _up(sess)

    def get_by_order_id(self, order_id: str, session: Session | None = None) -> Order | None:
        """Return domain Order by order_id. If session given, use it; else new scope."""

        def _get(sess: Session) -> Order | None:
            m = sess.query(OrderModel).filter(OrderModel.order_id == order_id).first()
            return _order_model_to_domain(m) if m else None

        if session is not None:
            return _get(session)
        with session_scope() as sess:
            return _get(sess)

    def list_active_orders(self) -> list[Order]:
        """Return all orders with status SUBMITTING, NEW, ACK, or PARTIAL. For cold start recovery and write-ahead reconcile."""
        with session_scope() as session:
            rows = (
                session.query(OrderModel)
                .filter(OrderModel.status.in_(["SUBMITTING", "NEW", "ACK", "PARTIAL"]))
                .order_by(desc(OrderModel.created_at))
                .all()
            )
            return [_order_model_to_domain(r) for r in rows]

    def list_submitting_orders(self) -> list[Order]:
        """Return orders with status SUBMITTING (write-ahead; need broker reconcile)."""
        with session_scope() as session:
            rows = (
                session.query(OrderModel)
                .filter(OrderModel.status == "SUBMITTING")
                .order_by(OrderModel.created_at)
                .all()
            )
            return [_order_model_to_domain(r) for r in rows]

    def list_orders_paginated(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        strategy_id: str | None = None,
    ) -> tuple[list[Order], int]:
        """Return (list of Order, total count). Status filter uses DB status (e.g. NEW, FILLED)."""
        with session_scope() as session:
            q = session.query(OrderModel)
            if status:
                q = q.filter(OrderModel.status == status)
            if strategy_id:
                q = q.filter(OrderModel.strategy_id == strategy_id)
            total = q.count()
            rows = q.order_by(desc(OrderModel.created_at)).offset(offset).limit(limit).all()
            return [_order_model_to_domain(r) for r in rows], total

    def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        filled_qty: float = 0.0,
        avg_price: float | None = None,
        session: Session | None = None,
    ) -> bool:
        """Update order status and append OrderEvent. Returns True if order was found and updated."""
        db_status = _status_to_db(status)

        def _update(sess: Session) -> bool:
            m = sess.query(OrderModel).filter(OrderModel.order_id == order_id).first()
            if not m:
                return False
            from_status = m.status
            if not is_allowed_transition(from_status, db_status):
                logger.warning(
                    "Order %s: illegal status transition %s -> %s; rejecting update", order_id, from_status, db_status
                )
                return False
            m.status = db_status
            m.filled_qty = filled_qty
            if avg_price is not None:
                m.avg_price = avg_price
            m.updated_at = datetime.now(UTC)
            ev = OrderEventModel(
                order_id_ref=m.order_id,
                event_type=db_status,
                from_status=from_status,
                to_status=db_status,
                filled_qty=filled_qty,
                avg_price=avg_price,
            )
            sess.add(ev)
            return True

        if session is not None:
            return _update(session)
        with session_scope() as sess:
            return _update(sess)
