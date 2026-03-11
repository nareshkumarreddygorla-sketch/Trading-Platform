"""SQLAlchemy models for Order, OrderEvent, Position. Institutional: constraints, version (OCC)."""
from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# Valid DB order statuses (enum constraint). SUBMITTING = write-ahead (before broker ack).
ORDER_STATUS_VALUES = ("SUBMITTING", "NEW", "ACK", "PARTIAL", "FILLED", "REJECTED", "CANCELLED")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class OrderModel(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(64), unique=True, nullable=False, index=True)
    broker_order_id = Column(String(128), nullable=True, index=True)
    idempotency_key = Column(String(256), nullable=True, index=True)
    strategy_id = Column(String(128), nullable=False, default="")
    symbol = Column(String(32), nullable=False, index=True)
    exchange = Column(String(16), nullable=False, default="NSE")
    side = Column(String(8), nullable=False)
    quantity = Column(Float, nullable=False)
    order_type = Column(String(16), nullable=False, default="LIMIT")
    limit_price = Column(Float, nullable=True)
    status = Column(String(32), nullable=False, default="NEW", index=True)
    filled_qty = Column(Float, nullable=False, default=0.0)
    avg_price = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now)

    events = relationship("OrderEventModel", back_populates="order", order_by="OrderEventModel.created_at")

    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_orders_idempotency_key"),
        CheckConstraint("quantity >= 0", name="ck_orders_quantity_non_negative"),
        CheckConstraint("filled_qty >= 0", name="ck_orders_filled_qty_non_negative"),
        CheckConstraint(
            "status IN ('SUBMITTING', 'NEW', 'ACK', 'PARTIAL', 'FILLED', 'REJECTED', 'CANCELLED')",
            name="ck_orders_status_valid",
        ),
    )


class OrderEventModel(Base):
    __tablename__ = "order_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id_ref = Column(String(64), ForeignKey("orders.order_id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(32), nullable=False)
    from_status = Column(String(32), nullable=True)
    to_status = Column(String(32), nullable=False)
    filled_qty = Column(Float, nullable=False, default=0.0)
    avg_price = Column(Float, nullable=True)
    payload = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)

    order = relationship("OrderModel", back_populates="events")


class PositionModel(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False, index=True)
    exchange = Column(String(16), nullable=False, default="NSE")
    side = Column(String(8), nullable=False)
    quantity = Column(Float, nullable=False)
    avg_price = Column(Float, nullable=False)
    realized_pnl = Column(Float, nullable=False, default=0.0)
    strategy_id = Column(String(128), nullable=True)
    version = Column(Integer, nullable=False, default=0)  # optimistic concurrency control
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now)

    __table_args__ = (
        UniqueConstraint("symbol", "exchange", "side", name="uq_position_symbol_exchange_side"),
        CheckConstraint("quantity >= 0", name="ck_positions_quantity_non_negative"),
    )


class RiskSnapshotModel(Base):
    """Single-row snapshot for cold-start restore of equity and daily_pnl."""
    __tablename__ = "risk_snapshot"

    id = Column(Integer, primary_key=True, default=1)
    equity = Column(Float, nullable=False, default=0.0)
    daily_pnl = Column(Float, nullable=False, default=0.0)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now)


class UserModel(Base):
    """App users: persisted so they survive restart. Password stored hashed."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(128), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    email = Column(String(256), nullable=True)
    roles = Column(String(256), nullable=False, default="user")  # comma-separated: user,admin
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now)

    __table_args__ = ()


class OpenTradeModel(Base):
    """
    Persisted open trades with SL/TP levels.
    Write-ahead: row written BEFORE in-memory update.
    Deleted on trade close. Cold-start recovery loads these into AutonomousLoop._open_trades.
    """
    __tablename__ = "open_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_key = Column(String(256), unique=True, nullable=False, index=True)  # symbol:strategy_id
    symbol = Column(String(32), nullable=False, index=True)
    exchange = Column(String(16), nullable=False, default="NSE")
    side = Column(String(8), nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    trailing_stop = Column(Float, nullable=True)
    strategy_id = Column(String(128), nullable=True)
    opened_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now)

    __table_args__ = (
        CheckConstraint("quantity > 0", name="ck_open_trades_quantity_positive"),
        CheckConstraint("entry_price > 0", name="ck_open_trades_entry_price_positive"),
    )


class AuditEventModel(Base):
    """Immutable audit log. All critical actions must be traceable to actor. Append-only."""
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    event_type = Column(String(64), nullable=False, index=True)
    actor = Column(String(256), nullable=False, default="system")
    payload = Column(Text, nullable=True)  # JSON
    tenant_id = Column(String(64), nullable=True, index=True)

    __table_args__ = ()
