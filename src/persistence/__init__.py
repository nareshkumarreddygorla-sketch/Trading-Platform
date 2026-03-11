"""Persistence layer: orders, order events, positions, risk snapshot, users (Postgres + SQLite fallback)."""

from .audit_repo import AuditRepository
from .database import get_engine, get_session_factory, session_scope
from .models import Base, OrderEventModel, OrderModel, PositionModel, RiskSnapshotModel, UserModel
from .order_repo import OrderRepository
from .position_repo import PositionRepository
from .reconciliation import reconcile_positions
from .risk_snapshot_repo import RiskSnapshotRepository
from .service import PersistenceService
from .trade_store import TradeStore
from .user_repo import UserRepository

__all__ = [
    "Base",
    "OrderModel",
    "OrderEventModel",
    "PositionModel",
    "RiskSnapshotModel",
    "UserModel",
    "get_engine",
    "get_session_factory",
    "session_scope",
    "OrderRepository",
    "PositionRepository",
    "RiskSnapshotRepository",
    "AuditRepository",
    "UserRepository",
    "PersistenceService",
    "reconcile_positions",
    "TradeStore",
]
