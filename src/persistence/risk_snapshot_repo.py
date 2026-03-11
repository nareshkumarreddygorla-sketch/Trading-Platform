"""Risk snapshot repo: persist/restore equity and daily_pnl for cold start."""

import logging
from datetime import UTC, datetime

from .database import session_scope
from .models import RiskSnapshotModel

logger = logging.getLogger(__name__)


class RiskSnapshotRepository:
    """Single-row store for latest equity and daily_pnl."""

    def __init__(self, session_factory=None):
        from .database import get_session_factory

        self._session_factory = session_factory or get_session_factory()

    def get_latest(self) -> tuple[float, float] | None:
        """Return (equity, daily_pnl) for id=1, or None if no row."""
        with session_scope() as session:
            row = session.query(RiskSnapshotModel).filter(RiskSnapshotModel.id == 1).first()
            if row is None:
                return None
            return (float(row.equity), float(row.daily_pnl))

    def save(self, equity: float, daily_pnl: float) -> None:
        """Upsert row id=1 with given equity and daily_pnl."""
        with session_scope() as session:
            row = session.query(RiskSnapshotModel).filter(RiskSnapshotModel.id == 1).first()
            now = datetime.now(UTC)
            if row is None:
                session.add(RiskSnapshotModel(id=1, equity=equity, daily_pnl=daily_pnl, updated_at=now))
            else:
                row.equity = equity
                row.daily_pnl = daily_pnl
                row.updated_at = now
