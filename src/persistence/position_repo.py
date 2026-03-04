"""Position repository. Sync API; merge fills by symbol+exchange+side. OCC via version column."""
import logging
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import and_, delete, update
from sqlalchemy.orm import Session

from src.core.events import Exchange, Position, SignalSide

from .database import session_scope
from .models import PositionModel

logger = logging.getLogger(__name__)


class PositionConcurrentUpdateError(Exception):
    """Raised when position update fails due to version mismatch (optimistic lock conflict)."""
    pass


def _model_to_domain(m: PositionModel, unrealized_pnl: float = 0.0) -> Position:
    return Position(
        symbol=m.symbol,
        exchange=Exchange(m.exchange) if m.exchange else Exchange.NSE,
        side=SignalSide(m.side) if m.side else SignalSide.BUY,
        quantity=m.quantity,
        avg_price=m.avg_price,
        unrealized_pnl=unrealized_pnl,
        strategy_id=m.strategy_id,
    )


class PositionRepository:
    """Sync repository for positions. Upsert on fill (merge quantity, VWAP)."""

    def __init__(self, session_factory=None):
        from .database import get_session_factory
        self._session_factory = session_factory or get_session_factory()

    def upsert_from_fill(
        self,
        symbol: str,
        exchange: str,
        side: str,
        fill_qty: float,
        avg_price: float,
        strategy_id: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> None:
        """Merge a fill into position (symbol, exchange, side). New position or add to existing (VWAP). If session given, use it (caller commits)."""
        if fill_qty <= 0 or avg_price <= 0:
            return

        def _upsert(sess: Session) -> None:
            m = sess.query(PositionModel).filter(
                and_(
                    PositionModel.symbol == symbol,
                    PositionModel.exchange == exchange,
                    PositionModel.side == side,
                )
            ).first()
            now = datetime.now(timezone.utc)
            if m is None:
                sess.add(
                    PositionModel(
                        symbol=symbol,
                        exchange=exchange,
                        side=side,
                        quantity=fill_qty,
                        avg_price=avg_price,
                        realized_pnl=0.0,
                        strategy_id=strategy_id,
                        version=0,
                        updated_at=now,
                    )
                )
            else:
                total_qty = m.quantity + fill_qty
                current_version = getattr(m, "version", 0)
                if total_qty <= 0:
                    result = sess.execute(
                        delete(PositionModel).where(
                            and_(
                                PositionModel.id == m.id,
                                PositionModel.version == current_version,
                            )
                        )
                    )
                    if result.rowcount == 0:
                        raise PositionConcurrentUpdateError(f"position id={m.id} version conflict on delete")
                else:
                    new_avg = (m.quantity * m.avg_price + fill_qty * avg_price) / total_qty
                    result = sess.execute(
                        update(PositionModel)
                        .where(
                            and_(
                                PositionModel.id == m.id,
                                PositionModel.version == current_version,
                            )
                        )
                        .values(
                            quantity=total_qty,
                            avg_price=new_avg,
                            version=current_version + 1,
                            updated_at=now,
                            strategy_id=strategy_id or m.strategy_id,
                        )
                    )
                    if result.rowcount == 0:
                        raise PositionConcurrentUpdateError(f"position id={m.id} version conflict on upsert")

        if session is not None:
            _upsert(session)
        else:
            with session_scope() as sess:
                _upsert(sess)

    def list_positions(self, session: Optional[Session] = None) -> List[Position]:
        """Return all positions. unrealized_pnl left 0 (computed elsewhere if needed)."""
        def _list(sess: Session) -> List[Position]:
            rows = sess.query(PositionModel).filter(PositionModel.quantity > 0).all()
            return [_model_to_domain(r) for r in rows]

        if session is not None:
            return _list(session)
        with session_scope() as sess:
            return _list(sess)
