"""
Trade Outcome Repository: record closed trade outcomes for self-learning and attribution.
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, func

from src.persistence.database import session_scope
from src.persistence.models_market_data import TradeOutcomeModel

logger = logging.getLogger(__name__)


class TradeOutcomeRepository:
    """Stores and queries closed trade outcomes."""

    def record_outcome(
        self,
        trade_key: str,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        strategy_id: str = "",
        model_id: str = "",
        signal_confidence: float = 0.0,
        signal_score: float = 0.0,
        regime_at_entry: str = "",
        features_at_entry: Optional[Dict[str, float]] = None,
        entry_time: Optional[datetime] = None,
        exit_reason: str = "",
        holding_bars: int = 0,
        exchange: str = "NSE",
    ) -> None:
        """Record a single closed trade outcome."""
        if side == "BUY":
            realized_pnl = (exit_price - entry_price) * quantity
            pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
        else:
            realized_pnl = (entry_price - exit_price) * quantity
            pnl_pct = (entry_price - exit_price) / entry_price if entry_price > 0 else 0.0

        with session_scope() as session:
            outcome = TradeOutcomeModel(
                trade_key=trade_key,
                symbol=symbol,
                exchange=exchange,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                exit_price=exit_price,
                realized_pnl=realized_pnl,
                pnl_pct=pnl_pct,
                holding_bars=holding_bars,
                strategy_id=strategy_id,
                model_id=model_id,
                signal_confidence=signal_confidence,
                signal_score=signal_score,
                regime_at_entry=regime_at_entry,
                features_at_entry=json.dumps(features_at_entry) if features_at_entry else None,
                entry_time=entry_time or datetime.utcnow(),
                exit_reason=exit_reason,
            )
            session.add(outcome)
            session.commit()
            logger.info("Recorded trade outcome: %s %s PnL=%.2f (%.2f%%)",
                        side, symbol, realized_pnl, pnl_pct * 100)

    def get_outcomes_by_model(
        self,
        model_id: str,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get trade outcomes for a specific model."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        with session_scope() as session:
            rows = (
                session.query(TradeOutcomeModel)
                .filter(and_(
                    TradeOutcomeModel.model_id == model_id,
                    TradeOutcomeModel.exit_time >= cutoff,
                ))
                .order_by(TradeOutcomeModel.exit_time.desc())
                .all()
            )
            return [self._row_to_dict(r) for r in rows]

    def get_outcomes_by_strategy(
        self,
        strategy_id: str,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get trade outcomes for a specific strategy."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        with session_scope() as session:
            rows = (
                session.query(TradeOutcomeModel)
                .filter(and_(
                    TradeOutcomeModel.strategy_id == strategy_id,
                    TradeOutcomeModel.exit_time >= cutoff,
                ))
                .order_by(TradeOutcomeModel.exit_time.desc())
                .all()
            )
            return [self._row_to_dict(r) for r in rows]

    def get_attribution_by_dimension(
        self,
        dimension: str = "model_id",
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Performance attribution aggregated by dimension.
        Dimensions: model_id, strategy_id, symbol, regime_at_entry, exit_reason.
        """
        valid_dims = {"model_id", "strategy_id", "symbol", "regime_at_entry", "exit_reason", "side"}
        if dimension not in valid_dims:
            raise ValueError(f"Invalid dimension: {dimension}")

        cutoff = datetime.utcnow() - timedelta(days=days)
        col = getattr(TradeOutcomeModel, dimension)

        with session_scope() as session:
            rows = (
                session.query(
                    col,
                    func.count().label("total_trades"),
                    func.sum(TradeOutcomeModel.realized_pnl).label("total_pnl"),
                    func.avg(TradeOutcomeModel.pnl_pct).label("avg_return_pct"),
                    func.sum(
                        func.cast(TradeOutcomeModel.realized_pnl > 0, Integer)
                    ).label("wins"),
                    func.avg(TradeOutcomeModel.holding_bars).label("avg_holding"),
                    func.avg(TradeOutcomeModel.signal_confidence).label("avg_confidence"),
                )
                .filter(TradeOutcomeModel.exit_time >= cutoff)
                .group_by(col)
                .order_by(func.sum(TradeOutcomeModel.realized_pnl).desc())
                .all()
            )
            results = []
            for r in rows:
                total = r.total_trades or 0
                wins = r.wins or 0
                results.append({
                    dimension: r[0],
                    "total_trades": total,
                    "total_pnl": float(r.total_pnl or 0),
                    "avg_return_pct": float(r.avg_return_pct or 0) * 100,
                    "win_rate": (wins / total * 100) if total > 0 else 0,
                    "avg_holding_bars": float(r.avg_holding or 0),
                    "avg_confidence": float(r.avg_confidence or 0),
                })
            return results

    def get_recent_outcomes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get most recent trade outcomes."""
        with session_scope() as session:
            rows = (
                session.query(TradeOutcomeModel)
                .order_by(TradeOutcomeModel.exit_time.desc())
                .limit(limit)
                .all()
            )
            return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(r: TradeOutcomeModel) -> Dict[str, Any]:
        return {
            "id": r.id,
            "trade_key": r.trade_key,
            "symbol": r.symbol,
            "exchange": r.exchange,
            "side": r.side,
            "quantity": r.quantity,
            "entry_price": r.entry_price,
            "exit_price": r.exit_price,
            "realized_pnl": r.realized_pnl,
            "pnl_pct": r.pnl_pct * 100,
            "holding_bars": r.holding_bars,
            "strategy_id": r.strategy_id,
            "model_id": r.model_id,
            "signal_confidence": r.signal_confidence,
            "regime_at_entry": r.regime_at_entry,
            "entry_time": r.entry_time.isoformat() if r.entry_time else None,
            "exit_time": r.exit_time.isoformat() if r.exit_time else None,
            "exit_reason": r.exit_reason,
        }


# Convenience function used by Integer in the query above
from sqlalchemy import Integer
