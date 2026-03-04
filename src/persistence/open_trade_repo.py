"""
Repository for open trades (SL/TP tracking).

Write-ahead pattern: persist to DB BEFORE in-memory update.
On close: DELETE row.  On cold-start: load all rows into _open_trades dict.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from .database import session_scope
from .models import OpenTradeModel

logger = logging.getLogger(__name__)


class OpenTradeRepository:
    """Sync repository for open trades with SL/TP levels."""

    def __init__(self, session_factory=None):
        from .database import get_session_factory
        self._session_factory = session_factory or get_session_factory()

    def upsert_trade(
        self,
        trade_key: str,
        symbol: str,
        exchange: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        strategy_id: Optional[str] = None,
    ) -> bool:
        """
        Insert or update an open trade.  Write-ahead: call this BEFORE updating in-memory dict.
        Returns True on success, False on failure (caller should NOT update in-memory).
        """
        try:
            with session_scope() as sess:
                existing = sess.query(OpenTradeModel).filter(
                    OpenTradeModel.trade_key == trade_key
                ).first()
                now = datetime.now(timezone.utc)
                if existing is None:
                    sess.add(OpenTradeModel(
                        trade_key=trade_key,
                        symbol=symbol,
                        exchange=exchange,
                        side=side,
                        quantity=quantity,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        trailing_stop=trailing_stop,
                        strategy_id=strategy_id,
                        opened_at=now,
                        updated_at=now,
                    ))
                else:
                    existing.quantity = quantity
                    existing.entry_price = entry_price
                    existing.stop_loss = stop_loss
                    existing.take_profit = take_profit
                    existing.trailing_stop = trailing_stop
                    existing.strategy_id = strategy_id
                    existing.updated_at = now
            return True
        except Exception as e:
            logger.error("Failed to persist open trade %s: %s", trade_key, e)
            return False

    def update_sl_tp(
        self,
        trade_key: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
    ) -> bool:
        """Update SL/TP/trailing stop for an existing trade."""
        try:
            with session_scope() as sess:
                existing = sess.query(OpenTradeModel).filter(
                    OpenTradeModel.trade_key == trade_key
                ).first()
                if existing is None:
                    logger.warning("update_sl_tp: trade_key=%s not found in DB", trade_key)
                    return False
                if stop_loss is not None:
                    existing.stop_loss = stop_loss
                if take_profit is not None:
                    existing.take_profit = take_profit
                if trailing_stop is not None:
                    existing.trailing_stop = trailing_stop
                existing.updated_at = datetime.now(timezone.utc)
            return True
        except Exception as e:
            logger.error("Failed to update SL/TP for %s: %s", trade_key, e)
            return False

    def delete_trade(self, trade_key: str) -> bool:
        """Delete a closed trade from the open_trades table."""
        try:
            with session_scope() as sess:
                sess.query(OpenTradeModel).filter(
                    OpenTradeModel.trade_key == trade_key
                ).delete()
            return True
        except Exception as e:
            logger.error("Failed to delete open trade %s: %s", trade_key, e)
            return False

    def load_all(self) -> List[Dict]:
        """
        Load all open trades from DB.  Used for cold-start recovery.
        Returns list of dicts matching the _open_trades in-memory format.
        """
        try:
            with session_scope() as sess:
                rows = sess.query(OpenTradeModel).all()
                trades = []
                for r in rows:
                    if r.entry_price is None or r.entry_price <= 0:
                        logger.warning("Skipping corrupt open_trade row: trade_key=%s entry_price=%s",
                                       r.trade_key, r.entry_price)
                        continue
                    trades.append({
                        "trade_key": r.trade_key,
                        "symbol": r.symbol,
                        "exchange": r.exchange or "NSE",
                        "side": r.side,
                        "entry_price": r.entry_price,
                        "qty": r.quantity,
                        "strategy_id": r.strategy_id,
                        "stop_loss": r.stop_loss,
                        "take_profit": r.take_profit,
                        "trailing_stop": r.trailing_stop,
                    })
                logger.info("Loaded %d open trades from DB (skipped %d corrupt)",
                            len(trades), len(rows) - len(trades))
                return trades
        except Exception as e:
            logger.error("Failed to load open trades from DB: %s", e)
            return []
