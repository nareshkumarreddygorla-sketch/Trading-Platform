"""
SQLite-based trade persistence store.

Lightweight alternative to PostgreSQL for trade tracking.
Works without any external database server -- trades survive server restarts.
Implements the same interface as OpenTradeRepository so AutonomousLoop can use either.
"""
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone, date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_CREATE_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_key   TEXT    UNIQUE NOT NULL,
    symbol      TEXT    NOT NULL,
    exchange    TEXT    NOT NULL DEFAULT 'NSE',
    side        TEXT    NOT NULL,
    qty         REAL    NOT NULL,
    entry_price REAL    NOT NULL,
    entry_time  TEXT    NOT NULL,
    exit_price  REAL,
    exit_time   TEXT,
    pnl         REAL,
    strategy_id TEXT,
    status      TEXT    NOT NULL DEFAULT 'OPEN',
    stop_loss   REAL,
    take_profit REAL,
    trailing_stop REAL,
    updated_at  TEXT    NOT NULL
)
"""

_CREATE_INDEX_STATUS = """
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)
"""
_CREATE_INDEX_SYMBOL = """
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)
"""
_CREATE_INDEX_ENTRY_TIME = """
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)
"""


class TradeStore:
    """
    SQLite-backed trade store.  Thread-safe via a lock around all DB operations.

    Dual purpose:
    1. Drop-in replacement for OpenTradeRepository (upsert_trade / delete_trade / load_all / update_sl_tp)
       so AutonomousLoop can persist open trades without PostgreSQL.
    2. Full trade lifecycle: save_trade, close_trade, get_daily_trades, get_all_trades.
    """

    def __init__(self, db_path: str = "data/trades.db"):
        self._db_path = db_path
        self._lock = threading.Lock()

        # Ensure parent directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        # Initialize schema
        with self._connect() as conn:
            conn.execute(_CREATE_TRADES_TABLE)
            conn.execute(_CREATE_INDEX_STATUS)
            conn.execute(_CREATE_INDEX_SYMBOL)
            conn.execute(_CREATE_INDEX_ENTRY_TIME)
            conn.commit()

        logger.info("TradeStore initialised (SQLite: %s)", os.path.abspath(db_path))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ------------------------------------------------------------------
    # OpenTradeRepository-compatible interface (used by AutonomousLoop)
    # ------------------------------------------------------------------

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
        """Insert or update an open trade. Returns True on success."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute(
                        """INSERT INTO trades
                               (trade_key, symbol, exchange, side, qty, entry_price,
                                entry_time, strategy_id, status, stop_loss, take_profit,
                                trailing_stop, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?)
                           ON CONFLICT(trade_key) DO UPDATE SET
                               qty=excluded.qty,
                               entry_price=excluded.entry_price,
                               stop_loss=excluded.stop_loss,
                               take_profit=excluded.take_profit,
                               trailing_stop=excluded.trailing_stop,
                               strategy_id=excluded.strategy_id,
                               updated_at=excluded.updated_at
                        """,
                        (trade_key, symbol, exchange, side, quantity, entry_price,
                         now, strategy_id, stop_loss, take_profit, trailing_stop, now),
                    )
                    conn.commit()
                return True
            except Exception as e:
                logger.error("TradeStore.upsert_trade failed for %s: %s", trade_key, e)
                return False

    def update_sl_tp(
        self,
        trade_key: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
    ) -> bool:
        """Update SL/TP/trailing stop for an existing open trade."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                with self._connect() as conn:
                    sets = ["updated_at=?"]
                    params: list = [now]
                    if stop_loss is not None:
                        sets.append("stop_loss=?")
                        params.append(stop_loss)
                    if take_profit is not None:
                        sets.append("take_profit=?")
                        params.append(take_profit)
                    if trailing_stop is not None:
                        sets.append("trailing_stop=?")
                        params.append(trailing_stop)
                    params.append(trade_key)
                    conn.execute(
                        f"UPDATE trades SET {', '.join(sets)} WHERE trade_key=? AND status='OPEN'",
                        params,
                    )
                    conn.commit()
                return True
            except Exception as e:
                logger.error("TradeStore.update_sl_tp failed for %s: %s", trade_key, e)
                return False

    def delete_trade(self, trade_key: str) -> bool:
        """Mark an open trade as CLOSED (soft delete). Used on SL/TP/kill-switch close."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute(
                        "UPDATE trades SET status='CLOSED', exit_time=?, updated_at=? WHERE trade_key=? AND status='OPEN'",
                        (now, now, trade_key),
                    )
                    conn.commit()
                return True
            except Exception as e:
                logger.error("TradeStore.delete_trade failed for %s: %s", trade_key, e)
                return False

    def load_all(self) -> List[Dict]:
        """Load all OPEN trades from SQLite. Used for cold-start recovery."""
        with self._lock:
            try:
                with self._connect() as conn:
                    rows = conn.execute(
                        "SELECT * FROM trades WHERE status='OPEN'"
                    ).fetchall()
                trades = []
                for r in rows:
                    entry_price = r["entry_price"]
                    if entry_price is None or entry_price <= 0:
                        logger.warning("Skipping corrupt trade row: trade_key=%s entry_price=%s",
                                       r["trade_key"], entry_price)
                        continue
                    trades.append({
                        "trade_key": r["trade_key"],
                        "symbol": r["symbol"],
                        "exchange": r["exchange"] or "NSE",
                        "side": r["side"],
                        "entry_price": entry_price,
                        "qty": r["qty"],
                        "strategy_id": r["strategy_id"],
                        "stop_loss": r["stop_loss"],
                        "take_profit": r["take_profit"],
                        "trailing_stop": r["trailing_stop"],
                    })
                logger.info("TradeStore: loaded %d open trades from SQLite", len(trades))
                return trades
            except Exception as e:
                logger.error("TradeStore.load_all failed: %s", e)
                return []

    # ------------------------------------------------------------------
    # Extended trade lifecycle API
    # ------------------------------------------------------------------

    def save_trade(self, trade_dict: dict) -> bool:
        """
        Save a trade record.  Accepts a dict with keys:
            symbol, side, qty, entry_price, strategy_id,
            stop_loss, take_profit, exit_price, exit_time, pnl, status
        trade_key is auto-generated as symbol:strategy_id if not provided.
        """
        trade_key = trade_dict.get("trade_key") or f"{trade_dict['symbol']}:{trade_dict.get('strategy_id', 'manual')}"
        now = datetime.now(timezone.utc).isoformat()
        status = trade_dict.get("status", "OPEN")
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute(
                        """INSERT INTO trades
                               (trade_key, symbol, exchange, side, qty, entry_price,
                                entry_time, exit_price, exit_time, pnl, strategy_id,
                                status, stop_loss, take_profit, trailing_stop, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(trade_key) DO UPDATE SET
                               qty=excluded.qty,
                               entry_price=excluded.entry_price,
                               exit_price=excluded.exit_price,
                               exit_time=excluded.exit_time,
                               pnl=excluded.pnl,
                               status=excluded.status,
                               stop_loss=excluded.stop_loss,
                               take_profit=excluded.take_profit,
                               trailing_stop=excluded.trailing_stop,
                               updated_at=excluded.updated_at
                        """,
                        (
                            trade_key,
                            trade_dict["symbol"],
                            trade_dict.get("exchange", "NSE"),
                            trade_dict["side"],
                            trade_dict["qty"],
                            trade_dict["entry_price"],
                            trade_dict.get("entry_time", now),
                            trade_dict.get("exit_price"),
                            trade_dict.get("exit_time"),
                            trade_dict.get("pnl"),
                            trade_dict.get("strategy_id"),
                            status,
                            trade_dict.get("stop_loss"),
                            trade_dict.get("take_profit"),
                            trade_dict.get("trailing_stop"),
                            now,
                        ),
                    )
                    conn.commit()
                return True
            except Exception as e:
                logger.error("TradeStore.save_trade failed: %s", e)
                return False

    def close_trade(self, trade_id: str, exit_price: float, exit_time: Optional[str] = None) -> bool:
        """
        Close a trade by trade_key. Computes PnL from entry_price and exit_price.
        """
        if exit_time is None:
            exit_time = datetime.now(timezone.utc).isoformat()
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                with self._connect() as conn:
                    row = conn.execute(
                        "SELECT * FROM trades WHERE trade_key=? AND status='OPEN'",
                        (trade_id,),
                    ).fetchone()
                    if row is None:
                        logger.warning("close_trade: trade_key=%s not found or already closed", trade_id)
                        return False
                    entry = row["entry_price"]
                    qty = row["qty"]
                    side = row["side"]
                    pnl = (exit_price - entry) * qty if side == "BUY" else (entry - exit_price) * qty
                    conn.execute(
                        """UPDATE trades
                           SET status='CLOSED', exit_price=?, exit_time=?, pnl=?, updated_at=?
                           WHERE trade_key=? AND status='OPEN'""",
                        (exit_price, exit_time, round(pnl, 2), now, trade_id),
                    )
                    conn.commit()
                return True
            except Exception as e:
                logger.error("TradeStore.close_trade failed for %s: %s", trade_id, e)
                return False

    def get_open_trades(self) -> List[dict]:
        """Return all currently open trades."""
        return self.load_all()

    def get_daily_trades(self, day: Optional[str] = None) -> List[dict]:
        """
        Return trades opened on a given date (YYYY-MM-DD).
        Defaults to today (UTC).
        """
        if day is None:
            day = date.today().isoformat()
        with self._lock:
            try:
                with self._connect() as conn:
                    rows = conn.execute(
                        "SELECT * FROM trades WHERE entry_time LIKE ? ORDER BY entry_time",
                        (f"{day}%",),
                    ).fetchall()
                return [dict(r) for r in rows]
            except Exception as e:
                logger.error("TradeStore.get_daily_trades failed: %s", e)
                return []

    def get_all_trades(self, limit: int = 100) -> List[dict]:
        """Return the most recent trades (both open and closed), newest first."""
        with self._lock:
            try:
                with self._connect() as conn:
                    rows = conn.execute(
                        "SELECT * FROM trades ORDER BY id DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                return [dict(r) for r in rows]
            except Exception as e:
                logger.error("TradeStore.get_all_trades failed: %s", e)
                return []
