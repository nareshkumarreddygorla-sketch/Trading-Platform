"""
TradeStore tests: SQLite persistence layer for autonomous trading.

Tests cover CRUD, SL/TP updates, PnL computation on close,
daily filters, concurrent access, and cross-instance persistence.

Each test uses a unique temp directory to avoid cross-test contamination.
"""
import os
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timezone, date

import pytest

from src.persistence.trade_store import TradeStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    """Return a unique SQLite path inside a pytest tmp_path."""
    return str(tmp_path / "test_trades.db")


@pytest.fixture
def store(db_path) -> TradeStore:
    """Create a fresh TradeStore instance for each test."""
    return TradeStore(db_path=db_path)


# =========================================================================
# 1. Create and load trade
# =========================================================================

class TestCreateAndLoad:
    def test_upsert_and_load_all(self, store):
        """upsert_trade persists a trade and load_all returns it."""
        ok = store.upsert_trade(
            trade_key="RELIANCE:strat1",
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            quantity=10,
            entry_price=2500.0,
            stop_loss=2400.0,
            take_profit=2700.0,
            strategy_id="strat1",
        )
        assert ok is True

        trades = store.load_all()
        assert len(trades) == 1
        t = trades[0]
        assert t["trade_key"] == "RELIANCE:strat1"
        assert t["symbol"] == "RELIANCE"
        assert t["exchange"] == "NSE"
        assert t["side"] == "BUY"
        assert t["qty"] == 10
        assert t["entry_price"] == 2500.0
        assert t["stop_loss"] == 2400.0
        assert t["take_profit"] == 2700.0
        assert t["strategy_id"] == "strat1"

    def test_upsert_updates_existing(self, store):
        """upsert_trade with the same trade_key updates rather than duplicates."""
        store.upsert_trade(
            trade_key="TCS:strat2",
            symbol="TCS",
            exchange="NSE",
            side="BUY",
            quantity=5,
            entry_price=3900.0,
            strategy_id="strat2",
        )
        # Update quantity and stop_loss
        store.upsert_trade(
            trade_key="TCS:strat2",
            symbol="TCS",
            exchange="NSE",
            side="BUY",
            quantity=15,
            entry_price=3900.0,
            stop_loss=3800.0,
            strategy_id="strat2",
        )

        trades = store.load_all()
        assert len(trades) == 1
        assert trades[0]["qty"] == 15
        assert trades[0]["stop_loss"] == 3800.0

    def test_save_trade_dict(self, store):
        """save_trade accepts a dict and persists it."""
        ok = store.save_trade({
            "symbol": "INFY",
            "exchange": "NSE",
            "side": "SELL",
            "qty": 20,
            "entry_price": 1500.0,
            "strategy_id": "rsi",
        })
        assert ok is True

        all_trades = store.get_all_trades()
        assert len(all_trades) == 1
        assert all_trades[0]["symbol"] == "INFY"
        assert all_trades[0]["side"] == "SELL"


# =========================================================================
# 2. Update stop loss / take profit
# =========================================================================

class TestUpdateSLTP:
    def test_update_stop_loss(self, store):
        """update_sl_tp changes the SL for an open trade."""
        store.upsert_trade(
            trade_key="SBIN:strat1",
            symbol="SBIN",
            exchange="NSE",
            side="BUY",
            quantity=25,
            entry_price=800.0,
            stop_loss=750.0,
            strategy_id="strat1",
        )

        ok = store.update_sl_tp(
            trade_key="SBIN:strat1",
            stop_loss=770.0,
        )
        assert ok is True

        trades = store.load_all()
        assert len(trades) == 1
        assert trades[0]["stop_loss"] == 770.0

    def test_update_take_profit(self, store):
        """update_sl_tp changes the TP for an open trade."""
        store.upsert_trade(
            trade_key="SBIN:strat2",
            symbol="SBIN",
            exchange="NSE",
            side="BUY",
            quantity=25,
            entry_price=800.0,
            take_profit=900.0,
            strategy_id="strat2",
        )

        ok = store.update_sl_tp(
            trade_key="SBIN:strat2",
            take_profit=950.0,
        )
        assert ok is True

        trades = store.load_all()
        assert trades[0]["take_profit"] == 950.0

    def test_update_trailing_stop(self, store):
        """update_sl_tp can update the trailing stop."""
        store.upsert_trade(
            trade_key="HDFC:ts",
            symbol="HDFC",
            exchange="NSE",
            side="BUY",
            quantity=10,
            entry_price=1600.0,
            trailing_stop=50.0,
            strategy_id="ts",
        )

        ok = store.update_sl_tp(
            trade_key="HDFC:ts",
            trailing_stop=40.0,
        )
        assert ok is True

        trades = store.load_all()
        assert trades[0]["trailing_stop"] == 40.0


# =========================================================================
# 3. Close trade calculates PnL
# =========================================================================

class TestCloseTrade:
    def test_close_trade_buy_calculates_pnl(self, store):
        """close_trade computes PnL correctly for a BUY trade."""
        store.upsert_trade(
            trade_key="RELIANCE:pnl_test",
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            quantity=10,
            entry_price=2500.0,
            strategy_id="pnl_test",
        )

        # Close at 2600 -> profit = (2600 - 2500) * 10 = 1000
        ok = store.close_trade("RELIANCE:pnl_test", exit_price=2600.0)
        assert ok is True

        # After close, load_all should return empty (only OPEN trades)
        open_trades = store.load_all()
        assert len(open_trades) == 0

        # But get_all_trades should show the closed trade with PnL
        all_trades = store.get_all_trades()
        assert len(all_trades) == 1
        closed = all_trades[0]
        assert closed["status"] == "CLOSED"
        assert closed["pnl"] == 1000.0
        assert closed["exit_price"] == 2600.0

    def test_close_trade_sell_calculates_pnl(self, store):
        """close_trade computes PnL correctly for a SELL (short) trade."""
        store.upsert_trade(
            trade_key="TCS:short",
            symbol="TCS",
            exchange="NSE",
            side="SELL",
            quantity=5,
            entry_price=4000.0,
            strategy_id="short",
        )

        # Close at 3900 -> profit = (4000 - 3900) * 5 = 500
        ok = store.close_trade("TCS:short", exit_price=3900.0)
        assert ok is True

        all_trades = store.get_all_trades()
        assert all_trades[0]["pnl"] == 500.0

    def test_close_trade_loss(self, store):
        """close_trade computes negative PnL for a losing BUY trade."""
        store.upsert_trade(
            trade_key="INFY:loss",
            symbol="INFY",
            exchange="NSE",
            side="BUY",
            quantity=10,
            entry_price=1500.0,
            strategy_id="loss_test",
        )

        # Close at 1450 -> loss = (1450 - 1500) * 10 = -500
        store.close_trade("INFY:loss", exit_price=1450.0)

        all_trades = store.get_all_trades()
        assert all_trades[0]["pnl"] == -500.0

    def test_close_nonexistent_trade_returns_false(self, store):
        """Closing a trade that doesn't exist returns False."""
        ok = store.close_trade("NONEXISTENT:trade", exit_price=100.0)
        assert ok is False

    def test_close_already_closed_trade_returns_false(self, store):
        """Closing a trade that is already CLOSED returns False."""
        store.upsert_trade(
            trade_key="DBL:close",
            symbol="DBL",
            exchange="NSE",
            side="BUY",
            quantity=1,
            entry_price=100.0,
            strategy_id="x",
        )
        store.close_trade("DBL:close", exit_price=110.0)
        # Second close should fail
        ok = store.close_trade("DBL:close", exit_price=120.0)
        assert ok is False


# =========================================================================
# 4. Daily trades filter
# =========================================================================

class TestDailyTrades:
    def test_get_daily_trades_returns_today(self, store):
        """get_daily_trades returns only trades opened today."""
        # Create a trade (entry_time defaults to now)
        store.upsert_trade(
            trade_key="TODAY:test",
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            quantity=5,
            entry_price=2500.0,
            strategy_id="daily",
        )

        today_str = date.today().isoformat()
        daily = store.get_daily_trades(day=today_str)
        assert len(daily) >= 1
        assert daily[0]["symbol"] == "RELIANCE"

    def test_get_daily_trades_excludes_other_days(self, store):
        """get_daily_trades for a different date returns empty."""
        store.upsert_trade(
            trade_key="OTHER:test",
            symbol="TCS",
            exchange="NSE",
            side="SELL",
            quantity=3,
            entry_price=3900.0,
            strategy_id="daily2",
        )

        # Query a date far in the past
        daily = store.get_daily_trades(day="2020-01-01")
        assert len(daily) == 0

    def test_get_daily_trades_default_is_today(self, store):
        """get_daily_trades without a day argument defaults to today."""
        store.upsert_trade(
            trade_key="DEFAULT:today",
            symbol="HDFC",
            exchange="NSE",
            side="BUY",
            quantity=10,
            entry_price=1600.0,
            strategy_id="d",
        )

        daily = store.get_daily_trades()
        assert len(daily) >= 1


# =========================================================================
# 5. Concurrent access
# =========================================================================

class TestConcurrentAccess:
    def test_concurrent_writes(self, store):
        """Multiple threads writing simultaneously should not corrupt the DB."""
        errors = []
        n_threads = 10
        trades_per_thread = 20

        def _writer(thread_id: int):
            try:
                for i in range(trades_per_thread):
                    store.upsert_trade(
                        trade_key=f"T{thread_id}:trade{i}",
                        symbol=f"SYM{thread_id}",
                        exchange="NSE",
                        side="BUY",
                        quantity=i + 1,
                        entry_price=100.0 + i,
                        strategy_id=f"strat{thread_id}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_writer, args=(tid,)) for tid in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Concurrent write errors: {errors}"

        # Verify all trades were written
        all_trades = store.load_all()
        expected_count = n_threads * trades_per_thread
        assert len(all_trades) == expected_count, (
            f"Expected {expected_count} trades, got {len(all_trades)}"
        )

    def test_concurrent_read_write(self, store):
        """Concurrent reads and writes should not cause corruption."""
        errors = []

        # Pre-populate some trades
        for i in range(50):
            store.upsert_trade(
                trade_key=f"PRE:{i}",
                symbol="PRE",
                exchange="NSE",
                side="BUY",
                quantity=1,
                entry_price=100.0,
                strategy_id="pre",
            )

        def _reader():
            try:
                for _ in range(30):
                    trades = store.load_all()
                    # Should always get at least the pre-populated trades
                    assert len(trades) >= 50
            except Exception as e:
                errors.append(("reader", e))

        def _writer():
            try:
                for i in range(20):
                    store.upsert_trade(
                        trade_key=f"NEW:{i}",
                        symbol="NEW",
                        exchange="NSE",
                        side="SELL",
                        quantity=1,
                        entry_price=200.0,
                        strategy_id="new",
                    )
            except Exception as e:
                errors.append(("writer", e))

        threads = [
            threading.Thread(target=_reader),
            threading.Thread(target=_reader),
            threading.Thread(target=_writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Concurrent r/w errors: {errors}"


# =========================================================================
# 6. Persistence across instances
# =========================================================================

class TestCrossInstancePersistence:
    def test_persist_across_instances(self, db_path):
        """Data written by one TradeStore instance is visible from a new instance
        pointing to the same SQLite file."""
        store1 = TradeStore(db_path=db_path)
        store1.upsert_trade(
            trade_key="PERSIST:test",
            symbol="WIPRO",
            exchange="NSE",
            side="BUY",
            quantity=50,
            entry_price=450.0,
            strategy_id="persist",
        )

        # Create a brand-new instance on the same DB
        store2 = TradeStore(db_path=db_path)
        trades = store2.load_all()
        assert len(trades) == 1
        assert trades[0]["trade_key"] == "PERSIST:test"
        assert trades[0]["symbol"] == "WIPRO"
        assert trades[0]["qty"] == 50

    def test_persist_closed_trade_across_instances(self, db_path):
        """Closed trades with PnL survive across instances."""
        store1 = TradeStore(db_path=db_path)
        store1.upsert_trade(
            trade_key="CLOSE:persist",
            symbol="TATASTEEL",
            exchange="NSE",
            side="BUY",
            quantity=100,
            entry_price=120.0,
            strategy_id="cp",
        )
        store1.close_trade("CLOSE:persist", exit_price=130.0)

        # New instance
        store2 = TradeStore(db_path=db_path)

        # load_all returns only OPEN trades
        open_trades = store2.load_all()
        assert len(open_trades) == 0

        # get_all_trades returns closed trades too
        all_trades = store2.get_all_trades()
        assert len(all_trades) == 1
        assert all_trades[0]["pnl"] == 1000.0  # (130 - 120) * 100
        assert all_trades[0]["status"] == "CLOSED"


# =========================================================================
# 7. Delete trade (soft delete)
# =========================================================================

class TestDeleteTrade:
    def test_delete_trade_marks_closed(self, store):
        """delete_trade soft-deletes by setting status=CLOSED."""
        store.upsert_trade(
            trade_key="DEL:test",
            symbol="ITC",
            exchange="NSE",
            side="BUY",
            quantity=100,
            entry_price=400.0,
            strategy_id="del",
        )

        ok = store.delete_trade("DEL:test")
        assert ok is True

        # Should no longer appear in load_all (OPEN only)
        open_trades = store.load_all()
        assert len(open_trades) == 0

    def test_delete_nonexistent_trade(self, store):
        """Deleting a trade that doesn't exist returns True
        (UPDATE affects 0 rows, no error)."""
        ok = store.delete_trade("NONE:exists")
        assert ok is True  # No error, just 0 rows affected


# =========================================================================
# 8. Edge cases
# =========================================================================

class TestEdgeCases:
    def test_get_open_trades_alias(self, store):
        """get_open_trades is an alias for load_all."""
        store.upsert_trade(
            trade_key="ALIAS:test",
            symbol="AXIS",
            exchange="NSE",
            side="BUY",
            quantity=10,
            entry_price=900.0,
            strategy_id="alias",
        )
        open_t = store.get_open_trades()
        load_t = store.load_all()
        assert len(open_t) == len(load_t)
        assert open_t[0]["trade_key"] == load_t[0]["trade_key"]

    def test_get_all_trades_ordering(self, store):
        """get_all_trades returns newest first."""
        for i in range(5):
            store.upsert_trade(
                trade_key=f"ORD:{i}",
                symbol=f"SYM{i}",
                exchange="NSE",
                side="BUY",
                quantity=1,
                entry_price=100.0,
                strategy_id=f"ord{i}",
            )

        all_trades = store.get_all_trades()
        assert len(all_trades) == 5
        # IDs are auto-incrementing, so newest has highest id
        ids = [t["id"] for t in all_trades]
        assert ids == sorted(ids, reverse=True)

    def test_get_all_trades_limit(self, store):
        """get_all_trades respects the limit parameter."""
        for i in range(10):
            store.upsert_trade(
                trade_key=f"LIM:{i}",
                symbol="LIM",
                exchange="NSE",
                side="BUY",
                quantity=1,
                entry_price=100.0,
                strategy_id="lim",
            )

        limited = store.get_all_trades(limit=3)
        assert len(limited) == 3

    def test_corrupt_entry_price_skipped(self, db_path):
        """Trades with entry_price=0 or NULL are skipped by load_all."""
        store = TradeStore(db_path=db_path)

        # Insert a valid trade
        store.upsert_trade(
            trade_key="VALID:trade",
            symbol="VALID",
            exchange="NSE",
            side="BUY",
            quantity=10,
            entry_price=100.0,
            strategy_id="v",
        )

        # Manually insert a corrupt trade with entry_price=0
        conn = sqlite3.connect(db_path)
        conn.execute(
            """INSERT INTO trades
               (trade_key, symbol, exchange, side, qty, entry_price,
                entry_time, status, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("CORRUPT:trade", "CORRUPT", "NSE", "BUY", 5, 0,
             datetime.now(timezone.utc).isoformat(), "OPEN",
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()

        trades = store.load_all()
        # Only the valid trade should be returned
        assert len(trades) == 1
        assert trades[0]["trade_key"] == "VALID:trade"
