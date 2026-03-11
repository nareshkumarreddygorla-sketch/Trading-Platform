"""
OHLCV Repository: store and retrieve historical bar data.
Supports bulk upsert for nightly data refresh.
"""
import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy import and_, text

from src.core.events import Bar, Exchange
from src.persistence.database import get_engine, session_scope
from src.persistence.models_market_data import OHLCVBarModel

logger = logging.getLogger(__name__)


def _is_sqlite() -> bool:
    return str(get_engine().url).startswith("sqlite")


class OHLCVRepository:
    """CRUD operations for OHLCV bars with conflict-free upsert."""

    def upsert_bars(self, bars: List[Bar]) -> int:
        """
        Insert or update bars. Uses PostgreSQL ON CONFLICT or SQLite INSERT OR REPLACE.
        Returns number of rows affected.
        """
        if not bars:
            return 0

        with session_scope() as session:
            rows = []
            for bar in bars:
                rows.append({
                    "symbol": bar.symbol,
                    "exchange": bar.exchange.value if hasattr(bar.exchange, "value") else str(bar.exchange),
                    "interval": bar.interval,
                    "timestamp": bar.ts,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                })

            if _is_sqlite():
                # SQLite: use INSERT OR REPLACE (must include created_at for NOT NULL constraint)
                now = datetime.utcnow().isoformat()
                for row in rows:
                    row["created_at"] = now
                    session.execute(
                        text("""INSERT OR REPLACE INTO ohlcv_bars
                            (symbol, exchange, interval, timestamp, open, high, low, close, volume, created_at)
                            VALUES (:symbol, :exchange, :interval, :timestamp, :open, :high, :low, :close, :volume, :created_at)"""),
                        row,
                    )
                session.commit()
                count = len(rows)
            else:
                # PostgreSQL: batch upsert
                from sqlalchemy.dialects.postgresql import insert as pg_insert
                stmt = pg_insert(OHLCVBarModel).values(rows)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_ohlcv_symbol_exchange_interval_ts",
                    set_={
                        "open": stmt.excluded.open,
                        "high": stmt.excluded.high,
                        "low": stmt.excluded.low,
                        "close": stmt.excluded.close,
                        "volume": stmt.excluded.volume,
                    },
                )
                result = session.execute(stmt)
                session.commit()
                count = result.rowcount or len(rows)

            logger.info("Upserted %d OHLCV bars", count)
            return count

    def get_bars(
        self,
        symbol: str,
        interval: str = "1d",
        exchange: str = "NSE",
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[Bar]:
        """Retrieve bars for a symbol, sorted by timestamp ascending."""
        with session_scope() as session:
            query = session.query(OHLCVBarModel).filter(
                and_(
                    OHLCVBarModel.symbol == symbol,
                    OHLCVBarModel.exchange == exchange,
                    OHLCVBarModel.interval == interval,
                )
            )
            if from_date:
                query = query.filter(OHLCVBarModel.timestamp >= from_date)
            if to_date:
                query = query.filter(OHLCVBarModel.timestamp <= to_date)

            query = query.order_by(OHLCVBarModel.timestamp.asc()).limit(limit)
            rows = query.all()

            return [
                Bar(
                    symbol=r.symbol,
                    exchange=Exchange.NSE if r.exchange == "NSE" else Exchange.BSE,
                    interval=r.interval,
                    ts=r.timestamp,
                    open=r.open,
                    high=r.high,
                    low=r.low,
                    close=r.close,
                    volume=r.volume,
                )
                for r in rows
            ]

    def get_symbols_with_data(self, interval: str = "1d", exchange: str = "NSE", min_bars: int = 50) -> List[str]:
        """Return symbols that have at least min_bars of data."""
        with session_scope() as session:
            result = session.execute(
                text("""
                    SELECT symbol, COUNT(*) as cnt
                    FROM ohlcv_bars
                    WHERE exchange = :exchange AND interval = :interval
                    GROUP BY symbol
                    HAVING COUNT(*) >= :min_bars
                    ORDER BY cnt DESC
                """),
                {"exchange": exchange, "interval": interval, "min_bars": min_bars},
            )
            return [row[0] for row in result]

    def get_latest_timestamp(self, symbol: str, interval: str = "1d", exchange: str = "NSE") -> Optional[datetime]:
        """Get the most recent bar timestamp for a symbol."""
        with session_scope() as session:
            row = (
                session.query(OHLCVBarModel.timestamp)
                .filter(
                    and_(
                        OHLCVBarModel.symbol == symbol,
                        OHLCVBarModel.exchange == exchange,
                        OHLCVBarModel.interval == interval,
                    )
                )
                .order_by(OHLCVBarModel.timestamp.desc())
                .first()
            )
            return row[0] if row else None

    def delete_old_bars(self, before_date: datetime, interval: str = "1m") -> int:
        """Cleanup: delete bars older than a date (useful for 1m data)."""
        with session_scope() as session:
            result = session.execute(
                text("""
                    DELETE FROM ohlcv_bars
                    WHERE interval = :interval AND timestamp < :before_date
                """),
                {"interval": interval, "before_date": before_date},
            )
            session.commit()
            count = result.rowcount or 0
            logger.info("Deleted %d old %s bars before %s", count, interval, before_date)
            return count
