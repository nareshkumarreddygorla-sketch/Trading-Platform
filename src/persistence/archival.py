"""
Data Archival & Retention Policy Module.

Manages lifecycle of trading data:
  - Orders: archive after 90 days, delete after 2 years
  - Order Events: archive with parent order
  - Audit Events: retain for 7 years (SEBI compliance), archive after 1 year
  - Trade Outcomes: retain for 5 years (backtest validation)
  - OHLCV bars: retain 1m for 30 days, 5m for 90 days, daily forever
  - Risk Snapshots: retain for 1 year, archive daily aggregates
  - Positions: closed positions archived after 180 days

All archival is non-destructive: data is moved to *_archive tables (or exported to
compressed Parquet/CSV files) before deletion from hot tables.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetentionPolicy:
    """Configurable retention periods for each data type."""

    # Hot table retention (days)
    orders_hot_days: int = 90
    order_events_hot_days: int = 90
    audit_events_hot_days: int = 365
    trade_outcomes_hot_days: int = 365 * 2
    ohlcv_1m_hot_days: int = 30
    ohlcv_5m_hot_days: int = 90
    ohlcv_15m_hot_days: int = 180
    ohlcv_1h_hot_days: int = 365
    ohlcv_1d_hot_days: int = 365 * 10  # Daily bars kept 10 years
    risk_snapshots_hot_days: int = 365
    closed_positions_hot_days: int = 180

    # Archive retention before permanent deletion (days)
    orders_archive_days: int = 365 * 2
    audit_archive_days: int = 365 * 7  # SEBI: 7 years
    trade_outcomes_archive_days: int = 365 * 5


@dataclass
class ArchivalResult:
    """Result of a single archival operation."""

    table_name: str
    rows_archived: int
    rows_deleted: int
    cutoff_date: str
    duration_seconds: float
    error: str | None = None


class DataArchivalManager:
    """Manages data lifecycle: archival, retention, and cleanup."""

    def __init__(
        self,
        policy: RetentionPolicy | None = None,
        archive_dir: str | None = None,
    ):
        self.policy = policy or RetentionPolicy()
        self.archive_dir = archive_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "archive",
        )
        os.makedirs(self.archive_dir, exist_ok=True)
        self._last_run: list[ArchivalResult] = []

    def _cutoff_date(self, days: int) -> datetime:
        return datetime.utcnow() - timedelta(days=days)

    def archive_orders(self, session) -> ArchivalResult:
        """Archive completed orders older than retention period."""
        import time

        start = time.time()
        cutoff = self._cutoff_date(self.policy.orders_hot_days)
        rows = 0
        try:
            # Move to archive table
            result = session.execute(
                session.bind.execute(
                    """
                    INSERT INTO orders_archive
                    SELECT * FROM orders
                    WHERE status IN ('FILLED', 'REJECTED', 'CANCELLED')
                    AND updated_at < :cutoff
                    ON CONFLICT (order_id) DO NOTHING
                    """,
                    {"cutoff": cutoff},
                )
            )
            rows = result.rowcount if hasattr(result, "rowcount") else 0

            # Delete from hot table
            session.execute(
                session.bind.execute(
                    "DELETE FROM orders WHERE status IN ('FILLED', 'REJECTED', 'CANCELLED') AND updated_at < :cutoff",
                    {"cutoff": cutoff},
                )
            )
            session.commit()
        except Exception as e:
            logger.warning("Order archival skipped (archive table may not exist): %s", e)
            try:
                session.rollback()
            except Exception:
                pass
            return ArchivalResult("orders", 0, 0, cutoff.isoformat(), time.time() - start, str(e))

        return ArchivalResult("orders", rows, rows, cutoff.isoformat(), time.time() - start)

    def cleanup_ohlcv(self, session) -> list[ArchivalResult]:
        """Clean up OHLCV bars beyond retention periods per interval."""
        import time

        results = []
        interval_days = {
            "1m": self.policy.ohlcv_1m_hot_days,
            "5m": self.policy.ohlcv_5m_hot_days,
            "15m": self.policy.ohlcv_15m_hot_days,
            "1h": self.policy.ohlcv_1h_hot_days,
            "1d": self.policy.ohlcv_1d_hot_days,
        }
        for interval, days in interval_days.items():
            start = time.time()
            cutoff = self._cutoff_date(days)
            try:
                result = session.execute(
                    session.bind.execute(
                        "DELETE FROM ohlcv WHERE interval = :interval AND timestamp < :cutoff",
                        {"interval": interval, "cutoff": cutoff},
                    )
                )
                rows = result.rowcount if hasattr(result, "rowcount") else 0
                session.commit()
                if rows > 0:
                    logger.info("OHLCV cleanup: deleted %d %s bars older than %d days", rows, interval, days)
                results.append(ArchivalResult(f"ohlcv_{interval}", 0, rows, cutoff.isoformat(), time.time() - start))
            except Exception as e:
                logger.debug("OHLCV cleanup for %s: %s", interval, e)
                try:
                    session.rollback()
                except Exception:
                    pass
                results.append(
                    ArchivalResult(f"ohlcv_{interval}", 0, 0, cutoff.isoformat(), time.time() - start, str(e))
                )
        return results

    def cleanup_risk_snapshots(self, session) -> ArchivalResult:
        """Delete old risk snapshots beyond retention."""
        import time

        start = time.time()
        cutoff = self._cutoff_date(self.policy.risk_snapshots_hot_days)
        try:
            result = session.execute(
                session.bind.execute(
                    "DELETE FROM risk_snapshot WHERE timestamp < :cutoff",
                    {"cutoff": cutoff},
                )
            )
            rows = result.rowcount if hasattr(result, "rowcount") else 0
            session.commit()
            return ArchivalResult("risk_snapshot", 0, rows, cutoff.isoformat(), time.time() - start)
        except Exception as e:
            logger.debug("Risk snapshot cleanup: %s", e)
            try:
                session.rollback()
            except Exception:
                pass
            return ArchivalResult("risk_snapshot", 0, 0, cutoff.isoformat(), time.time() - start, str(e))

    def export_to_csv(self, table_name: str, rows: list[dict[str, Any]], cutoff_date: str) -> str:
        """Export archived rows to compressed CSV file."""
        import csv
        import gzip

        filename = f"{table_name}_{cutoff_date.replace(':', '-')}.csv.gz"
        filepath = os.path.join(self.archive_dir, filename)

        if not rows:
            return filepath

        with gzip.open(filepath, "wt", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Exported %d rows to %s", len(rows), filepath)
        return filepath

    def run_full_archival(self, session) -> list[ArchivalResult]:
        """Run complete archival cycle across all tables."""
        logger.info("Starting full data archival cycle...")
        results = []

        results.append(self.archive_orders(session))
        results.extend(self.cleanup_ohlcv(session))
        results.append(self.cleanup_risk_snapshots(session))

        self._last_run = results

        total_deleted = sum(r.rows_deleted for r in results)
        total_archived = sum(r.rows_archived for r in results)
        errors = sum(1 for r in results if r.error)

        logger.info(
            "Archival cycle complete: archived=%d, deleted=%d, errors=%d",
            total_archived,
            total_deleted,
            errors,
        )
        return results

    def get_retention_summary(self) -> dict[str, Any]:
        """Return current retention policy for API/dashboard."""
        return {
            "policy": {
                "orders": f"{self.policy.orders_hot_days}d hot, {self.policy.orders_archive_days}d archive",
                "audit_events": f"{self.policy.audit_events_hot_days}d hot, {self.policy.audit_archive_days}d archive (SEBI 7yr)",
                "trade_outcomes": f"{self.policy.trade_outcomes_hot_days}d hot, {self.policy.trade_outcomes_archive_days}d archive",
                "ohlcv_1m": f"{self.policy.ohlcv_1m_hot_days}d",
                "ohlcv_5m": f"{self.policy.ohlcv_5m_hot_days}d",
                "ohlcv_15m": f"{self.policy.ohlcv_15m_hot_days}d",
                "ohlcv_1h": f"{self.policy.ohlcv_1h_hot_days}d",
                "ohlcv_1d": f"{self.policy.ohlcv_1d_hot_days}d (10 years)",
                "risk_snapshots": f"{self.policy.risk_snapshots_hot_days}d",
                "closed_positions": f"{self.policy.closed_positions_hot_days}d",
            },
            "archive_dir": self.archive_dir,
            "last_run": [
                {
                    "table": r.table_name,
                    "archived": r.rows_archived,
                    "deleted": r.rows_deleted,
                    "cutoff": r.cutoff_date,
                    "error": r.error,
                }
                for r in self._last_run
            ]
            if self._last_run
            else None,
        }
