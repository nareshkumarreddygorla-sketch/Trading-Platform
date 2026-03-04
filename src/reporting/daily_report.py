"""
Daily performance report generator.

Designed to run at 16:00 IST daily after market close.
Aggregates trade results, cost breakdowns, risk events, and rolling metrics
into a single DailyReport snapshot suitable for persistence and downstream analytics.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CostSummary:
    """Aggregated transaction costs for the day, mirroring india_costs.CostBreakdown."""
    stt: float = 0.0
    brokerage: float = 0.0
    exchange_charges: float = 0.0
    sebi_fee: float = 0.0
    gst: float = 0.0
    stamp_duty: float = 0.0

    @property
    def total(self) -> float:
        return self.stt + self.brokerage + self.exchange_charges + self.sebi_fee + self.gst + self.stamp_duty

    def as_dict(self) -> dict:
        return {
            "stt": round(self.stt, 4),
            "brokerage": round(self.brokerage, 4),
            "exchange_charges": round(self.exchange_charges, 4),
            "sebi_fee": round(self.sebi_fee, 4),
            "gst": round(self.gst, 4),
            "stamp_duty": round(self.stamp_duty, 4),
            "total": round(self.total, 4),
        }


@dataclass
class DailyReport:
    """Complete daily performance snapshot."""

    # Identification
    date: str  # ISO format YYYY-MM-DD

    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L
    gross_pnl: float = 0.0
    total_costs: CostSummary = field(default_factory=CostSummary)
    net_pnl: float = 0.0

    # Risk metrics
    sharpe_ratio_20d: float = 0.0
    max_drawdown_pct: float = 0.0

    # Equity
    equity: float = 0.0
    equity_curve_point: float = 0.0

    # Activity counters
    signals_generated: int = 0
    orders_submitted: int = 0
    orders_rejected: int = 0

    # Risk events captured during the session
    risk_events: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SQLAlchemy model for daily_snapshots table
# ---------------------------------------------------------------------------

def _get_daily_snapshot_model():
    """
    Lazily import / define the DailySnapshotModel so the module can be imported
    without a live database or SQLAlchemy dependency at import time.
    """
    from sqlalchemy import Column, DateTime, Float, Integer, String, Text
    from src.persistence.models import Base

    class DailySnapshotModel(Base):
        __tablename__ = "daily_snapshots"

        id = Column(Integer, primary_key=True, autoincrement=True)
        date = Column(String(10), unique=True, nullable=False, index=True)

        total_trades = Column(Integer, nullable=False, default=0)
        winning_trades = Column(Integer, nullable=False, default=0)
        losing_trades = Column(Integer, nullable=False, default=0)
        win_rate = Column(Float, nullable=False, default=0.0)

        gross_pnl = Column(Float, nullable=False, default=0.0)
        net_pnl = Column(Float, nullable=False, default=0.0)

        # Cost breakdown stored as individual columns
        cost_stt = Column(Float, nullable=False, default=0.0)
        cost_brokerage = Column(Float, nullable=False, default=0.0)
        cost_exchange_charges = Column(Float, nullable=False, default=0.0)
        cost_sebi_fee = Column(Float, nullable=False, default=0.0)
        cost_gst = Column(Float, nullable=False, default=0.0)
        cost_stamp_duty = Column(Float, nullable=False, default=0.0)
        cost_total = Column(Float, nullable=False, default=0.0)

        sharpe_ratio_20d = Column(Float, nullable=False, default=0.0)
        max_drawdown_pct = Column(Float, nullable=False, default=0.0)

        equity = Column(Float, nullable=False, default=0.0)
        equity_curve_point = Column(Float, nullable=False, default=0.0)

        signals_generated = Column(Integer, nullable=False, default=0)
        orders_submitted = Column(Integer, nullable=False, default=0)
        orders_rejected = Column(Integer, nullable=False, default=0)

        # risk_events stored as JSON text
        risk_events_json = Column(Text, nullable=True, default="[]")

        created_at = Column(DateTime(timezone=True), nullable=False,
                            default=lambda: datetime.now(timezone.utc))

    return DailySnapshotModel


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class DailyReportGenerator:
    """
    Generates, persists, and retrieves daily performance reports.

    Intended lifecycle:
        1. Throughout the trading day, external code feeds trade results and events
           via ``record_trade``, ``record_signal``, ``record_order_submitted``,
           ``record_order_rejected``, ``record_risk_event``, and ``record_costs``.
        2. At 16:00 IST (after market close), call ``generate()`` to produce the
           DailyReport and optionally ``save_to_db()`` to persist it.

    Rolling Sharpe calculation keeps daily returns in memory so the 20-day
    window is always available without a database round-trip.
    """

    def __init__(
        self,
        risk_manager=None,
        persistence_service=None,
    ):
        # Optional dependencies --------------------------------------------------
        # risk_manager: src.risk_engine.manager.RiskManager (for live equity, positions)
        self._risk_manager = risk_manager
        # persistence_service: src.persistence.service.PersistenceService (for DB)
        self._persistence_service = persistence_service

        # In-memory accumulators (reset each day via ``_reset_daily``) -----------
        self._trades: List[Dict[str, Any]] = []
        self._cost_accumulator = CostSummary()
        self._signals_generated: int = 0
        self._orders_submitted: int = 0
        self._orders_rejected: int = 0
        self._risk_events: List[Dict[str, Any]] = []

        # Rolling history (persists across days for Sharpe / drawdown) -----------
        self._daily_returns: List[float] = []
        self._equity_curve: List[float] = []

        # Lazy-loaded model reference
        self._DailySnapshotModel = None

        logger.info("DailyReportGenerator initialised")

    # ------------------------------------------------------------------
    # Recording helpers (called during the trading day)
    # ------------------------------------------------------------------

    def record_trade(self, pnl: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a completed trade with its realised P&L."""
        self._trades.append({"pnl": pnl, "metadata": metadata or {}})

    def record_signal(self, count: int = 1) -> None:
        self._signals_generated += count

    def record_order_submitted(self, count: int = 1) -> None:
        self._orders_submitted += count

    def record_order_rejected(self, count: int = 1) -> None:
        self._orders_rejected += count

    def record_risk_event(self, event: Dict[str, Any]) -> None:
        """Record a risk event (circuit breaker, limit breach, etc.)."""
        self._risk_events.append(event)

    def record_costs(self, cost_breakdown) -> None:
        """
        Accumulate costs from a ``CostBreakdown`` (src.costs.india_costs).
        Accepts any object with stt, brokerage, exchange_charges, sebi_fee, gst, stamp_duty attributes.
        """
        self._cost_accumulator.stt += getattr(cost_breakdown, "stt", 0.0)
        self._cost_accumulator.brokerage += getattr(cost_breakdown, "brokerage", 0.0)
        self._cost_accumulator.exchange_charges += getattr(cost_breakdown, "exchange_charges", 0.0)
        self._cost_accumulator.sebi_fee += getattr(cost_breakdown, "sebi_fee", 0.0)
        self._cost_accumulator.gst += getattr(cost_breakdown, "gst", 0.0)
        self._cost_accumulator.stamp_duty += getattr(cost_breakdown, "stamp_duty", 0.0)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate(self, date: Optional[str] = None) -> DailyReport:
        """
        Generate the daily report from accumulated data.

        Args:
            date: ISO date string (``YYYY-MM-DD``). Defaults to today (UTC).

        Returns:
            Fully populated ``DailyReport`` dataclass instance.
        """
        report_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        logger.info("Generating daily report for %s", report_date)

        # Trade statistics -------------------------------------------------------
        total_trades = len(self._trades)
        winning_trades = sum(1 for t in self._trades if t["pnl"] > 0)
        losing_trades = sum(1 for t in self._trades if t["pnl"] < 0)
        win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_pnl = sum(t["pnl"] for t in self._trades)
        net_pnl = gross_pnl - self._cost_accumulator.total

        # Equity (prefer live from RiskManager, fall back to curve extrapolation) -
        equity = 0.0
        if self._risk_manager is not None:
            equity = self._risk_manager.equity
        elif self._equity_curve:
            equity = self._equity_curve[-1]

        # Update rolling history -------------------------------------------------
        if equity > 0:
            if self._equity_curve:
                prev_equity = self._equity_curve[-1]
                daily_return = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            else:
                daily_return = 0.0
            self._daily_returns.append(daily_return)
            self._equity_curve.append(equity)
        else:
            # If no equity info, derive return from net P&L relative to last known equity
            if self._equity_curve:
                prev = self._equity_curve[-1]
                daily_return = net_pnl / prev if prev > 0 else 0.0
                self._daily_returns.append(daily_return)
                self._equity_curve.append(prev + net_pnl)
            else:
                self._daily_returns.append(0.0)

        equity_curve_point = self._equity_curve[-1] if self._equity_curve else equity

        # Rolling risk metrics ---------------------------------------------------
        sharpe_20d = self._calculate_sharpe(self._daily_returns, window=20)
        max_dd_pct = self._calculate_max_drawdown(self._equity_curve)

        report = DailyReport(
            date=report_date,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate, 2),
            gross_pnl=round(gross_pnl, 2),
            total_costs=CostSummary(
                stt=self._cost_accumulator.stt,
                brokerage=self._cost_accumulator.brokerage,
                exchange_charges=self._cost_accumulator.exchange_charges,
                sebi_fee=self._cost_accumulator.sebi_fee,
                gst=self._cost_accumulator.gst,
                stamp_duty=self._cost_accumulator.stamp_duty,
            ),
            net_pnl=round(net_pnl, 2),
            sharpe_ratio_20d=round(sharpe_20d, 4),
            max_drawdown_pct=round(max_dd_pct, 4),
            equity=round(equity, 2),
            equity_curve_point=round(equity_curve_point, 2),
            signals_generated=self._signals_generated,
            orders_submitted=self._orders_submitted,
            orders_rejected=self._orders_rejected,
            risk_events=list(self._risk_events),
        )

        logger.info(
            "Daily report %s: trades=%d win_rate=%.1f%% net_pnl=%.2f sharpe_20d=%.4f max_dd=%.2f%%",
            report_date, total_trades, win_rate, net_pnl, sharpe_20d, max_dd_pct,
        )

        # Reset intraday accumulators for next session ---------------------------
        self._reset_daily()

        return report

    def _reset_daily(self) -> None:
        """Clear intraday accumulators; rolling history is preserved."""
        self._trades.clear()
        self._cost_accumulator = CostSummary()
        self._signals_generated = 0
        self._orders_submitted = 0
        self._orders_rejected = 0
        self._risk_events.clear()

    # ------------------------------------------------------------------
    # Rolling metrics
    # ------------------------------------------------------------------

    def _calculate_sharpe(self, daily_returns: List[float], window: int = 20) -> float:
        """
        Calculate annualised Sharpe ratio over the most recent ``window`` trading days.
        Risk-free rate assumed zero.

        Returns 0.0 if insufficient data or zero volatility.
        """
        if len(daily_returns) < 2:
            return 0.0
        returns = daily_returns[-window:]
        arr = np.array(returns, dtype=np.float64)
        std = float(np.std(arr, ddof=1))
        if std < 1e-12:
            return 0.0
        mean_ret = float(np.mean(arr))
        # Annualise: sqrt(252) scaling
        sharpe = (mean_ret / std) * math.sqrt(252)
        return sharpe

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """
        Calculate maximum drawdown percentage from the equity curve.

        Returns a non-negative value representing the worst peak-to-trough
        decline as a percentage (e.g. 5.2 means -5.2%).
        """
        if len(equity_curve) < 2:
            return 0.0
        arr = np.array(equity_curve, dtype=np.float64)
        cummax = np.maximum.accumulate(arr)
        drawdowns = (arr - cummax) / (cummax + 1e-12)
        max_dd = float(np.min(drawdowns))  # most negative value
        return abs(max_dd) * 100.0  # return as positive percentage

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazy-load the SQLAlchemy model (avoids import-time DB dependency)."""
        if self._DailySnapshotModel is None:
            self._DailySnapshotModel = _get_daily_snapshot_model()
        return self._DailySnapshotModel

    def save_to_db(self, report: DailyReport) -> None:
        """
        Save report to the ``daily_snapshots`` table.
        Upserts on (date) so re-running for the same date overwrites.
        """
        from src.persistence.database import session_scope

        Model = self._get_model()

        def _risk_events_json(events: List[Dict[str, Any]]) -> str:
            try:
                return json.dumps(events, default=str)
            except (TypeError, ValueError):
                logger.warning("Failed to serialise risk_events to JSON; storing empty list")
                return "[]"

        try:
            with session_scope() as session:
                existing = session.query(Model).filter(Model.date == report.date).first()
                if existing:
                    # Update in place
                    existing.total_trades = report.total_trades
                    existing.winning_trades = report.winning_trades
                    existing.losing_trades = report.losing_trades
                    existing.win_rate = report.win_rate
                    existing.gross_pnl = report.gross_pnl
                    existing.net_pnl = report.net_pnl
                    existing.cost_stt = report.total_costs.stt
                    existing.cost_brokerage = report.total_costs.brokerage
                    existing.cost_exchange_charges = report.total_costs.exchange_charges
                    existing.cost_sebi_fee = report.total_costs.sebi_fee
                    existing.cost_gst = report.total_costs.gst
                    existing.cost_stamp_duty = report.total_costs.stamp_duty
                    existing.cost_total = report.total_costs.total
                    existing.sharpe_ratio_20d = report.sharpe_ratio_20d
                    existing.max_drawdown_pct = report.max_drawdown_pct
                    existing.equity = report.equity
                    existing.equity_curve_point = report.equity_curve_point
                    existing.signals_generated = report.signals_generated
                    existing.orders_submitted = report.orders_submitted
                    existing.orders_rejected = report.orders_rejected
                    existing.risk_events_json = _risk_events_json(report.risk_events)
                    logger.info("Updated daily_snapshots for %s", report.date)
                else:
                    row = Model(
                        date=report.date,
                        total_trades=report.total_trades,
                        winning_trades=report.winning_trades,
                        losing_trades=report.losing_trades,
                        win_rate=report.win_rate,
                        gross_pnl=report.gross_pnl,
                        net_pnl=report.net_pnl,
                        cost_stt=report.total_costs.stt,
                        cost_brokerage=report.total_costs.brokerage,
                        cost_exchange_charges=report.total_costs.exchange_charges,
                        cost_sebi_fee=report.total_costs.sebi_fee,
                        cost_gst=report.total_costs.gst,
                        cost_stamp_duty=report.total_costs.stamp_duty,
                        cost_total=report.total_costs.total,
                        sharpe_ratio_20d=report.sharpe_ratio_20d,
                        max_drawdown_pct=report.max_drawdown_pct,
                        equity=report.equity,
                        equity_curve_point=report.equity_curve_point,
                        signals_generated=report.signals_generated,
                        orders_submitted=report.orders_submitted,
                        orders_rejected=report.orders_rejected,
                        risk_events_json=_risk_events_json(report.risk_events),
                    )
                    session.add(row)
                    logger.info("Inserted daily_snapshots for %s", report.date)
        except Exception:
            logger.exception("Failed to save daily report for %s", report.date)
            raise

    def get_report(self, date: str) -> Optional[DailyReport]:
        """Retrieve a single day's report from the database."""
        from src.persistence.database import session_scope

        Model = self._get_model()
        try:
            with session_scope() as session:
                row = session.query(Model).filter(Model.date == date).first()
                if row is None:
                    return None
                return self._row_to_report(row)
        except Exception:
            logger.exception("Failed to retrieve report for %s", date)
            return None

    def get_reports_range(self, start: str, end: str) -> List[DailyReport]:
        """Retrieve reports for a date range (inclusive) ordered chronologically."""
        from src.persistence.database import session_scope

        Model = self._get_model()
        try:
            with session_scope() as session:
                rows = (
                    session.query(Model)
                    .filter(Model.date >= start, Model.date <= end)
                    .order_by(Model.date)
                    .all()
                )
                return [self._row_to_report(r) for r in rows]
        except Exception:
            logger.exception("Failed to retrieve reports for range %s to %s", start, end)
            return []

    def _row_to_report(self, row) -> DailyReport:
        """Convert a DailySnapshotModel row to a DailyReport dataclass."""
        risk_events: List[Dict[str, Any]] = []
        if row.risk_events_json:
            try:
                risk_events = json.loads(row.risk_events_json)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Corrupt risk_events_json for date %s", row.date)

        return DailyReport(
            date=row.date,
            total_trades=row.total_trades,
            winning_trades=row.winning_trades,
            losing_trades=row.losing_trades,
            win_rate=row.win_rate,
            gross_pnl=row.gross_pnl,
            total_costs=CostSummary(
                stt=row.cost_stt,
                brokerage=row.cost_brokerage,
                exchange_charges=row.cost_exchange_charges,
                sebi_fee=row.cost_sebi_fee,
                gst=row.cost_gst,
                stamp_duty=row.cost_stamp_duty,
            ),
            net_pnl=row.net_pnl,
            sharpe_ratio_20d=row.sharpe_ratio_20d,
            max_drawdown_pct=row.max_drawdown_pct,
            equity=row.equity,
            equity_curve_point=row.equity_curve_point,
            signals_generated=row.signals_generated,
            orders_submitted=row.orders_submitted,
            orders_rejected=row.orders_rejected,
            risk_events=risk_events,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def to_dict(report: DailyReport) -> dict:
        """Convert a DailyReport to a JSON-serialisable dictionary."""
        return {
            "date": report.date,
            "total_trades": report.total_trades,
            "winning_trades": report.winning_trades,
            "losing_trades": report.losing_trades,
            "win_rate": report.win_rate,
            "gross_pnl": report.gross_pnl,
            "total_costs": report.total_costs.as_dict(),
            "net_pnl": report.net_pnl,
            "sharpe_ratio_20d": report.sharpe_ratio_20d,
            "max_drawdown_pct": report.max_drawdown_pct,
            "equity": report.equity,
            "equity_curve_point": report.equity_curve_point,
            "signals_generated": report.signals_generated,
            "orders_submitted": report.orders_submitted,
            "orders_rejected": report.orders_rejected,
            "risk_events": report.risk_events,
        }
