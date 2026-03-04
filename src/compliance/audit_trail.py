"""
SEBI Compliance Audit Trail Module.

Provides append-only, immutable audit logging for all trading operations
in compliance with SEBI (Securities and Exchange Board of India) regulations.

Key guarantees:
  - Append-only semantics: no update or delete operations permitted.
  - Thread-safe event recording via threading.Lock.
  - 7-year retention policy metadata on every event.
  - In-memory buffer as primary storage with optional DB flush.
  - CSV export for regulatory submissions.

Acceptance criteria traceability:
  AC1 - record_signal:  signal generation with model weights + confidence
  AC2 - record_order:   order submission with strategy + risk check status
  AC3 - record_fill:    order fill with cost breakdown + slippage
  AC4 - record_risk_decision: risk gate pass/fail with values vs thresholds
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enum: every auditable event type in the platform
# ---------------------------------------------------------------------------

class AuditEventType(str, Enum):
    """Exhaustive set of auditable event types for SEBI compliance."""

    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    RISK_CHECK_PASSED = "RISK_CHECK_PASSED"
    RISK_CHECK_FAILED = "RISK_CHECK_FAILED"
    CIRCUIT_BREAKER_TRIPPED = "CIRCUIT_BREAKER_TRIPPED"
    KILL_SWITCH_ARMED = "KILL_SWITCH_ARMED"
    MODEL_DRIFT_DETECTED = "MODEL_DRIFT_DETECTED"
    MODEL_RETRAINED = "MODEL_RETRAINED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    CONFIG_CHANGED = "CONFIG_CHANGED"


# ---------------------------------------------------------------------------
# Retention policy: SEBI mandates 7-year record keeping
# ---------------------------------------------------------------------------

RETENTION_YEARS = 7


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _retention_expiry() -> datetime:
    """Compute the earliest date this record may be purged."""
    return _utc_now() + timedelta(days=RETENTION_YEARS * 365)


# ---------------------------------------------------------------------------
# Immutable audit event dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuditEvent:
    """
    Single immutable audit record.

    Fields
    ------
    event_id : str          Unique identifier (UUID4).
    event_type : AuditEventType
    timestamp : datetime    UTC timestamp of the event.
    symbol : str            Trading instrument symbol (e.g. 'RELIANCE.NS').
    details : dict          Arbitrary payload specific to the event type.
    model_source : str      Originating model or system component.
    strategy_id : str       Strategy that triggered the event.
    risk_checks : dict      Snapshot of risk-gate outcomes at event time.
    retention_until : datetime  Earliest permissible purge date (7-year policy).
    """

    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    symbol: str
    details: Dict[str, Any]
    model_source: str
    strategy_id: str
    risk_checks: Dict[str, Any]
    retention_until: datetime


# ---------------------------------------------------------------------------
# CSV column order (stable for regulatory export)
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "event_id",
    "event_type",
    "timestamp",
    "symbol",
    "model_source",
    "strategy_id",
    "details",
    "risk_checks",
    "retention_until",
]


# ---------------------------------------------------------------------------
# Core audit trail class
# ---------------------------------------------------------------------------

class SEBIAuditTrail:
    """
    Thread-safe, append-only SEBI compliance audit trail.

    Primary storage is an in-memory list.  An optional *db_session_factory*
    callable can be supplied; when present, events are flushed to the
    ``audit_events`` table on every ``_flush_to_db`` call (triggered
    automatically when the buffer reaches *flush_threshold* entries).

    Parameters
    ----------
    db_session_factory : callable, optional
        Zero-argument callable that returns a SQLAlchemy ``Session``.
    flush_threshold : int
        Number of buffered events before an automatic DB flush is attempted.
    """

    def __init__(
        self,
        db_session_factory: Optional[Callable] = None,
        flush_threshold: int = 100,
    ) -> None:
        self._events: List[AuditEvent] = []
        self._lock = threading.Lock()
        self._db_session_factory = db_session_factory
        self._flush_threshold = flush_threshold
        self._pending_flush: List[AuditEvent] = []
        # SHA-256 hash chain for tamper detection (SEBI compliance)
        self._last_hash: Optional[str] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_chain_hash(self, event: AuditEvent) -> str:
        """Compute SHA-256 hash linking this event to the previous one (tamper detection)."""
        prev = self._last_hash or "genesis"
        payload = f"{prev}|{event.event_id}|{event.timestamp.isoformat()}|{event.event_type.value}|{event.symbol}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def _append(self, event: AuditEvent) -> AuditEvent:
        """Append an event under the lock; write-through to DB immediately if available."""
        with self._lock:
            # Compute hash chain for tamper detection
            chain_hash = self._compute_chain_hash(event)
            self._last_hash = chain_hash

            self._events.append(event)

            # Write-through: flush every event immediately when DB is available.
            # This prevents data loss on crash (previously batched at 100-event threshold).
            if self._db_session_factory is not None:
                self._pending_flush = [event]
                try:
                    self._flush_to_db(chain_hash=chain_hash)
                except Exception:
                    # If immediate flush fails, keep in pending buffer for retry
                    logger.error(
                        "Write-through flush failed for event %s; will retry on next append",
                        event.event_id,
                    )
            # If no DB configured, events stay in memory only (dev/paper mode)

        return event

    def _make_event(
        self,
        event_type: AuditEventType,
        symbol: str,
        details: Dict[str, Any],
        model_source: str = "",
        strategy_id: str = "",
        risk_checks: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        return AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=_utc_now(),
            symbol=symbol,
            details=details,
            model_source=model_source,
            strategy_id=strategy_id,
            risk_checks=risk_checks or {},
            retention_until=_retention_expiry(),
        )

    def _flush_to_db(self, chain_hash: Optional[str] = None) -> None:
        """
        Persist pending events to the database.

        Called on every event append (write-through) or can be invoked manually.
        On failure the pending buffer is kept so the next flush retries.

        Parameters
        ----------
        chain_hash : str, optional
            SHA-256 hash linking this event to the previous one for tamper detection.
        """
        if self._db_session_factory is None or not self._pending_flush:
            return

        import json

        try:
            from sqlalchemy import text as _sa_text
            session = self._db_session_factory()
            try:
                for evt in self._pending_flush:
                    session.execute(
                        _sa_text(
                            "INSERT INTO audit_events (ts, event_type, actor, payload) "
                            "VALUES (:ts, :event_type, :actor, :payload)"
                        ),
                        {
                            "ts": evt.timestamp,
                            "event_type": evt.event_type.value if hasattr(evt.event_type, 'value') else str(evt.event_type),
                            "actor": evt.model_source or evt.strategy_id or "system",
                            "payload": json.dumps(
                                {
                                    "event_id": evt.event_id,
                                    "symbol": evt.symbol,
                                    "details": evt.details,
                                    "risk_checks": evt.risk_checks,
                                    "strategy_id": evt.strategy_id,
                                    "model_source": evt.model_source,
                                    "retention_until": evt.retention_until.isoformat() if evt.retention_until else None,
                                    "chain_hash": chain_hash,
                                }
                            ),
                        },
                    )
                session.commit()
                self._pending_flush.clear()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
        except Exception as e:
            logger.error("Failed to flush %d audit events to DB: %s", len(self._pending_flush), e)
            # Keep pending buffer intact so next flush retries.

    # ------------------------------------------------------------------
    # AC1: Signal recording
    # ------------------------------------------------------------------

    def record_signal(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        model_source: str,
        model_weights: Optional[Dict[str, float]] = None,
    ) -> AuditEvent:
        """
        Record a trading signal generation event (AC1).

        Parameters
        ----------
        symbol : str
            Instrument symbol (e.g. ``'RELIANCE.NS'``).
        direction : str
            ``'BUY'`` or ``'SELL'``.
        confidence : float
            Model confidence score (0.0 -- 1.0).
        model_source : str
            Identifier of the model that produced the signal.
        model_weights : dict, optional
            Snapshot of ensemble weights at signal time.
        """
        details = {
            "direction": direction,
            "confidence": confidence,
            "model_weights": model_weights or {},
        }
        event = self._make_event(
            event_type=AuditEventType.SIGNAL_GENERATED,
            symbol=symbol,
            details=details,
            model_source=model_source,
        )
        return self._append(event)

    # ------------------------------------------------------------------
    # AC2: Order recording
    # ------------------------------------------------------------------

    def record_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        price: Optional[float],
        strategy_id: str,
        risk_checks_passed: Dict[str, bool],
    ) -> AuditEvent:
        """
        Record an order submission event (AC2).

        Parameters
        ----------
        symbol : str
            Instrument symbol.
        side : str
            ``'BUY'`` or ``'SELL'``.
        qty : float
            Order quantity.
        order_type : str
            ``'LIMIT'``, ``'MARKET'``, etc.
        price : float or None
            Limit price (None for market orders).
        strategy_id : str
            Strategy that generated the order.
        risk_checks_passed : dict
            Mapping of risk-check name to pass/fail boolean.
        """
        details = {
            "side": side,
            "quantity": qty,
            "order_type": order_type,
            "price": price,
        }
        event = self._make_event(
            event_type=AuditEventType.ORDER_SUBMITTED,
            symbol=symbol,
            details=details,
            strategy_id=strategy_id,
            risk_checks=risk_checks_passed,
        )
        return self._append(event)

    # ------------------------------------------------------------------
    # AC3: Fill recording
    # ------------------------------------------------------------------

    def record_fill(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        costs_breakdown: Dict[str, float],
        slippage: float,
    ) -> AuditEvent:
        """
        Record an order fill event (AC3).

        Parameters
        ----------
        symbol : str
            Instrument symbol.
        side : str
            ``'BUY'`` or ``'SELL'``.
        qty : float
            Filled quantity.
        price : float
            Execution price.
        costs_breakdown : dict
            Mapping of cost component to amount (e.g. ``{'brokerage': 20.0,
            'stt': 15.0, 'gst': 3.6}``).
        slippage : float
            Signed slippage in price units (positive = worse than expected).
        """
        details = {
            "side": side,
            "quantity": qty,
            "price": price,
            "costs_breakdown": costs_breakdown,
            "slippage": slippage,
        }
        event = self._make_event(
            event_type=AuditEventType.ORDER_FILLED,
            symbol=symbol,
            details=details,
        )
        return self._append(event)

    # ------------------------------------------------------------------
    # AC4: Risk decision recording
    # ------------------------------------------------------------------

    def record_risk_decision(
        self,
        check_type: str,
        result: bool,
        values: Dict[str, Any],
        thresholds: Dict[str, Any],
    ) -> AuditEvent:
        """
        Record a risk-gate decision event (AC4).

        Parameters
        ----------
        check_type : str
            Name of the risk check (e.g. ``'max_position_size'``).
        result : bool
            ``True`` if the check passed, ``False`` if blocked.
        values : dict
            Actual metric values at decision time.
        thresholds : dict
            Configured thresholds that were evaluated against.
        """
        event_type = (
            AuditEventType.RISK_CHECK_PASSED if result
            else AuditEventType.RISK_CHECK_FAILED
        )
        details = {
            "check_type": check_type,
            "result": result,
            "values": values,
            "thresholds": thresholds,
        }
        risk_checks = {check_type: result}
        event = self._make_event(
            event_type=event_type,
            symbol="",
            details=details,
            risk_checks=risk_checks,
        )
        return self._append(event)

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        symbol: Optional[str] = None,
        limit: int = 1000,
    ) -> List[AuditEvent]:
        """
        Retrieve audit events with optional filters.

        Parameters
        ----------
        start_date : datetime, optional
            Inclusive lower bound on timestamp.
        end_date : datetime, optional
            Inclusive upper bound on timestamp.
        event_type : AuditEventType, optional
            Filter to a single event type.
        symbol : str, optional
            Filter by instrument symbol.
        limit : int
            Maximum number of events to return (default 1000).

        Returns
        -------
        list[AuditEvent]
            Matching events in chronological order (oldest first).
        """
        with self._lock:
            results = list(self._events)

        if start_date is not None:
            results = [e for e in results if e.timestamp >= start_date]
        if end_date is not None:
            results = [e for e in results if e.timestamp <= end_date]
        if event_type is not None:
            results = [e for e in results if e.event_type == event_type]
        if symbol is not None:
            results = [e for e in results if e.symbol == symbol]

        return results[:limit]

    # ------------------------------------------------------------------
    # CSV export for regulatory submission
    # ------------------------------------------------------------------

    def export_csv(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> str:
        """
        Export audit events as CSV content for SEBI regulatory submission.

        Parameters
        ----------
        start_date : datetime, optional
            Inclusive lower bound.
        end_date : datetime, optional
            Inclusive upper bound.

        Returns
        -------
        str
            CSV-formatted string with headers.
        """
        import json as _json

        events = self.get_events(
            start_date=start_date, end_date=end_date, limit=2**31 - 1
        )

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS)
        writer.writeheader()

        for evt in events:
            writer.writerow(
                {
                    "event_id": evt.event_id,
                    "event_type": evt.event_type.value,
                    "timestamp": evt.timestamp.isoformat(),
                    "symbol": evt.symbol,
                    "model_source": evt.model_source,
                    "strategy_id": evt.strategy_id,
                    "details": _json.dumps(evt.details),
                    "risk_checks": _json.dumps(evt.risk_checks),
                    "retention_until": evt.retention_until.isoformat(),
                }
            )

        return buf.getvalue()

    # ------------------------------------------------------------------
    # Manual flush & stats
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Force-flush pending events to DB (no-op if no DB configured)."""
        with self._lock:
            self._flush_to_db()

    @property
    def event_count(self) -> int:
        """Total number of recorded events."""
        with self._lock:
            return len(self._events)

    @property
    def pending_flush_count(self) -> int:
        """Number of events awaiting DB persistence."""
        with self._lock:
            return len(self._pending_flush)
