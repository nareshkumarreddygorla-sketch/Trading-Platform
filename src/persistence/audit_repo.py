"""Immutable audit log. Append-only. All critical actions traceable to actor."""
import json
import logging
from datetime import datetime, timezone
from typing import Any, List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from .database import session_scope
from .models import AuditEventModel

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_payload(payload_str: Optional[str]) -> Optional[dict]:
    if not payload_str:
        return None
    try:
        return json.loads(payload_str)
    except (json.JSONDecodeError, TypeError):
        return None


class AuditRepository:
    """Append-only audit. No update or delete."""

    def __init__(self, session_factory=None):
        from .database import get_session_factory
        self._session_factory = session_factory or get_session_factory()

    def append(
        self,
        event_type: str,
        actor: str,
        payload: Optional[dict] = None,
        tenant_id: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> None:
        """Append one audit event. Immutable."""
        payload_str = json.dumps(payload, default=str) if payload is not None else None
        row = AuditEventModel(
            ts=_utc_now(),
            event_type=event_type,
            actor=actor or "system",
            payload=payload_str,
            tenant_id=tenant_id,
        )

        def _add(sess: Session) -> None:
            sess.add(row)
            sess.flush()

        if session is not None:
            _add(session)
        else:
            with session_scope() as sess:
                _add(sess)

    def append_sync(
        self,
        event_type: str,
        actor: str,
        payload: Optional[dict] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Synchronous append for use from sync code or executor."""
        with session_scope() as session:
            self.append(event_type=event_type, actor=actor, payload=payload, tenant_id=tenant_id, session=session)

    def list_events(
        self,
        limit: int = 500,
        event_type: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> List[dict]:
        """Return recent audit events (newest first). For API GET /audit/logs."""
        with session_scope() as session:
            q = session.query(AuditEventModel)
            if event_type:
                q = q.filter(AuditEventModel.event_type == event_type)
            if tenant_id is not None:
                q = q.filter(AuditEventModel.tenant_id == tenant_id)
            q = q.order_by(desc(AuditEventModel.ts)).limit(limit)
            rows = q.all()
            return [
                {
                    "id": str(r.id),
                    "ts": r.ts.isoformat() if r.ts else "",
                    "event_type": r.event_type or "",
                    "actor": r.actor or "system",
                    "payload": _parse_payload(r.payload),
                }
                for r in rows
            ]
