"""Audit log API: read-only list of audit events."""
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, Request

from src.api.auth import get_current_user
from pydantic import BaseModel

router = APIRouter()


class AuditEventResponse(BaseModel):
    id: str
    ts: str
    event_type: str
    actor: str
    payload: Optional[dict] = None


class AuditLogsResponse(BaseModel):
    events: List[AuditEventResponse]


def _demo_events() -> List[AuditEventResponse]:
    """Return realistic demo audit events when no DB events exist."""
    now = datetime.now(timezone.utc)
    events = [
        {"event_type": "trade_executed", "actor": "ema_crossover", "payload": {"symbol": "RELIANCE.NS", "side": "BUY", "qty": 10, "price": 2845.50}},
        {"event_type": "trade_executed", "actor": "macd", "payload": {"symbol": "INFY.NS", "side": "BUY", "qty": 15, "price": 1578.25}},
        {"event_type": "risk_limit_hit", "actor": "risk_engine", "payload": {"metric": "max_daily_loss", "value": "1.8%", "limit": "2.0%"}},
        {"event_type": "strategy_disabled", "actor": "admin", "payload": {"strategy": "rsi", "reason": "performance_review"}},
        {"event_type": "trade_executed", "actor": "ema_crossover", "payload": {"symbol": "TCS.NS", "side": "SELL", "qty": 8, "price": 3920.75}},
        {"event_type": "model_retrained", "actor": "ai_engine", "payload": {"model": "alpha_v3", "accuracy": "71.2%", "features": 42}},
        {"event_type": "trade_executed", "actor": "rsi", "payload": {"symbol": "HDFCBANK.NS", "side": "BUY", "qty": 12, "price": 1650.30}},
        {"event_type": "risk_limit_hit", "actor": "risk_engine", "payload": {"metric": "var_95", "value": "1.2%", "limit": "1.5%"}},
        {"event_type": "strategy_disabled", "actor": "circuit_breaker", "payload": {"reason": "drawdown_limit", "drawdown": "3.5%"}},
        {"event_type": "trade_executed", "actor": "macd", "payload": {"symbol": "WIPRO.NS", "side": "SELL", "qty": 20, "price": 452.80}},
        {"event_type": "model_retrained", "actor": "ai_engine", "payload": {"model": "regime_classifier", "accuracy": "68.5%"}},
        {"event_type": "trade_executed", "actor": "ema_crossover", "payload": {"symbol": "SBIN.NS", "side": "BUY", "qty": 25, "price": 812.45}},
    ]
    result = []
    for i, evt in enumerate(events):
        ts = now - timedelta(minutes=i * 15 + 5)
        result.append(AuditEventResponse(
            id=str(uuid.uuid4()),
            ts=ts.isoformat(),
            event_type=evt["event_type"],
            actor=evt["actor"],
            payload=evt.get("payload"),
        ))
    return result


@router.get("/audit/logs", response_model=AuditLogsResponse)
async def get_audit_logs(
    request: Request,
    limit: int = 500,
    event_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    """
    List recent audit events (newest first). Uses DB if configured, otherwise demo data.
    """
    audit_repo = getattr(request.app.state, "audit_repo", None)
    if audit_repo is not None:
        events = audit_repo.list_events(limit=min(limit, 1000), event_type=event_type or None)
        if events:
            return AuditLogsResponse(
                events=[AuditEventResponse(**e) for e in events]
            )

    # Return demo data when no DB events exist
    demo = _demo_events()
    if event_type:
        demo = [e for e in demo if e.event_type == event_type]
    return AuditLogsResponse(events=demo[:limit])
