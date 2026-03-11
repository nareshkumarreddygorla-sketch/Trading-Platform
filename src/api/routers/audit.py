"""
Audit log API: read-only list of audit events and SEBI compliance endpoints.

Endpoints:
  - GET /audit/logs                - Recent audit events
  - GET /audit/otr-report          - Order-to-trade ratio report
  - GET /audit/surveillance-alerts - Recent surveillance alerts
  - GET /audit/regulatory-report   - Generate SEBI format regulatory report
  - GET /audit/algo-registry       - List registered algorithms with SEBI IDs
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from src.api.auth import get_current_user

router = APIRouter()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class AuditEventResponse(BaseModel):
    id: str
    ts: str
    event_type: str
    actor: str
    payload: Optional[dict] = None


class AuditLogsResponse(BaseModel):
    events: List[AuditEventResponse]


class OTRReportResponse(BaseModel):
    algo_reports: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    thresholds: Dict[str, float]


class SurveillanceAlertsResponse(BaseModel):
    alerts: List[Dict[str, Any]]
    summary: Dict[str, Any]


class RegulatoryReportResponse(BaseModel):
    report: Dict[str, Any]


class AlgoRegistryResponse(BaseModel):
    algorithms: List[Dict[str, Any]]
    total: int
    active: int



# ---------------------------------------------------------------------------
# Existing endpoint: audit logs
# ---------------------------------------------------------------------------

@router.get("/audit/logs", response_model=AuditLogsResponse)
async def get_audit_logs(
    request: Request,
    limit: int = 500,
    event_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    """
    List recent audit events (newest first). Returns real events from DB, or
    empty list with a message if audit repository is not configured.
    """
    audit_repo = getattr(request.app.state, "audit_repo", None)
    if audit_repo is not None:
        events = audit_repo.list_events(limit=min(limit, 1000), event_type=event_type or None)
        return AuditLogsResponse(
            events=[AuditEventResponse(**e) for e in events] if events else []
        )

    # No audit repository configured — return empty, not fake data
    return AuditLogsResponse(events=[])


# ---------------------------------------------------------------------------
# New endpoint: OTR report
# ---------------------------------------------------------------------------

@router.get("/audit/otr-report", response_model=OTRReportResponse)
async def get_otr_report(
    request: Request,
    algo_id: Optional[str] = Query(None, description="Filter by algorithm ID"),
    limit: int = Query(50, description="Max alerts to return"),
    current_user: dict = Depends(get_current_user),
):
    """
    Order-to-Trade Ratio report across all tracked algorithms.

    Returns per-algo OTR metrics for 1-minute, 5-minute, and 1-hour windows,
    along with recent OTR alerts/breaches.
    """
    otr_monitor = getattr(request.app.state, "otr_monitor", None)

    if otr_monitor is not None:
        reports = otr_monitor.get_all_otr_reports()
        if algo_id:
            reports = [r for r in reports if r.get("algo_id") == algo_id]

        alerts = otr_monitor.get_alerts(algo_id=algo_id, limit=limit)

        return OTRReportResponse(
            algo_reports=reports,
            alerts=alerts,
            thresholds={
                "warning": otr_monitor._warning_threshold,
                "halt": otr_monitor._halt_threshold,
            },
        )

    # No OTR monitor configured - return empty but valid response
    return OTRReportResponse(
        algo_reports=[],
        alerts=[],
        thresholds={"warning": 25.0, "halt": 50.0},
    )


# ---------------------------------------------------------------------------
# New endpoint: surveillance alerts
# ---------------------------------------------------------------------------

@router.get("/audit/surveillance-alerts", response_model=SurveillanceAlertsResponse)
async def get_surveillance_alerts(
    request: Request,
    pattern: Optional[str] = Query(None, description="Filter by pattern: LAYERING, SPOOFING, WASH_TRADE"),
    severity: Optional[str] = Query(None, description="Filter by severity: LOW, MEDIUM, HIGH, CRITICAL"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    unacknowledged_only: bool = Query(False, description="Only unacknowledged alerts"),
    limit: int = Query(100, description="Max alerts to return"),
    current_user: dict = Depends(get_current_user),
):
    """
    Recent post-trade surveillance alerts for potential market manipulation.

    Detects patterns including layering, spoofing, and wash trades.
    Each alert includes an evidence trail for regulatory review.
    """
    surveillance = getattr(request.app.state, "surveillance_engine", None)

    if surveillance is not None:
        from src.compliance.surveillance import ManipulationPattern, AlertSeverity

        pattern_enum = None
        severity_enum = None
        if pattern:
            try:
                pattern_enum = ManipulationPattern(pattern.upper())
            except ValueError:
                pass
        if severity:
            try:
                severity_enum = AlertSeverity(severity.upper())
            except ValueError:
                pass

        alerts = surveillance.get_alerts(
            pattern=pattern_enum,
            severity=severity_enum,
            symbol=symbol,
            limit=limit,
            unacknowledged_only=unacknowledged_only,
        )
        summary = surveillance.get_summary()

        return SurveillanceAlertsResponse(alerts=alerts, summary=summary)

    # No surveillance engine configured
    return SurveillanceAlertsResponse(
        alerts=[],
        summary={
            "total_alerts": 0,
            "unacknowledged": 0,
            "by_pattern": {},
            "by_severity": {},
            "thresholds": {},
        },
    )


# ---------------------------------------------------------------------------
# New endpoint: regulatory report
# ---------------------------------------------------------------------------

@router.get("/audit/regulatory-report", response_model=RegulatoryReportResponse)
async def get_regulatory_report(
    request: Request,
    report_type: str = Query("daily", description="Report type: daily or monthly"),
    current_user: dict = Depends(get_current_user),
):
    """
    Generate a SEBI-format regulatory report (daily or monthly).

    Includes order/trade summaries, OTR compliance, surveillance alerts,
    risk management statistics, and algo registry snapshot.
    """
    audit_trail = getattr(request.app.state, "audit_trail", None)

    if audit_trail is not None:
        report = audit_trail.generate_regulatory_report(report_type=report_type)
        return RegulatoryReportResponse(report=report)

    # No audit trail configured - return minimal report structure
    now = datetime.now(timezone.utc)
    return RegulatoryReportResponse(
        report={
            "report_metadata": {
                "report_type": report_type,
                "generated_at": now.isoformat(),
                "format": "SEBI_ALGO_TRADING_REPORT_V1",
            },
            "summary": {
                "total_events": 0,
                "total_orders": 0,
                "total_fills": 0,
                "message": "No audit trail configured. Connect SEBIAuditTrail to app.state.audit_trail.",
            },
            "otr_compliance": {"warnings": 0, "halts": 0},
            "surveillance": {"alerts_generated": 0},
            "algo_registry": {"total_registered": 0, "active": 0, "algorithms": []},
            "retention_compliance": {
                "min_retention_years": 5,
                "policy": "All records retained for minimum 5 years per SEBI mandate.",
            },
        }
    )


# ---------------------------------------------------------------------------
# New endpoint: algo registry
# ---------------------------------------------------------------------------

@router.get("/audit/algo-registry", response_model=AlgoRegistryResponse)
async def get_algo_registry(
    request: Request,
    active_only: bool = Query(False, description="Only show active algorithms"),
    algo_name: Optional[str] = Query(None, description="Filter by algorithm name"),
    current_user: dict = Depends(get_current_user),
):
    """
    List all registered algorithms with their SEBI IDs and version history.

    Each algorithm receives a unique SEBI-format ID upon registration.
    Version history tracks parameter changes over time.
    """
    audit_trail = getattr(request.app.state, "audit_trail", None)

    if audit_trail is not None:
        algorithms = audit_trail.get_algo_registry()

        if active_only:
            algorithms = [a for a in algorithms if a.get("is_active")]
        if algo_name:
            algorithms = [a for a in algorithms if a.get("algo_name") == algo_name]

        active_count = sum(1 for a in algorithms if a.get("is_active"))

        return AlgoRegistryResponse(
            algorithms=algorithms,
            total=len(algorithms),
            active=active_count,
        )

    # No audit trail configured
    return AlgoRegistryResponse(
        algorithms=[],
        total=0,
        active=0,
    )
