"""Reconciliation: broker vs DB positions. Log discrepancies; no auto-correct."""
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List

from src.persistence.reconciliation import reconcile_positions, ReconciliationResult
from src.monitoring.metrics import track_reconciliation_mismatches_total

router = APIRouter()


class ReconciliationResponse(BaseModel):
    in_sync: bool
    mismatches: List[str]
    ts: str


@router.post("/reconcile/positions", response_model=ReconciliationResponse)
async def run_reconcile_positions(request: Request):
    """
    Compare broker positions vs internal DB. Logs discrepancies and increments
    reconciliation_mismatches_total. Does NOT auto-correct.
    """
    order_entry = getattr(request.app.state, "order_entry_service", None)
    position_repo = getattr(request.app.state, "position_repo", None)
    if order_entry is None or position_repo is None:
        return ReconciliationResponse(in_sync=True, mismatches=["persistence_or_gateway_not_configured"], ts="")
    gateway = getattr(order_entry.order_router, "default_gateway", None)
    if gateway is None:
        return ReconciliationResponse(in_sync=True, mismatches=["gateway_not_available"], ts="")

    def on_mismatch_count(n: int):
        track_reconciliation_mismatches_total(n)

    result: ReconciliationResult = await reconcile_positions(
        gateway.get_positions,
        position_repo,
        on_mismatch_count,
    )
    return ReconciliationResponse(
        in_sync=result.in_sync,
        mismatches=result.mismatches,
        ts=result.ts.isoformat() if result.ts else "",
    )
