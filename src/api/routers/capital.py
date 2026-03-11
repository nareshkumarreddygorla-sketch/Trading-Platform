"""
Capital deployment gate: GET /capital/validate.
If validate() fails, autonomous live mode must not be enabled; manual paper still allowed.
"""
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from src.api.auth import get_current_user

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class CapitalValidateResponse(BaseModel):
    ok: bool
    checks: Dict[str, Any]
    message: Optional[str] = None


router = APIRouter()


@router.get("/capital/validate", response_model=CapitalValidateResponse)
async def capital_validate(request: Request, current_user: dict = Depends(get_current_user)):
    """
    Run capital gate checks (Redis, broker, market data, stress/restart).
    If ok is false, do not enable autonomous live mode. Manual paper mode still allowed.
    """
    gate = getattr(request.app.state, "capital_gate", None)
    if gate is None:
        # No capital gate configured -- fall back to checking real services directly
        checks = {}
        all_ok = True

        # Redis
        try:
            import redis.asyncio as aioredis
            import os
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            r = aioredis.from_url(redis_url, decode_responses=True)
            await r.ping()
            await r.aclose()
            checks["redis"] = "ok"
        except Exception as e:
            checks["redis"] = f"unavailable: {e}"
            all_ok = False

        # Broker gateway
        gateway = getattr(request.app.state, "gateway", None)
        if gateway is not None:
            checks["broker"] = "ok" if not getattr(gateway, "_auth_failed", False) else "auth_failed"
            if checks["broker"] != "ok":
                all_ok = False
        else:
            checks["broker"] = "not configured"

        # Market data
        mds = getattr(request.app.state, "market_data_service", None)
        if mds is not None:
            try:
                checks["market_data"] = "ok" if mds.is_healthy() else "degraded"
                if checks["market_data"] != "ok":
                    all_ok = False
            except Exception:
                checks["market_data"] = "error"
                all_ok = False
        else:
            yf = getattr(request.app.state, "yf_feeder", None)
            if yf and getattr(yf, "_running", False):
                checks["market_data"] = "ok (yfinance fallback)"
            else:
                checks["market_data"] = "not configured"

        # Risk manager equity
        rm = getattr(request.app.state, "risk_manager", None)
        if rm is not None:
            checks["equity"] = round(rm.equity, 2)
            checks["daily_pnl"] = round(rm.daily_pnl, 2)
            checks["circuit_open"] = rm.is_circuit_open()
            if rm.is_circuit_open():
                all_ok = False
        else:
            checks["risk_manager"] = "not configured"
            all_ok = False

        return {
            "ok": all_ok,
            "checks": checks,
            "message": "Capital gate not configured; checked services directly",
        }
    return await gate.validate()
