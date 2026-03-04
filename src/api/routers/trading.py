"""
Trading readiness: only 200 when execution path is wired and system is allowed to trade.
Used by K8s readiness probe and dashboards; 503 when kill switch armed or circuit open.
"""
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/trading/ready")
async def trading_ready(request: Request):
    """
    Returns 200 only when:
    - Cold start recovery complete (startup_lock released, recovery_complete True)
    - Not in safe mode (broker reachable at startup when live)
    - OrderEntryService is configured (execution path wired)
    - Kill switch is NOT armed (trading allowed)
    - Circuit breaker is NOT open (risk limits not tripped)
    - Risk manager equity > 0

    Use for K8s readiness probe; 503 otherwise.
    """
    if getattr(request.app.state, "startup_lock", False):
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "reason": "startup_recovery_pending",
                "message": "Cold start recovery in progress; trading not allowed",
            },
        )

    if not getattr(request.app.state, "recovery_complete", False):
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "reason": "recovery_not_complete",
                "message": "Startup recovery has not completed; trading not allowed",
            },
        )

    if getattr(request.app.state, "safe_mode", False):
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "reason": "safe_mode_broker_unreachable",
                "message": "Broker unreachable at startup; live trading disabled for capital safety",
            },
        )

    order_entry = getattr(request.app.state, "order_entry_service", None)
    kill_switch = getattr(request.app.state, "kill_switch", None)

    if order_entry is None:
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "reason": "order_entry_not_configured",
                "message": "Execution path not wired",
            },
        )

    risk_manager = getattr(order_entry, "risk_manager", None)
    if risk_manager is None:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "reason": "risk_manager_missing", "message": "Risk manager not attached"},
        )

    if kill_switch is not None:
        state = await kill_switch.get_state()
        if getattr(state, "armed", False):
            return JSONResponse(
                status_code=503,
                content={
                    "ready": False,
                    "reason": "kill_switch_armed",
                    "message": "Trading halted by kill switch",
                },
            )

    if risk_manager.is_circuit_open():
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "reason": "circuit_breaker_open",
                "message": "Circuit breaker tripped; new orders blocked",
            },
        )

    if getattr(risk_manager, "equity", 0) <= 0:
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "reason": "zero_equity",
                "message": "Equity must be positive to trade",
            },
        )

    return {
        "ready": True,
        "reason": "ok",
        "message": "Execution path wired; kill switch disarmed; circuit closed; equity positive",
    }


@router.put("/trading/exposure_multiplier")
async def set_exposure_multiplier(request: Request, body: dict = None):
    """
    Set risk exposure multiplier from LLM advisory (0.5--1.5).
    Applied to effective equity for position sizing; caps at [0.5, 1.5].
    Body: {"multiplier": 0.5 | 1.0 | 1.5}.
    """
    body = body or {}
    mult = body.get("multiplier", 1.0)
    try:
        mult = float(mult)
    except (TypeError, ValueError):
        return JSONResponse(status_code=400, content={"error": "multiplier must be a number"})
    mult = max(0.5, min(1.5, mult))
    order_entry = getattr(request.app.state, "order_entry_service", None)
    if order_entry is None:
        return JSONResponse(status_code=503, content={"error": "order_entry_not_configured"})
    risk_manager = getattr(order_entry, "risk_manager", None)
    if risk_manager is None or not hasattr(risk_manager, "set_exposure_multiplier"):
        return JSONResponse(status_code=503, content={"error": "risk_manager_not_ready"})
    risk_manager.set_exposure_multiplier(mult)
    return {"status": "ok", "exposure_multiplier": mult}


@router.get("/trading/mode")
async def trading_mode(request: Request):
    """
    Returns current system trading mode and status:
    - mode: paper | live
    - autonomous: whether autonomous loop is running
    - safe_mode: whether safe mode is active
    - circuit_open: whether circuit breaker is tripped
    - kill_switch_armed: whether kill switch is armed
    """
    gateway = getattr(request.app.state, "gateway", None)
    is_paper = True
    if gateway is not None:
        is_paper = getattr(gateway, "paper", True)

    autonomous_loop = getattr(request.app.state, "autonomous_loop", None)
    autonomous_running = False
    if autonomous_loop is not None:
        autonomous_running = getattr(autonomous_loop, "_running", False)

    safe_mode = getattr(request.app.state, "safe_mode", False)

    circuit_open = False
    order_entry = getattr(request.app.state, "order_entry_service", None)
    if order_entry is not None:
        risk_manager = getattr(order_entry, "risk_manager", None)
        if risk_manager is not None:
            circuit_open = risk_manager.is_circuit_open()

    kill_switch = getattr(request.app.state, "kill_switch", None)
    kill_switch_armed = False
    if kill_switch is not None:
        try:
            import asyncio
            state = await kill_switch.get_state()
            kill_switch_armed = getattr(state, "armed", False)
        except Exception:
            pass

    return {
        "mode": "paper" if is_paper else "live",
        "autonomous": autonomous_running,
        "safe_mode": safe_mode,
        "circuit_open": circuit_open,
        "kill_switch_armed": kill_switch_armed,
    }


@router.put("/trading/autonomous")
async def set_autonomous_mode(request: Request, body: dict = None):
    """
    Toggle the autonomous loop on/off.
    Body: {"enabled": true/false}
    """
    body = body or {}
    enabled = body.get("enabled", False)

    autonomous_loop = getattr(request.app.state, "autonomous_loop", None)
    if autonomous_loop is None:
        return JSONResponse(
            status_code=503,
            content={"error": "autonomous_loop_not_configured", "message": "Autonomous loop not started at boot"},
        )

    if getattr(request.app.state, "safe_mode", False):
        return JSONResponse(
            status_code=403,
            content={"error": "safe_mode_active", "message": "Cannot enable autonomous trading in safe mode"},
        )

    if enabled:
        autonomous_loop._running = True
        logger.info("Autonomous loop ENABLED via API")
    else:
        autonomous_loop._running = False
        logger.info("Autonomous loop DISABLED via API")

    return {"status": "ok", "autonomous": enabled}
