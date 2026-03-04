"""
Trading readiness: only 200 when execution path is wired and system is allowed to trade.
Used by K8s readiness probe and dashboards; 503 when kill switch armed or circuit open.

Includes explicit /trading/stop and /trading/start endpoints for admin control
of the autonomous trading loop with optional position close-out.
"""
import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from src.api.auth import get_current_user, require_roles

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


# ---------------------------------------------------------------------------
# POST /trading/stop  —  Admin-only: stop autonomous loop, optionally close positions
# ---------------------------------------------------------------------------

@router.post("/trading/stop")
async def trading_stop(
    request: Request,
    close_positions: bool = Query(False, description="If true, close all open positions with market orders"),
    current_user: dict = Depends(require_roles(["admin"])),
):
    """
    Immediately stop the autonomous trading loop.

    - Requires admin role.
    - If ``close_positions=true``, iterates all tracked open trades and submits
      market close orders (SELL for long, BUY for short).
    - All actions are logged to the audit trail.

    Returns:
        stopped (bool): Always True on success.
        positions_closed (int): Number of positions for which close orders were submitted.
        mode (str): ``"paper"`` or ``"live"``.
    """
    actor = current_user.get("user_id", "admin")

    # Determine trading mode
    gateway = getattr(request.app.state, "gateway", None)
    is_paper = True
    if gateway is not None:
        is_paper = getattr(gateway, "paper", True)
    mode = "paper" if is_paper else "live"

    # ── Stop the autonomous loop ──
    autonomous_loop = getattr(request.app.state, "autonomous_loop", None)
    if autonomous_loop is None:
        logger.warning("[trading/stop] No autonomous loop configured (actor=%s)", actor)
        return JSONResponse(
            status_code=503,
            content={"error": "autonomous_loop_not_configured", "message": "Autonomous loop not available"},
        )

    was_running = getattr(autonomous_loop, "_running", False)
    try:
        await autonomous_loop.stop()
    except Exception as exc:
        logger.exception("[trading/stop] Error stopping autonomous loop: %s", exc)

    logger.info(
        "AUDIT | trading/stop | actor=%s | was_running=%s | close_positions=%s | mode=%s",
        actor, was_running, close_positions, mode,
    )

    # ── Optionally close all open positions ──
    positions_closed = 0
    if close_positions:
        open_trades = dict(getattr(autonomous_loop, "_open_trades", {}))
        if open_trades:
            logger.info("[trading/stop] Closing %d open positions (actor=%s)", len(open_trades), actor)

            for trade_key, trade in list(open_trades.items()):
                symbol = trade.get("symbol") or trade_key.split(":")[0]
                try:
                    # Determine closing side
                    trade_side = trade.get("side", "BUY")
                    close_side_str = "SELL" if trade_side == "BUY" else "BUY"

                    # Get current market price for the close order
                    exit_price = trade.get("entry_price", 0)
                    get_bars_fn = getattr(autonomous_loop, "get_bars", None)
                    if get_bars_fn:
                        try:
                            from src.models.common import Exchange
                            bars = get_bars_fn(symbol, Exchange.NSE, "1m", 2)
                            if bars:
                                exit_price = bars[-1].close
                        except Exception:
                            pass

                    # Build and submit close order via order entry service
                    order_entry = getattr(request.app.state, "order_entry_service", None)
                    if order_entry is not None and hasattr(order_entry, "submit_order"):
                        from src.models.common import (
                            Exchange,
                            OrderType,
                            Signal,
                            SignalSide,
                        )
                        from src.execution.order_entry import OrderEntryRequest
                        from src.execution.idempotency import stable_idempotency_key

                        close_side = SignalSide.SELL if trade_side == "BUY" else SignalSide.BUY
                        close_signal = Signal(
                            strategy_id=trade.get("strategy_id", "manual_close"),
                            symbol=symbol,
                            exchange=Exchange.NSE,
                            side=close_side,
                            score=1.0,
                            portfolio_weight=0.0,
                            risk_level="EMERGENCY",
                            reason=f"ADMIN_STOP_CLOSE: actor={actor} {trade_side} {symbol}",
                            price=exit_price,
                            ts=datetime.now(timezone.utc),
                        )
                        bar_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                        idem_key = stable_idempotency_key(
                            bar_ts, "admin_stop_close", symbol, close_side.value,
                        )
                        req = OrderEntryRequest(
                            signal=close_signal,
                            quantity=trade.get("qty", trade.get("quantity", 0)),
                            order_type=OrderType.MARKET,
                            idempotency_key=idem_key,
                            source="admin_trading_stop",
                            force_reduce=True,
                        )
                        result = await order_entry.submit_order(req)
                        if result.success:
                            positions_closed += 1
                            # Remove from tracked open trades
                            autonomous_loop._open_trades.pop(trade_key, None)
                            # Remove from persistence
                            repo = getattr(autonomous_loop, "_open_trade_repo", None)
                            if repo and hasattr(repo, "delete_trade"):
                                try:
                                    repo.delete_trade(trade_key)
                                except Exception:
                                    pass
                            logger.info(
                                "AUDIT | position_closed | actor=%s | symbol=%s | side=%s | qty=%s",
                                actor, symbol, close_side_str, trade.get("qty", trade.get("quantity", 0)),
                            )
                        else:
                            logger.warning(
                                "[trading/stop] Close order rejected for %s: %s",
                                symbol, getattr(result, "reject_reason", "unknown"),
                            )
                    else:
                        logger.warning("[trading/stop] OrderEntryService unavailable; cannot close %s", symbol)

                except Exception as exc:
                    logger.exception("[trading/stop] Failed to close position %s: %s", trade_key, exc)

    # Audit trail via SEBI audit if available
    sebi_audit = getattr(request.app.state, "sebi_audit", None)
    if sebi_audit and hasattr(sebi_audit, "log"):
        try:
            sebi_audit.log(
                event="TRADING_STOP",
                actor=actor,
                details={
                    "was_running": was_running,
                    "close_positions": close_positions,
                    "positions_closed": positions_closed,
                    "mode": mode,
                },
            )
        except Exception:
            pass

    return {
        "stopped": True,
        "positions_closed": positions_closed,
        "mode": mode,
    }


# ---------------------------------------------------------------------------
# POST /trading/start  —  Admin-only: start autonomous loop
# ---------------------------------------------------------------------------

@router.post("/trading/start")
async def trading_start(
    request: Request,
    current_user: dict = Depends(require_roles(["admin"])),
):
    """
    Start the autonomous trading loop if not already running.

    - Requires admin role.
    - Refuses to start if safe_mode is active (broker unreachable).
    - All actions are logged to the audit trail.

    Returns:
        started (bool): True if the loop was started.
        mode (str): ``"paper"`` or ``"live"``.
        strategies_active (int): Number of enabled strategies in the runner.
    """
    actor = current_user.get("user_id", "admin")

    # Determine trading mode
    gateway = getattr(request.app.state, "gateway", None)
    is_paper = True
    if gateway is not None:
        is_paper = getattr(gateway, "paper", True)
    mode = "paper" if is_paper else "live"

    # Block if safe mode
    if getattr(request.app.state, "safe_mode", False):
        logger.warning("[trading/start] Blocked: safe_mode active (actor=%s)", actor)
        return JSONResponse(
            status_code=403,
            content={
                "error": "safe_mode_active",
                "message": "Cannot start autonomous trading while broker is unreachable (safe mode)",
            },
        )

    # Get autonomous loop
    autonomous_loop = getattr(request.app.state, "autonomous_loop", None)
    if autonomous_loop is None:
        logger.warning("[trading/start] No autonomous loop configured (actor=%s)", actor)
        return JSONResponse(
            status_code=503,
            content={"error": "autonomous_loop_not_configured", "message": "Autonomous loop not available"},
        )

    already_running = getattr(autonomous_loop, "_running", False)
    if already_running:
        logger.info("[trading/start] Loop already running (actor=%s)", actor)
    else:
        autonomous_loop.start()
        logger.info(
            "AUDIT | trading/start | actor=%s | mode=%s",
            actor, mode,
        )

    # Count active strategies
    strategies_active = 0
    strategy_runner = getattr(autonomous_loop, "strategy_runner", None)
    if strategy_runner is not None:
        registry = getattr(strategy_runner, "registry", None)
        if registry is not None and hasattr(registry, "get_enabled_strategies"):
            try:
                strategies_active = len(registry.get_enabled_strategies())
            except Exception:
                pass

    # Audit trail
    sebi_audit = getattr(request.app.state, "sebi_audit", None)
    if sebi_audit and hasattr(sebi_audit, "log"):
        try:
            sebi_audit.log(
                event="TRADING_START",
                actor=actor,
                details={
                    "already_running": already_running,
                    "mode": mode,
                    "strategies_active": strategies_active,
                },
            )
        except Exception:
            pass

    return {
        "started": True,
        "mode": mode,
        "strategies_active": strategies_active,
    }
