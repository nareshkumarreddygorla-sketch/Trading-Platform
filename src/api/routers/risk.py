import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from src.api.auth import get_current_user, require_roles

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class RiskSnapshotResponse(BaseModel):
    equity: float
    daily_pnl: float
    positions: list[dict]
    circuit_open: bool = False
    kill_switch_armed: bool = False


class RiskPositionsResponse(BaseModel):
    positions: list[dict]


class RiskStateResponse(BaseModel):
    circuit_open: bool
    daily_pnl: float
    open_positions: int
    var_95: float = 0.0
    max_drawdown_pct: float = 0.0


class RiskLimitsResponse(BaseModel):
    max_position_pct: float
    max_daily_loss_pct: float
    max_open_positions: int


class UpdateLimitsResponse(BaseModel):
    status: str
    limits: dict[str, Any]


class VarResponse(BaseModel):
    var_95: float = 0.0
    var_99: float = 0.0
    portfolio_vol: float = 0.0
    horizon_days: int = 1
    per_position_var: dict[str, Any] = {}


class SectorResponse(BaseModel):
    sectors: dict[str, Any]
    max_sector_pct: float


class TailRiskResponse(BaseModel):
    vix_level: str
    exposure_scale: float
    recovery_phase: str
    last_vix_value: float


class CorrelationResponse(BaseModel):
    status: str
    max_pairwise_corr: float
    max_portfolio_vol_pct: float


class ModelWeightsResponse(BaseModel):
    weights: dict[str, Any]
    ic_scores: dict[str, Any]


class ModelDriftResponse(BaseModel):
    drifted: bool
    layers: dict[str, Any]
    error: str | None = None


class AlertHistoryResponse(BaseModel):
    alerts: list[Any]


class DatabaseHealthResponse(BaseModel):
    healthy: bool = True
    error: str | None = None


router = APIRouter()


def _get_risk_manager(request: Request):
    return getattr(request.app.state, "risk_manager", None)


def _get_bar_cache(request: Request):
    return getattr(request.app.state, "bar_cache", None)


def _latest_price_from_cache(bar_cache, symbol: str, exchange: str = "NSE") -> float | None:
    """Get the latest close price from bar cache for mark-to-market."""
    if bar_cache is None:
        return None
    try:
        bars = bar_cache.get_bars(symbol, exchange, "1d", 1)
        if bars and len(bars) > 0:
            bar = bars[-1]
            return getattr(bar, "close", None) or (bar.get("close") if isinstance(bar, dict) else None)
    except Exception as e:
        logger.debug("Failed to compute mark-to-market for %s: %s", symbol, e)
    return None


def _position_to_dict(p, bar_cache=None) -> dict:
    symbol = getattr(p, "symbol", "")
    exchange_val = getattr(getattr(p, "exchange", None), "value", "NSE")
    avg_price = getattr(p, "avg_price", 0.0)
    quantity = getattr(p, "quantity", 0)
    side = getattr(getattr(p, "side", None), "value", "BUY")
    strategy_id = getattr(p, "strategy_id", None)

    # Mark-to-market: get current price from bar cache
    current_price = _latest_price_from_cache(bar_cache, symbol, exchange_val)
    if current_price is None:
        current_price = avg_price  # fallback to entry price

    # Compute unrealized P&L
    if avg_price > 0 and current_price > 0 and quantity > 0:
        if side == "BUY":
            unrealized_pnl = (current_price - avg_price) * quantity
        else:
            unrealized_pnl = (avg_price - current_price) * quantity
        pct_change = (
            ((current_price - avg_price) / avg_price) * 100
            if side == "BUY"
            else ((avg_price - current_price) / avg_price) * 100
        )
    else:
        unrealized_pnl = getattr(p, "unrealized_pnl", 0.0)
        pct_change = 0.0

    return {
        "symbol": symbol,
        "exchange": exchange_val,
        "side": side,
        "quantity": quantity,
        "avg_price": avg_price,
        "entry_price": avg_price,
        "current_price": round(current_price, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "pct_change": round(pct_change, 2),
        "strategy_id": strategy_id,
    }


@router.get("/snapshot", response_model=RiskSnapshotResponse)
async def risk_snapshot(request: Request, current_user: dict = Depends(get_current_user)):
    """Dashboard snapshot: equity, daily_pnl, positions (same shape as frontend expects)."""
    risk_manager = _get_risk_manager(request)
    if risk_manager is None:
        raise HTTPException(status_code=503, detail="Risk manager not initialized")
    bar_cache = _get_bar_cache(request)
    with risk_manager._positions_lock:
        positions_snapshot = list(risk_manager.positions)
    positions = [_position_to_dict(p, bar_cache) for p in positions_snapshot]
    total_unrealized = sum(pos["unrealized_pnl"] for pos in positions)
    # Circuit breaker and kill switch state
    circuit_open = risk_manager.is_circuit_open()
    kill_switch = getattr(request.app.state, "kill_switch", None)
    kill_switch_armed = kill_switch.armed if kill_switch and hasattr(kill_switch, "armed") else False

    return {
        "equity": risk_manager.equity,
        "daily_pnl": risk_manager.daily_pnl if risk_manager.daily_pnl != 0 else total_unrealized,
        "positions": positions,
        "circuit_open": circuit_open,
        "kill_switch_armed": kill_switch_armed,
    }


@router.get("/positions", response_model=RiskPositionsResponse)
async def risk_positions(request: Request, current_user: dict = Depends(get_current_user)):
    """Open positions from risk manager (for dashboard/positions page)."""
    risk_manager = _get_risk_manager(request)
    if risk_manager is None:
        raise HTTPException(status_code=503, detail="Risk manager not initialized")
    bar_cache = _get_bar_cache(request)
    with risk_manager._positions_lock:
        positions_snapshot = list(risk_manager.positions)
    positions = [_position_to_dict(p, bar_cache) for p in positions_snapshot]
    return {"positions": positions}


@router.get("/state", response_model=RiskStateResponse)
async def risk_state(request: Request, current_user: dict = Depends(get_current_user)):
    """Return current risk state from RiskManager (circuit, PnL, positions)."""
    risk_manager = _get_risk_manager(request)
    if risk_manager is None:
        raise HTTPException(status_code=503, detail="Risk manager not initialized")
    with risk_manager._positions_lock:
        positions_snapshot = list(risk_manager.positions)
    return {
        "circuit_open": risk_manager.is_circuit_open(),
        "daily_pnl": risk_manager.daily_pnl,
        "open_positions": len(positions_snapshot),
        "var_95": getattr(risk_manager, "var_95", 0.0),
        "max_drawdown_pct": getattr(risk_manager, "max_drawdown_pct", 0.0),
    }


@router.get("/limits", response_model=RiskLimitsResponse)
async def get_limits(request: Request, current_user: dict = Depends(get_current_user)):
    """Return current risk limits from RiskManager."""
    risk_manager = _get_risk_manager(request)
    if risk_manager is None:
        raise HTTPException(status_code=503, detail="Risk manager not initialized")
    limits = risk_manager.limits
    return {
        "max_position_pct": limits.max_position_pct,
        "max_daily_loss_pct": limits.max_daily_loss_pct,
        "max_open_positions": limits.max_open_positions,
    }


class UpdateLimitsBody(BaseModel):
    max_position_pct: float | None = None
    max_daily_loss_pct: float | None = None
    max_open_positions: int | None = None


@router.put("/limits")
async def update_limits(
    request: Request,
    body: UpdateLimitsBody,
    current_user: dict = Depends(require_roles(["admin"])),
):
    """Update risk limits. Validates ranges; applies to RiskManager if configured. Admin only."""
    limits_dict: dict[str, Any] = body.model_dump(exclude_none=True)
    if not limits_dict:
        raise HTTPException(400, "At least one limit field required")
    # Validate
    if "max_position_pct" in limits_dict:
        v = limits_dict["max_position_pct"]
        if not isinstance(v, (int, float)) or v <= 0 or v > 100:
            raise HTTPException(400, "max_position_pct must be in (0, 100]")
        limits_dict["max_position_pct"] = float(v)
    if "max_daily_loss_pct" in limits_dict:
        v = limits_dict["max_daily_loss_pct"]
        if not isinstance(v, (int, float)) or v < 0 or v > 100:
            raise HTTPException(400, "max_daily_loss_pct must be in [0, 100]")
        limits_dict["max_daily_loss_pct"] = float(v)
    if "max_open_positions" in limits_dict:
        v = limits_dict["max_open_positions"]
        if not isinstance(v, int) or v < 1 or v > 1000:
            raise HTTPException(400, "max_open_positions must be integer in [1, 1000]")

    # Capture old values for audit
    risk_manager = _get_risk_manager(request)
    old_limits = {}
    if risk_manager is not None:
        for k in limits_dict:
            old_limits[k] = getattr(risk_manager.limits, k, None)
        for k, v in limits_dict.items():
            setattr(risk_manager.limits, k, v)

    # Audit log the limits change
    audit_repo = getattr(request.app.state, "audit_repo", None)
    if audit_repo:
        try:
            actor = current_user.get("user_id", "admin") if current_user else "admin"
            audit_repo.append_sync("limits_updated", actor, {"old": old_limits, "new": limits_dict})
        except Exception as e:
            logger.error("CRITICAL: Audit log append failed for limits_updated: %s", e)

    return UpdateLimitsResponse(status="ok", limits=limits_dict)


@router.get("/var", response_model=VarResponse)
async def risk_var(request: Request, current_user: dict = Depends(get_current_user)):
    """Real portfolio VaR (95% and 99%) with position breakdown."""
    var_result = getattr(request.app.state, "_last_var", None)

    # If no cached VaR yet, compute on-demand from PortfolioVaR + RiskManager
    if var_result is None:
        portfolio_var = getattr(request.app.state, "portfolio_var", None)
        risk_manager = _get_risk_manager(request)
        if portfolio_var is not None and risk_manager is not None:
            try:
                with risk_manager._positions_lock:
                    positions = [
                        {
                            "symbol": p.symbol,
                            "notional": abs(getattr(p, "avg_price", 0) * p.quantity),
                        }
                        for p in risk_manager.positions
                    ]
                var_result = portfolio_var.compute(positions, risk_manager.equity)
                # Cache the result for future requests
                request.app.state._last_var = var_result
            except Exception as e:
                logger.warning("On-demand VaR computation failed: %s", e)

    if var_result is None:
        raise HTTPException(
            status_code=503, detail="VaR data not available — no positions or PortfolioVaR not initialized"
        )
    return {
        "var_95": getattr(var_result, "var_95", 0),
        "var_99": getattr(var_result, "var_99", 0),
        "portfolio_vol": getattr(var_result, "portfolio_vol", 0),
        "horizon_days": getattr(var_result, "horizon_days", 1),
        "per_position_var": getattr(var_result, "per_position_var", {}),
    }


@router.get("/sectors", response_model=SectorResponse)
async def risk_sectors(request: Request, current_user: dict = Depends(get_current_user)):
    """Sector concentration breakdown for current portfolio."""
    sc = getattr(request.app.state, "sector_classifier", None)
    rm = _get_risk_manager(request)
    if sc is None or rm is None:
        raise HTTPException(status_code=503, detail="Sector classifier or risk manager not initialized")
    positions = [{"symbol": p.symbol, "notional": abs(getattr(p, "avg_price", 0) * p.quantity)} for p in rm.positions]
    breakdown = sc.get_sector_breakdown(positions)
    return {"sectors": breakdown, "max_sector_pct": 30.0}


@router.get("/tail", response_model=TailRiskResponse)
async def risk_tail(request: Request, current_user: dict = Depends(get_current_user)):
    """Tail risk status: VIX level, rapid drawdown, recovery phase."""
    trp = getattr(request.app.state, "tail_risk_protector", None)
    if trp is None:
        raise HTTPException(status_code=503, detail="Tail risk protector not initialized")
    state = getattr(trp, "state", None)
    if state is None:
        raise HTTPException(status_code=503, detail="Tail risk state not available")
    return {
        "vix_level": getattr(getattr(state, "vix_level", None), "value", "UNKNOWN"),
        "exposure_scale": getattr(state, "exposure_scale", 1.0),
        "recovery_phase": getattr(getattr(state, "recovery_phase", None), "value", "NORMAL"),
        "last_vix_value": getattr(state, "vix_value", 0),
    }


@router.get("/vol-targeting")
async def risk_vol_targeting(request: Request, current_user: dict = Depends(get_current_user)):
    """Volatility targeting state: target vs realized vol, scale factor."""
    vt = getattr(request.app.state, "vol_targeter", None)
    if vt is None:
        raise HTTPException(status_code=503, detail="Volatility targeter not initialized")
    return vt.state.as_dict()


@router.get("/correlation", response_model=CorrelationResponse)
async def risk_correlation(request: Request, current_user: dict = Depends(get_current_user)):
    """Current correlation matrix status."""
    cg = getattr(request.app.state, "correlation_guard", None)
    if cg is None:
        raise HTTPException(status_code=503, detail="Correlation guard not initialized")
    return {
        "status": "active",
        "max_pairwise_corr": cg.max_pairwise_correlation,
        "max_portfolio_vol_pct": cg.max_portfolio_vol_pct,
    }


@router.get("/models/weights", response_model=ModelWeightsResponse)
async def model_weights(request: Request, current_user: dict = Depends(get_current_user)):
    """Current ensemble model weights (IC-weighted)."""
    ee = getattr(request.app.state, "ensemble_engine", None)
    if ee is None:
        raise HTTPException(status_code=503, detail="Ensemble engine not initialized")
    weights = getattr(ee, "weights", {})
    ic_scores = {}
    if hasattr(ee, "get_model_ic_scores"):
        ic_scores = ee.get_model_ic_scores()
    return {"weights": weights, "ic_scores": ic_scores}


@router.get("/models/drift", response_model=ModelDriftResponse)
async def model_drift(request: Request, current_user: dict = Depends(get_current_user)):
    """Current drift detection status per layer."""
    dd = getattr(request.app.state, "drift_detector", None)
    if dd is None:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")
    try:
        signals = dd.check_all()
        drifted = sum(1 for s in signals if s.drifted) >= 2
        layers = {
            s.drift_type.value: {"drifted": s.drifted, "value": round(s.value, 4), "threshold": s.threshold}
            for s in signals
        }
        return {"drifted": drifted, "layers": layers}
    except Exception:
        logger.exception("Drift detection check failed")
        return {"drifted": False, "layers": {}, "error": "drift_check_failed"}


@router.get("/alerts/history", response_model=AlertHistoryResponse)
async def alert_history(request: Request, current_user: dict = Depends(get_current_user)):
    """Recent alert history."""
    an = getattr(request.app.state, "alert_notifier", None)
    if an is None:
        raise HTTPException(status_code=503, detail="Alert notifier not initialized")
    return {"alerts": an.get_history(limit=50)}


@router.post("/stress-test/run")
async def run_stress_test(request: Request, current_user: dict = Depends(require_roles(["admin", "user"]))):
    """Run full stress test suite against current portfolio."""
    try:
        from src.risk_engine.stress_testing import StressTestEngine
    except ImportError:
        raise HTTPException(status_code=501, detail="Stress testing module not available") from None

    rm = _get_risk_manager(request)
    if rm is None:
        raise HTTPException(status_code=503, detail="Risk manager not available")

    sector_clf = getattr(request.app.state, "sector_classifier", None)
    engine = StressTestEngine(sector_classifier=sector_clf)
    _results = engine.run_full_suite(list(rm.positions), rm.equity)

    # Store on app state for dashboard
    request.app.state.stress_test_engine = engine

    return engine.get_summary()


@router.get("/stress-test/results")
async def stress_test_results(request: Request, current_user: dict = Depends(get_current_user)):
    """Get results from last stress test run."""
    engine = getattr(request.app.state, "stress_test_engine", None)
    if engine is None:
        return {"status": "no_results", "scenarios_run": 0}
    return engine.get_summary()


@router.get("/archival/policy")
async def archival_policy(request: Request, current_user: dict = Depends(get_current_user)):
    """Get current data retention/archival policy."""
    try:
        from src.persistence.archival import DataArchivalManager

        mgr = DataArchivalManager()
        return mgr.get_retention_summary()
    except ImportError:
        raise HTTPException(status_code=503, detail="Archival module not available") from None


@router.post("/reconciliation/run")
async def run_reconciliation(request: Request, current_user: dict = Depends(require_roles(["admin", "trader"]))):
    """Run broker position reconciliation on demand."""
    reconciliator = getattr(request.app.state, "broker_reconciliator", None)
    if reconciliator is None:
        raise HTTPException(status_code=503, detail="Broker reconciliator not configured")
    report = await reconciliator.reconcile()
    return report.as_dict()


@router.get("/reconciliation/history")
async def reconciliation_history(request: Request, current_user: dict = Depends(get_current_user)):
    """Get reconciliation history."""
    reconciliator = getattr(request.app.state, "broker_reconciliator", None)
    if reconciliator is None:
        raise HTTPException(status_code=503, detail="Broker reconciliator not initialized")
    return {"history": await reconciliator.get_history()}


@router.get("/feature-shift")
async def feature_shift_status(request: Request, current_user: dict = Depends(get_current_user)):
    """Check feature distribution shift status."""
    detector = getattr(request.app.state, "feature_shift_detector", None)
    if detector is None:
        raise HTTPException(status_code=503, detail="Feature shift detector not initialized")
    report = detector.check_shift()
    return report.as_dict()


@router.get("/feature-shift/history")
async def feature_shift_history(request: Request, current_user: dict = Depends(get_current_user)):
    """Get feature shift detection history."""
    detector = getattr(request.app.state, "feature_shift_detector", None)
    if detector is None:
        raise HTTPException(status_code=503, detail="Feature shift detector not initialized")
    return {"history": detector.get_shift_history()}


@router.get("/database/health", response_model=DatabaseHealthResponse)
async def database_health(request: Request, current_user: dict = Depends(get_current_user)):
    """Check database health: disk usage, connection pool, connectivity."""
    try:
        from src.persistence.database import check_database_health

        return check_database_health()
    except Exception as e:
        return {"healthy": False, "error": str(e)}
