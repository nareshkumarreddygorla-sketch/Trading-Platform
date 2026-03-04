from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from src.api.auth import get_current_user, require_roles

router = APIRouter()


def _get_risk_manager(request: Request):
    return getattr(request.app.state, "risk_manager", None)


def _get_bar_cache(request: Request):
    return getattr(request.app.state, "bar_cache", None)


def _latest_price_from_cache(bar_cache, symbol: str, exchange: str = "NSE") -> Optional[float]:
    """Get the latest close price from bar cache for mark-to-market."""
    if bar_cache is None:
        return None
    try:
        bars = bar_cache.get_bars(symbol, exchange, "1d", 1)
        if bars and len(bars) > 0:
            bar = bars[-1]
            return getattr(bar, "close", None) or (bar.get("close") if isinstance(bar, dict) else None)
    except Exception:
        pass
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
        pct_change = ((current_price - avg_price) / avg_price) * 100 if side == "BUY" else ((avg_price - current_price) / avg_price) * 100
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


@router.get("/snapshot")
async def risk_snapshot(request: Request, current_user: dict = Depends(get_current_user)):
    """Dashboard snapshot: equity, daily_pnl, positions (same shape as frontend expects)."""
    risk_manager = _get_risk_manager(request)
    if risk_manager is None:
        return {
            "equity": 0.0,
            "daily_pnl": 0.0,
            "positions": [],
        }
    bar_cache = _get_bar_cache(request)
    positions = [_position_to_dict(p, bar_cache) for p in risk_manager.positions]
    total_unrealized = sum(pos["unrealized_pnl"] for pos in positions)
    return {
        "equity": risk_manager.equity,
        "daily_pnl": risk_manager.daily_pnl if risk_manager.daily_pnl != 0 else total_unrealized,
        "positions": positions,
    }


@router.get("/positions")
async def risk_positions(request: Request, current_user: dict = Depends(get_current_user)):
    """Open positions from risk manager (for dashboard/positions page)."""
    risk_manager = _get_risk_manager(request)
    if risk_manager is None:
        return {"positions": []}
    bar_cache = _get_bar_cache(request)
    positions = [_position_to_dict(p, bar_cache) for p in risk_manager.positions]
    return {"positions": positions}


@router.get("/state")
async def risk_state(request: Request, current_user: dict = Depends(get_current_user)):
    """Return current risk state from RiskManager (circuit, PnL, positions)."""
    risk_manager = _get_risk_manager(request)
    if risk_manager is None:
        return {
            "circuit_open": False,
            "daily_pnl": 0.0,
            "open_positions": 0,
            "var_95": 0.0,
            "max_drawdown_pct": 0.0,
        }
    return {
        "circuit_open": risk_manager.is_circuit_open(),
        "daily_pnl": risk_manager.daily_pnl,
        "open_positions": len(risk_manager.positions),
        "var_95": getattr(risk_manager, "var_95", 0.0),
        "max_drawdown_pct": getattr(risk_manager, "max_drawdown_pct", 0.0),
    }


@router.get("/limits")
async def get_limits(request: Request, current_user: dict = Depends(get_current_user)):
    """Return current risk limits from RiskManager."""
    risk_manager = _get_risk_manager(request)
    if risk_manager is None:
        from src.risk_engine import RiskLimits
        limits = RiskLimits()
    else:
        limits = risk_manager.limits
    return {
        "max_position_pct": limits.max_position_pct,
        "max_daily_loss_pct": limits.max_daily_loss_pct,
        "max_open_positions": limits.max_open_positions,
    }


class UpdateLimitsBody(BaseModel):
    max_position_pct: Optional[float] = None
    max_daily_loss_pct: Optional[float] = None
    max_open_positions: Optional[int] = None


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
        from src.risk_engine.limits import RiskLimits
        for k, v in limits_dict.items():
            setattr(risk_manager.limits, k, v)

    # Audit log the limits change
    audit_repo = getattr(request.app.state, "audit_repo", None)
    if audit_repo:
        try:
            actor = current_user.get("user_id", "admin") if current_user else "admin"
            audit_repo.append_sync("limits_updated", actor, {"old": old_limits, "new": limits_dict})
        except Exception:
            pass

    return {"status": "ok", "limits": limits_dict}


@router.get("/var")
async def risk_var(request: Request, current_user: dict = Depends(get_current_user)):
    """Real portfolio VaR (95% and 99%) with position breakdown."""
    var_result = getattr(request.app.state, "_last_var", None)
    if var_result is None:
        return {"var_95": 0.0, "var_99": 0.0, "portfolio_vol": 0.0, "horizon_days": 1, "positions": []}
    return {
        "var_95": getattr(var_result, "var_95", 0),
        "var_99": getattr(var_result, "var_99", 0),
        "portfolio_vol": getattr(var_result, "portfolio_vol", 0),
        "horizon_days": getattr(var_result, "horizon_days", 1),
        "per_position_var": getattr(var_result, "per_position_var", {}),
    }


@router.get("/sectors")
async def risk_sectors(request: Request, current_user: dict = Depends(get_current_user)):
    """Sector concentration breakdown for current portfolio."""
    sc = getattr(request.app.state, "sector_classifier", None)
    rm = _get_risk_manager(request)
    if sc is None or rm is None:
        return {"sectors": {}, "max_sector_pct": 30.0}
    positions = [{"symbol": p.symbol, "notional": abs(getattr(p, "avg_price", 0) * p.quantity)} for p in rm.positions]
    breakdown = sc.get_sector_breakdown(positions)
    return {"sectors": breakdown, "max_sector_pct": 30.0}


@router.get("/tail")
async def risk_tail(request: Request, current_user: dict = Depends(get_current_user)):
    """Tail risk status: VIX level, rapid drawdown, recovery phase."""
    trp = getattr(request.app.state, "tail_risk_protector", None)
    if trp is None:
        return {"vix_level": "UNKNOWN", "exposure_scale": 1.0, "recovery_phase": "NORMAL"}
    state = getattr(trp, "state", None)
    if state is None:
        return {"vix_level": "UNKNOWN", "exposure_scale": 1.0, "recovery_phase": "NORMAL"}
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
        return {"target_vol": 12.0, "realized_vol": 0.0, "scale_factor": 1.0, "n_observations": 0}
    return vt.state.as_dict()


@router.get("/correlation")
async def risk_correlation(request: Request, current_user: dict = Depends(get_current_user)):
    """Current correlation matrix status."""
    cg = getattr(request.app.state, "correlation_guard", None)
    if cg is None:
        return {"status": "not_initialized", "max_pairwise_corr": 0.70}
    return {
        "status": "active",
        "max_pairwise_corr": cg.max_pairwise_correlation,
        "max_portfolio_vol_pct": cg.max_portfolio_vol_pct,
    }


@router.get("/models/weights")
async def model_weights(request: Request, current_user: dict = Depends(get_current_user)):
    """Current ensemble model weights (IC-weighted)."""
    ee = getattr(request.app.state, "ensemble_engine", None)
    if ee is None:
        return {"weights": {}, "ic_scores": {}}
    weights = getattr(ee, "weights", {})
    ic_scores = {}
    if hasattr(ee, "get_model_ic_scores"):
        ic_scores = ee.get_model_ic_scores()
    return {"weights": weights, "ic_scores": ic_scores}


@router.get("/models/drift")
async def model_drift(request: Request, current_user: dict = Depends(get_current_user)):
    """Current drift detection status per layer."""
    dd = getattr(request.app.state, "drift_detector", None)
    if dd is None:
        return {"drifted": False, "layers": {}}
    try:
        signals = dd.check_all()
        drifted = sum(1 for s in signals if s.drifted) >= 2
        layers = {s.drift_type.value: {"drifted": s.drifted, "value": round(s.value, 4), "threshold": s.threshold} for s in signals}
        return {"drifted": drifted, "layers": layers}
    except Exception:
        return {"drifted": False, "layers": {}}


@router.get("/alerts/history")
async def alert_history(request: Request, current_user: dict = Depends(get_current_user)):
    """Recent alert history."""
    an = getattr(request.app.state, "alert_notifier", None)
    if an is None:
        return {"alerts": []}
    return {"alerts": an.get_history(limit=50)}
