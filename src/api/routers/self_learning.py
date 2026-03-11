"""Self-learning pipeline API: status, history, manual trigger, drift info."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.auth import get_current_user, require_roles

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/self-learning", tags=["self-learning"])


@router.get("/status")
async def get_self_learning_status(request: Request, current_user: dict = Depends(get_current_user)):
    """Get self-learning scheduler status: running, last run, next run, drift state."""
    sls = getattr(request.app.state, "self_learning_scheduler", None)
    if sls is None:
        return {
            "enabled": False,
            "reason": "SelfLearningScheduler not initialized",
        }

    history = sls.get_history()
    last_run = history[-1] if history else None

    return {
        "enabled": True,
        "running": sls._running,
        "last_run_date": sls._last_run_date,
        "schedule": {
            "post_market_time_ist": f"{sls.post_market_hour:02d}:{sls.post_market_minute:02d}",
            "weekly_revalidation_day": sls.weekly_revalidation_day,
            "min_drift_layers_for_retrain": sls.min_drift_layers,
        },
        "last_run": {
            "date": last_run.get("date") if last_run else None,
            "action": last_run.get("action") if last_run else None,
            "drift_layers_fired": last_run.get("drift_layers_fired", 0) if last_run else 0,
            "drift_reasons": last_run.get("drift_reasons", []) if last_run else [],
            "elapsed_seconds": last_run.get("elapsed_seconds", 0) if last_run else 0,
            "retrain_results": last_run.get("retrain_results") if last_run else None,
        },
        "total_runs": len(history),
    }


@router.get("/history")
async def get_self_learning_history(
    request: Request,
    limit: int = 30,
    current_user: dict = Depends(get_current_user),
):
    """Get self-learning execution history (last N runs)."""
    sls = getattr(request.app.state, "self_learning_scheduler", None)
    if sls is None:
        return {"history": [], "enabled": False}

    history = sls.get_history()
    return {
        "history": history[-limit:],
        "total": len(history),
        "enabled": True,
    }


@router.get("/drift")
async def get_drift_status(request: Request, current_user: dict = Depends(get_current_user)):
    """Get current drift detector and distribution monitor state."""
    result = {"drift_detector": None, "distribution_monitor": None}

    dd = getattr(request.app.state, "sl_drift_detector", None)
    if dd is not None:
        result["drift_detector"] = {
            "threshold": dd.threshold,
            "reference_features": len(dd.reference_stats),
            "reference_feature_names": list(dd.reference_stats.keys())[:20],
        }

    dm = getattr(request.app.state, "sl_distribution_monitor", None)
    if dm is not None:
        psi_values = {}
        for feat_name in list(dm._samples.keys())[:20]:
            try:
                psi_values[feat_name] = round(dm.psi(feat_name), 4)
            except Exception:
                pass
        high_psi = {k: v for k, v in psi_values.items() if v > 0.2}
        result["distribution_monitor"] = {
            "window": dm.window,
            "tracked_features": len(dm._samples),
            "psi_samples": psi_values,
            "high_psi_features": high_psi,
            "drift_alert": len(high_psi) > 0,
        }

    # Multi-layer drift detector (from setup_drift_detector)
    mld = getattr(request.app.state, "drift_detector", None)
    if mld is not None:
        try:
            signals = mld.check_all()
            result["multi_layer_drift"] = {
                s.drift_type.value: {
                    "drifted": s.drifted,
                    "value": round(s.value, 4) if isinstance(s.value, float) else s.value,
                    "threshold": s.threshold,
                }
                for s in signals
            }
        except Exception:
            logger.exception("Multi-layer drift check failed")
            result["multi_layer_drift"] = {"error": "Drift check failed"}

    return result


@router.post("/trigger")
async def trigger_self_learning_cycle(
    request: Request,
    current_user: dict = Depends(require_roles(["admin"])),
):
    """Manually trigger a self-learning cycle (admin only). Returns cycle result."""
    sls = getattr(request.app.state, "self_learning_scheduler", None)
    if sls is None:
        raise HTTPException(status_code=503, detail="SelfLearningScheduler not initialized")

    try:
        result = await sls.run_now()
        logger.info(
            "Manual self-learning cycle triggered by %s: %s",
            current_user.get("sub", "unknown"),
            result.get("action", "unknown"),
        )
        return {
            "status": "completed",
            "result": result,
        }
    except Exception:
        logger.exception("Manual self-learning cycle failed")
        raise HTTPException(status_code=500, detail="Self-learning cycle failed") from None
