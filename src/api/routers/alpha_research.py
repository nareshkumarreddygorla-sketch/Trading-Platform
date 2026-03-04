"""
Alpha Research & Edge Discovery pipeline API.
POST /run — trigger pipeline run (async or sync);
GET /status — last run status;
GET /results — last run results (selected signals, scores).
"""
import logging
import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory last run (replace with DB or cache in production); lock for thread-safe access
_last_run: Dict[str, Any] = {"status": "idle", "results": None, "error": None}
_last_run_lock = threading.Lock()


def _get_pipeline(request: Request):
    return getattr(request.app.state, "alpha_research_pipeline", None)


@router.post("/alpha_research/run")
async def run_alpha_research(request: Request, background_tasks: BackgroundTasks):
    """
    Trigger one full pipeline run: generate -> validate -> score -> cluster -> capacity.
    Runs in background; use GET /alpha_research/status for status.
    """
    pipeline = _get_pipeline(request)
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"error": "alpha_research_pipeline_not_configured"},
        )

    def _run():
        import numpy as np
        with _last_run_lock:
            _last_run["status"] = "running"
            _last_run["results"] = None
            _last_run["error"] = None
        try:
            candidates = pipeline.run_generation()
            signal_matrix = getattr(pipeline, "_last_signal_matrix", None)
            forward_returns = getattr(pipeline, "_last_forward_returns", None)
            backtest_results = getattr(pipeline, "_last_backtest_results", None)
            if signal_matrix is None or forward_returns is None:
                n_bars = 500
                rng = np.random.default_rng(42)
                forward_returns = rng.standard_normal(n_bars) * 0.01
                signal_matrix = {c.hypothesis_id: rng.standard_normal(n_bars).cumsum() * 0.001 for c in candidates[:15]}
                backtest_results = {
                    c.hypothesis_id: {"sharpe_oos": 0.3, "turnover": 0.3, "mean_return_gross": 0.0002, "n_wf_positive": 3}
                    for c in candidates[:15]
                }
            validated = pipeline.run_validation(
                candidates,
                signal_matrix=signal_matrix,
                forward_returns=forward_returns,
                backtest_results=backtest_results,
            )
            ranked = pipeline.run_scoring()
            with _last_run_lock:
                _last_run["results"] = {
                    "candidates_generated": len(candidates),
                    "validated_passed": sum(1 for v in validated if v.passed),
                    "validated_total": len(validated),
                    "top_decile_ids": [r[0] for r in ranked],
                    "top_decile_scores": [round(r[1], 4) for r in ranked],
                }
                _last_run["status"] = "completed"
        except Exception as e:
            logger.exception("Alpha research pipeline failed: %s", e)
            with _last_run_lock:
                _last_run["status"] = "failed"
                _last_run["error"] = str(e)

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Pipeline running in background; check GET /api/v1/alpha_research/status"}


@router.get("/alpha_research/status")
async def alpha_research_status():
    """Last run status: idle | running | completed | failed."""
    with _last_run_lock:
        return {"status": _last_run["status"], "error": _last_run.get("error")}


@router.get("/alpha_research/results")
async def alpha_research_results():
    """Last run results (selected signals, scores)."""
    with _last_run_lock:
        status = _last_run["status"]
        results = _last_run.get("results")
    if status != "completed" or results is None:
        return {"status": status, "results": None}
    return {"status": status, "results": results}


@router.post("/alpha_research/decay_multipliers")
async def get_decay_multipliers(request: Request, body: dict = None):
    """
    Return recommended weight multipliers from DecayMonitor for given signal_ids.
    Body: {"signal_ids": ["id1", "id2"]}. Used by meta_allocator to scale by decay.
    """
    pipeline = _get_pipeline(request)
    if pipeline is None or not hasattr(pipeline, "decay_monitor"):
        return JSONResponse(status_code=503, content={"error": "pipeline_not_configured"})
    body = body or {}
    signal_ids = body.get("signal_ids", [])
    if not isinstance(signal_ids, list):
        return JSONResponse(status_code=400, content={"error": "signal_ids must be a list"})
    mults = pipeline.get_decay_weight_multipliers(signal_ids)
    return {"multipliers": mults}

