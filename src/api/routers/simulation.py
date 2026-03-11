"""
Simulation API: nightly simulation control and results.
"""
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Query

from src.api.auth import get_current_user as require_auth

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/simulation", tags=["simulation"])

_orchestrator = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from src.simulation.orchestrator import SimulationOrchestrator
        _orchestrator = SimulationOrchestrator()
    return _orchestrator


@router.post("/run")
async def trigger_simulation(
    background_tasks: BackgroundTasks,
    intervals: str = Query("15m", description="Comma-separated intervals"),
    _user=Depends(require_auth),
):
    """Trigger nightly simulation (runs in background)."""
    try:
        orch = get_orchestrator()
        if orch.is_running:
            return {"status": "already_running", "message": "Simulation is already in progress"}

        interval_list = [i.strip() for i in intervals.split(",")]

        async def _run():
            await orch.run_nightly_pipeline(intervals=interval_list)

        background_tasks.add_task(_run)
        return {"status": "started", "intervals": interval_list}
    except Exception as e:
        logger.exception("Simulation run error")
        return {"status": "error", "message": "Simulation failed to start"}


@router.get("/status")
async def simulation_status(_user=Depends(require_auth)):
    """Get current simulation status."""
    try:
        orch = get_orchestrator()
        return {
            "running": orch.is_running,
            "last_run": orch.last_results[0].strategy_id if orch.last_results else None,
            "total_permutations": len(orch.last_results) if orch.last_results else 0,
        }
    except Exception as e:
        return {"running": False, "last_run": None, "total_permutations": 0}


@router.get("/results")
async def get_results(
    limit: int = Query(20, ge=1, le=100),
    selected_only: bool = False,
    _user=Depends(require_auth),
):
    """Get simulation results."""
    try:
        orch = get_orchestrator()
        if selected_only:
            results = orch.simulator.get_selected_strategies()
        else:
            results = orch.simulator.get_all_results()[:limit]

        selected = [r for r in results if r.selected]
        top_sharpe = max((r.sharpe_ratio for r in results), default=0)
        top_wr = max((r.win_rate for r in results), default=0)

        return {
            "results": [
                {
                    "rank": r.rank,
                    "strategy_id": r.strategy_id,
                    "params": r.params,
                    "interval": r.interval,
                    "sharpe": r.sharpe_ratio,
                    "sortino": r.sortino_ratio,
                    "max_dd": r.max_drawdown_pct,
                    "win_rate": r.win_rate,
                    "profit_factor": r.profit_factor,
                    "trades": r.total_trades,
                    "selected": r.selected,
                }
                for r in results
            ],
            "total_tested": len(orch.simulator.get_all_results()) if hasattr(orch, 'simulator') else 0,
            "qualified": len(selected),
            "top_sharpe": top_sharpe,
            "top_win_rate": top_wr,
        }
    except Exception as e:
        logger.warning("Simulation results error: %s", e)
        return {"results": [], "total_tested": 0, "qualified": 0, "top_sharpe": 0, "top_win_rate": 0}


@router.get("/permutation-count")
async def get_permutation_count(_user=Depends(require_auth)):
    """Preview how many permutations would be generated."""
    try:
        orch = get_orchestrator()
        # Use first symbol from dynamic universe for preview
        from src.scanner.dynamic_universe import get_dynamic_universe
        preview_symbols = get_dynamic_universe().get_tradeable_stocks(count=1) or ["SAMPLE"]
        perms = orch.simulator.generate_permutations(
            symbols=preview_symbols,
            intervals=["15m"],
        )
        return {"permutation_count": len(perms)}
    except Exception as e:
        return {"permutation_count": 0}
