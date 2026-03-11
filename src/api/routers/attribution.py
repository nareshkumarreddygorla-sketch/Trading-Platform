"""
Performance Attribution API: break down P&L by model, strategy, symbol, sector, regime.
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query

from src.api.auth import get_current_user as require_auth

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/attribution", tags=["attribution"])


def _get_repo():
    from src.persistence.trade_outcome_repo import TradeOutcomeRepository
    return TradeOutcomeRepository()


def _get_engine(repo=None):
    from src.reporting.performance_attribution import PerformanceAttributionEngine
    return PerformanceAttributionEngine(trade_outcome_repo=repo or _get_repo())


@router.get("/by-dimension")
async def get_attribution(
    dimension: str = Query("model_id", pattern="^(model_id|strategy_id|symbol|sector|regime|time_bucket|exit_reason|side)$"),
    days: int = Query(30, ge=1, le=365),
    _user=Depends(require_auth),
):
    """Get performance attribution by a specific dimension."""
    try:
        repo = _get_repo()
        engine = _get_engine(repo)
        trades = repo.get_recent_outcomes(limit=1000)
        results = engine.compute_attribution(trades, dimension)
        return {
            "dimension": dimension,
            "days": days,
            "rows": [
                {
                    "label": r.value,
                    "pnl": r.total_pnl,
                    "trades": r.total_trades,
                    "win_rate": r.win_rate,
                    "sharpe": r.sharpe,
                    "contribution_pct": r.contribution_pct,
                }
                for r in results
            ],
        }
    except Exception as e:
        logger.warning("Attribution by-dimension error: %s", e)
        return {"dimension": dimension, "days": days, "rows": []}


@router.get("/full")
async def get_full_attribution(
    days: int = Query(30, ge=1, le=365),
    _user=Depends(require_auth),
):
    """Get full multi-dimensional attribution."""
    try:
        repo = _get_repo()
        engine = _get_engine(repo)
        trades = repo.get_recent_outcomes(limit=2000)
        all_results = engine.compute_full_attribution(trades)
        return {
            "days": days,
            "total_trades": len(trades),
            "total_pnl": sum(t.get("realized_pnl", 0) if isinstance(t, dict) else getattr(t, "realized_pnl", 0) for t in trades),
            "win_rate": 0,
            "best_model": "--",
            "dimensions": {
                dim: [
                    {
                        "label": r.value,
                        "pnl": r.total_pnl,
                        "trades": r.total_trades,
                        "win_rate": r.win_rate,
                        "sharpe": r.sharpe,
                        "contribution_pct": r.contribution_pct,
                    }
                    for r in results[:10]
                ]
                for dim, results in all_results.items()
            },
        }
    except Exception as e:
        logger.warning("Attribution full error: %s", e)
        return {"days": days, "total_trades": 0, "total_pnl": 0, "win_rate": 0, "best_model": "--", "dimensions": {}}


@router.get("/feature-importance")
async def get_feature_importance(
    top_n: int = Query(15, ge=5, le=50),
    _user=Depends(require_auth),
):
    """Get feature importance based on correlation with trade P&L."""
    try:
        repo = _get_repo()
        engine = _get_engine(repo)
        trades = repo.get_recent_outcomes(limit=1000)
        importance = engine.compute_factor_importance(trades, top_n=top_n)
        return {"features": importance, "total_trades_analyzed": len(trades)}
    except Exception as e:
        logger.warning("Feature importance error: %s", e)
        return {"features": [], "total_trades_analyzed": 0}


@router.get("/trade-outcomes")
async def get_trade_outcomes(
    limit: int = Query(50, ge=1, le=500),
    model_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    _user=Depends(require_auth),
):
    """Get recent trade outcomes with optional filtering."""
    try:
        repo = _get_repo()
        if model_id:
            outcomes = repo.get_outcomes_by_model(model_id)
        elif strategy_id:
            outcomes = repo.get_outcomes_by_strategy(strategy_id)
        else:
            outcomes = repo.get_recent_outcomes(limit=limit)
        return {"trades": outcomes[:limit]}
    except Exception as e:
        logger.warning("Trade outcomes error: %s", e)
        return {"trades": []}
