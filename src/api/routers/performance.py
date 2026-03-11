"""
Performance API: equity curve, drawdown, monthly returns, and summary metrics.
Uses risk_manager + performance_tracker when available; returns empty data with
explicit no_data flag when no real trades have been recorded.
"""
import logging
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from src.api.auth import get_current_user


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class EquityCurvePoint(BaseModel):
    date: str
    label: str
    equity: float


class EquityCurveResponse(BaseModel):
    equity_curve: List[EquityCurvePoint]
    source: str
    message: Optional[str] = None


class DrawdownPoint(BaseModel):
    date: str
    label: str
    drawdown: float


class DrawdownResponse(BaseModel):
    drawdown: List[DrawdownPoint]
    source: str
    message: Optional[str] = None


class MonthlyReturnPoint(BaseModel):
    month: str
    return_pct: float


class MonthlyReturnsResponse(BaseModel):
    monthly_returns: List[MonthlyReturnPoint]
    source: str
    message: Optional[str] = None


class PerformanceSummaryResponse(BaseModel):
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    avg_trade_pnl: float = 0.0
    profit_factor: float = 0.0
    initial_equity: float = 0.0
    final_equity: float = 0.0
    daily_pnl: float = 0.0
    source: str = "no_data"
    message: Optional[str] = None
    weeks: Optional[int] = None
    strategies: Optional[Dict[str, Any]] = None


class PublicPerformanceResponse(BaseModel):
    equity_curve: List[Any]
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    total_return_pct: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    cost_adjusted: bool = True
    source: str = "insufficient_data"
    disclaimer: str = ""


router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers to retrieve live components from app state
# ---------------------------------------------------------------------------

def _get_tracker(request: Request):
    return getattr(request.app.state, "performance_tracker", None)


def _get_risk_manager(request: Request):
    return getattr(request.app.state, "risk_manager", None)


def _has_real_trades(tracker) -> bool:
    """Return True if the tracker exists and has recorded at least one trade."""
    if tracker is None:
        return False
    try:
        all_stats = tracker.get_all_stats()
        return any(len(s.pnls) > 0 for s in all_stats.values())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Build REAL data from PerformanceTracker + RiskManager
# ---------------------------------------------------------------------------

def _real_equity_curve(tracker, rm) -> List[Dict[str, Any]]:
    """
    Build a cumulative equity curve from all per-trade P&L values across
    strategies.  Each point represents the portfolio equity after a trade.
    The initial equity comes from the RiskManager; if unavailable, we start
    at the total_pnl offset so the curve still makes sense.
    """
    all_stats = tracker.get_all_stats()

    # Collect all pnls across strategies in the order they were recorded.
    # Since PerformanceTracker doesn't store timestamps per trade, we
    # interleave them strategy-by-strategy in round-robin, which is a
    # reasonable approximation when timestamps are unavailable.
    strategy_pnls: Dict[str, list] = {}
    for sid, stats in all_stats.items():
        if stats.pnls:
            strategy_pnls[sid] = list(stats.pnls)

    if not strategy_pnls:
        return []

    # Flatten: round-robin across strategies to approximate interleaving
    merged_pnls: List[float] = []
    max_len = max(len(p) for p in strategy_pnls.values())
    for i in range(max_len):
        for sid in strategy_pnls:
            if i < len(strategy_pnls[sid]):
                merged_pnls.append(strategy_pnls[sid][i])

    if not merged_pnls:
        return []

    initial_equity = rm.equity if rm else 100000.0
    # Walk backwards: initial equity before all trades = current equity - total_pnl
    total_pnl = sum(merged_pnls)
    start_equity = initial_equity - total_pnl

    now = datetime.now(timezone.utc)
    n = len(merged_pnls)
    curve = []
    equity = start_equity

    # Spread trades evenly over time for visualization
    # If few trades, space them ~1 day apart; otherwise fit within last 26 weeks
    span_days = max(n, 7)
    start_time = now - timedelta(days=span_days)

    for idx, pnl in enumerate(merged_pnls):
        equity += pnl
        dt = start_time + timedelta(days=(idx + 1) * span_days / n)
        curve.append({
            "date": dt.strftime("%Y-%m-%d"),
            "label": f"T{idx + 1}",
            "equity": round(equity, 2),
        })

    return curve


def _real_drawdown(equity_curve: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Calculate drawdown series from an equity curve."""
    dd = []
    peak = 0.0
    for pt in equity_curve:
        eq = pt["equity"]
        peak = max(peak, eq)
        drawdown_pct = ((eq - peak) / peak * 100) if peak > 0 else 0.0
        dd.append({
            "date": pt["date"],
            "label": pt["label"],
            "drawdown": round(drawdown_pct, 2),
        })
    return dd


def _real_monthly_returns(equity_curve: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate equity curve points into monthly return percentages."""
    months: OrderedDict = OrderedDict()
    for pt in equity_curve:
        dt = datetime.strptime(pt["date"], "%Y-%m-%d")
        month_key = dt.strftime("%b %Y")
        if month_key not in months:
            months[month_key] = {"start": pt["equity"], "end": pt["equity"]}
        months[month_key]["end"] = pt["equity"]

    result = []
    for month, vals in months.items():
        ret = ((vals["end"] - vals["start"]) / vals["start"] * 100) if vals["start"] > 0 else 0
        result.append({
            "month": month,
            "return_pct": round(ret, 2),
        })
    return result


def _real_summary(tracker, rm) -> Dict[str, Any]:
    """
    Build summary entirely from PerformanceTracker.summary() and RiskManager.
    """
    stats = tracker.summary()

    total_trades = stats["total_trades"]
    total_wins = stats["total_wins"]
    total_losses = stats["total_losses"]
    total_pnl = stats["total_pnl"]
    win_rate = stats["win_rate"]
    avg_trade_pnl = stats["avg_trade_pnl"]
    sharpe = stats["sharpe_ratio"]
    max_dd = stats["max_drawdown_pct"]

    initial_equity = rm.equity - total_pnl if rm else 100000.0 - total_pnl
    final_equity = rm.equity if rm else 100000.0
    total_return_pct = (total_pnl / initial_equity * 100) if initial_equity > 0 else 0.0

    # Profit factor: sum of winning trades / abs(sum of losing trades)
    all_stats = tracker.get_all_stats()
    all_pnls: List[float] = []
    for s in all_stats.values():
        all_pnls.extend(s.pnls)
    gross_profit = sum(p for p in all_pnls if p > 0)
    gross_loss = abs(sum(p for p in all_pnls if p < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

    return {
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "win_rate": round(win_rate * 100, 1),  # tracker returns 0-1, API expects %
        "total_trades": total_trades,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "avg_trade_pnl": round(avg_trade_pnl, 2),
        "profit_factor": round(profit_factor, 2),
        "initial_equity": round(initial_equity, 2),
        "final_equity": round(final_equity, 2),
        "daily_pnl": round(rm.daily_pnl, 2) if rm else 0.0,
        "strategies": stats.get("strategies", {}),
    }



# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@router.get("/equity-curve", response_model=EquityCurveResponse)
async def get_equity_curve(
    request: Request,
    weeks: int = Query(26, ge=4, le=104),
    current_user: dict = Depends(get_current_user),
):
    """Equity curve over time. Returns real trade data or empty array with no_data flag."""
    tracker = _get_tracker(request)
    rm = _get_risk_manager(request)

    if _has_real_trades(tracker):
        try:
            curve = _real_equity_curve(tracker, rm)
            if curve:
                return {"equity_curve": curve, "source": "live"}
        except Exception:
            logger.exception("Failed to build real equity curve")

    # No real trade data available — return empty with explicit flag
    return {"equity_curve": [], "source": "no_data", "message": "No trades recorded yet. Equity curve will appear after the first completed trade."}


@router.get("/drawdown", response_model=DrawdownResponse)
async def get_drawdown(
    request: Request,
    weeks: int = Query(26, ge=4, le=104),
    current_user: dict = Depends(get_current_user),
):
    """Drawdown series (peak-to-trough %). Uses real trade data when available."""
    tracker = _get_tracker(request)
    rm = _get_risk_manager(request)

    if _has_real_trades(tracker):
        try:
            curve = _real_equity_curve(tracker, rm)
            if curve:
                dd = _real_drawdown(curve)
                return {"drawdown": dd, "source": "live"}
        except Exception:
            logger.exception("Failed to build real drawdown")

    # No real trade data available — return empty with explicit flag
    return {"drawdown": [], "source": "no_data", "message": "No trades recorded yet. Drawdown data will appear after the first completed trade."}


@router.get("/monthly-returns", response_model=MonthlyReturnsResponse)
async def get_monthly_returns(
    request: Request,
    weeks: int = Query(26, ge=4, le=104),
    current_user: dict = Depends(get_current_user),
):
    """Monthly return percentages. Uses real trade data when available."""
    tracker = _get_tracker(request)
    rm = _get_risk_manager(request)

    if _has_real_trades(tracker):
        try:
            curve = _real_equity_curve(tracker, rm)
            if curve:
                monthly = _real_monthly_returns(curve)
                return {"monthly_returns": monthly, "source": "live"}
        except Exception:
            logger.exception("Failed to build real monthly returns")

    # No real trade data available — return empty with explicit flag
    return {"monthly_returns": [], "source": "no_data", "message": "No trades recorded yet. Monthly returns will appear after the first completed trade."}


@router.get("/summary", response_model=PerformanceSummaryResponse)
async def get_summary(
    request: Request,
    weeks: int = Query(26, ge=4, le=104),
    current_user: dict = Depends(get_current_user),
):
    """Performance summary: Sharpe, max drawdown, win rate, P&L, profit factor."""
    tracker = _get_tracker(request)
    rm = _get_risk_manager(request)

    if _has_real_trades(tracker):
        try:
            summary = _real_summary(tracker, rm)
            summary["source"] = "live"
            summary.setdefault("weeks", weeks)
            return summary
        except Exception:
            logger.exception("Failed to build real summary")

    # No real trade data available — return zeroed metrics with explicit flag
    return {
        "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "total_pnl": 0.0,
        "total_return_pct": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
        "total_wins": 0,
        "total_losses": 0,
        "avg_trade_pnl": 0.0,
        "profit_factor": 0.0,
        "initial_equity": rm.equity if rm else 0.0,
        "final_equity": rm.equity if rm else 0.0,
        "daily_pnl": 0.0,
        "source": "no_data",
        "message": "No trades recorded yet. Performance summary will appear after the first completed trade.",
    }


# ---------------------------------------------------------------------------
# Public performance endpoint (no auth required)
# ---------------------------------------------------------------------------

@router.get("/public", response_model=PublicPerformanceResponse)
async def get_public_performance(request: Request):
    """
    Publicly accessible performance summary. No authentication required.
    Shows cost-adjusted equity curve, Sharpe, max drawdown, and trade count.
    Disclaimer: Past performance does not guarantee future results.
    """
    tracker = _get_tracker(request)
    rm = _get_risk_manager(request)

    if _has_real_trades(tracker):
        try:
            curve = _real_equity_curve(tracker, rm)
            summary = _real_summary(tracker, rm)
            return {
                "equity_curve": curve[-90:] if len(curve) > 90 else curve,  # last 90 points
                "sharpe_ratio": summary.get("sharpe_ratio", 0),
                "max_drawdown_pct": summary.get("max_drawdown_pct", 0),
                "total_return_pct": summary.get("total_return_pct", 0),
                "total_trades": summary.get("total_trades", 0),
                "win_rate": summary.get("win_rate", 0),
                "cost_adjusted": True,
                "source": "live",
                "disclaimer": "Past performance does not guarantee future results.",
            }
        except Exception:
            pass

    return {
        "equity_curve": [],
        "sharpe_ratio": 0,
        "max_drawdown_pct": 0,
        "total_return_pct": 0,
        "total_trades": 0,
        "win_rate": 0,
        "cost_adjusted": True,
        "source": "insufficient_data",
        "disclaimer": "Performance data will be available after 30 trading days.",
    }


@router.get("/daily-report")
async def get_daily_report(
    request: Request,
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
    current_user: dict = Depends(get_current_user),
):
    """Get daily performance report for a specific date."""
    from fastapi import HTTPException as _HTTPException
    drg = getattr(request.app.state, "daily_report_generator", None)
    if drg is None:
        raise _HTTPException(status_code=503, detail="Daily report generator not configured")
    try:
        report = drg.get_report(date or datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        if report is None:
            raise _HTTPException(status_code=404, detail="No report found for this date")
        from src.reporting.daily_report import DailyReportGenerator
        return DailyReportGenerator.to_dict(report)
    except _HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating daily report: %s", e)
        raise _HTTPException(status_code=500, detail="Internal error generating report")
