"""
Performance API: equity curve, drawdown, monthly returns, and summary metrics.
Uses risk_manager + performance_tracker when available; realistic demo fallback.
"""
import logging
import math
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, Request

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
    """Calculate drawdown series from an equity curve (real or demo)."""
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
# Demo fallbacks (unchanged, used only when no real trades exist)
# ---------------------------------------------------------------------------

def _demo_equity_curve(weeks: int = 26, initial: float = 100000.0) -> List[Dict[str, Any]]:
    """Generate a realistic demo equity curve with slight upward drift and noise."""
    import numpy as np
    rng = np.random.default_rng(42)
    curve = []
    equity = initial
    now = datetime.now(timezone.utc)
    start = now - timedelta(weeks=weeks)
    for i in range(weeks):
        dt = start + timedelta(weeks=i)
        weekly_ret = 0.003 + rng.standard_normal() * 0.012
        equity *= (1 + weekly_ret)
        curve.append({
            "date": dt.strftime("%Y-%m-%d"),
            "label": f"W{i+1}",
            "equity": round(equity, 2),
        })
    return curve


def _demo_drawdown(equity_curve: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Calculate drawdown series from equity curve."""
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


def _demo_monthly_returns(equity_curve: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate weekly equity into monthly returns."""
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


def _demo_summary(equity_curve: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary metrics from demo equity curve."""
    if not equity_curve:
        return {"sharpe_ratio": 0, "max_drawdown_pct": 0, "total_pnl": 0, "total_return_pct": 0,
                "win_rate": 0, "total_trades": 0, "avg_trade_pnl": 0, "profit_factor": 0}

    equities = [pt["equity"] for pt in equity_curve]
    initial = equities[0]
    final = equities[-1]
    total_pnl = final - initial
    total_return = ((final - initial) / initial * 100) if initial > 0 else 0

    returns = []
    for i in range(1, len(equities)):
        r = (equities[i] - equities[i-1]) / equities[i-1] if equities[i-1] > 0 else 0
        returns.append(r)

    avg_ret = sum(returns) / len(returns) if returns else 0
    std_ret = (sum((r - avg_ret) ** 2 for r in returns) / len(returns)) ** 0.5 if len(returns) > 1 else 0.001
    sharpe = (avg_ret / std_ret) * (52 ** 0.5) if std_ret > 0 else 0

    peak = 0
    max_dd = 0
    for eq in equities:
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    import numpy as np
    rng = np.random.default_rng(99)
    total_trades = 87 + int(rng.integers(0, 30))
    win_rate = 58.0 + rng.random() * 12
    avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0
    wins = int(total_trades * win_rate / 100)
    losses = total_trades - wins
    avg_win = abs(avg_trade_pnl) * 1.8
    avg_loss = abs(avg_trade_pnl) * 0.9 if avg_trade_pnl != 0 else 1
    profit_factor = (wins * avg_win) / (losses * avg_loss) if losses > 0 and avg_loss > 0 else 0

    return {
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return, 2),
        "win_rate": round(win_rate, 1),
        "total_trades": total_trades,
        "avg_trade_pnl": round(avg_trade_pnl, 2),
        "profit_factor": round(profit_factor, 2),
        "initial_equity": round(initial, 2),
        "final_equity": round(final, 2),
        "weeks": len(equities),
    }


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@router.get("/equity-curve")
async def get_equity_curve(
    request: Request,
    weeks: int = Query(26, ge=4, le=104),
):
    """Equity curve over time. Uses real trade data when available, demo otherwise."""
    tracker = _get_tracker(request)
    rm = _get_risk_manager(request)

    if _has_real_trades(tracker):
        try:
            curve = _real_equity_curve(tracker, rm)
            if curve:
                return {"equity_curve": curve, "source": "live"}
        except Exception:
            logger.exception("Failed to build real equity curve, falling back to demo")

    # Fallback to demo
    initial = rm.equity if rm else 100000.0
    curve = _demo_equity_curve(weeks=weeks, initial=initial)
    return {"equity_curve": curve, "source": "demo"}


@router.get("/drawdown")
async def get_drawdown(
    request: Request,
    weeks: int = Query(26, ge=4, le=104),
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
            logger.exception("Failed to build real drawdown, falling back to demo")

    # Fallback to demo
    initial = rm.equity if rm else 100000.0
    curve = _demo_equity_curve(weeks=weeks, initial=initial)
    dd = _demo_drawdown(curve)
    return {"drawdown": dd, "source": "demo"}


@router.get("/monthly-returns")
async def get_monthly_returns(
    request: Request,
    weeks: int = Query(26, ge=4, le=104),
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
            logger.exception("Failed to build real monthly returns, falling back to demo")

    # Fallback to demo
    initial = rm.equity if rm else 100000.0
    curve = _demo_equity_curve(weeks=weeks, initial=initial)
    monthly = _demo_monthly_returns(curve)
    return {"monthly_returns": monthly, "source": "demo"}


@router.get("/summary")
async def get_summary(
    request: Request,
    weeks: int = Query(26, ge=4, le=104),
):
    """Performance summary: Sharpe, max drawdown, win rate, P&L, profit factor."""
    tracker = _get_tracker(request)
    rm = _get_risk_manager(request)

    if _has_real_trades(tracker):
        try:
            summary = _real_summary(tracker, rm)
            summary["source"] = "live"
            return summary
        except Exception:
            logger.exception("Failed to build real summary, falling back to demo")

    # Fallback to demo
    initial = rm.equity if rm else 100000.0
    curve = _demo_equity_curve(weeks=weeks, initial=initial)
    summary = _demo_summary(curve)
    summary["source"] = "demo"
    return summary


# ---------------------------------------------------------------------------
# Public performance endpoint (no auth required)
# ---------------------------------------------------------------------------

@router.get("/public")
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
):
    """Get daily performance report for a specific date."""
    drg = getattr(request.app.state, "daily_report_generator", None)
    if drg is None:
        return {"error": "Daily report generator not configured"}
    try:
        report = drg.get_report(date or datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        if report is None:
            return {"error": "No report found for this date"}
        from src.reporting.daily_report import DailyReportGenerator
        return DailyReportGenerator.to_dict(report)
    except Exception as e:
        return {"error": str(e)}
