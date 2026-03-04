"""
Backtest API: run backtests via BacktestEngine, store results, return job status and equity.
"""
import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel

from src.core.events import Bar, Exchange

router = APIRouter()
logger = logging.getLogger(__name__)

# Job store: job_id -> { status, body, result (equity_curve, metrics, trades), error }
_backtest_jobs: Dict[str, Dict[str, Any]] = {}
_backtest_lock = asyncio.Lock()


def _parse_date(s: str) -> datetime:
    s = (s or "").strip()
    if not s:
        raise ValueError("Empty date")
    for fmt, size in (("%Y-%m-%d", 10), ("%Y-%m-%dT%H:%M:%S", 19)):
        try:
            return datetime.strptime(s[:size], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Invalid date: {s}")


def _generate_synthetic_bars(
    symbol: str,
    exchange: str,
    start: datetime,
    end: datetime,
    interval: str = "1d",
    seed: int = 42,
) -> List[Bar]:
    """Generate synthetic OHLCV bars for backtest when no market data store."""
    import numpy as np
    rng = np.random.default_rng(seed)
    delta = timedelta(days=1) if interval == "1d" else timedelta(hours=1)
    bars: List[Bar] = []
    price = 100.0
    current = start
    exch = Exchange(exchange) if exchange in ("NSE", "BSE", "NYSE", "NASDAQ", "LSE", "FX") else Exchange.NSE
    while current <= end and len(bars) < 2000:
        ret = rng.standard_normal() * 0.02
        price = price * (1 + ret)
        o, c = price, price
        h = max(o, c) + abs(rng.standard_normal() * 0.5)
        l = min(o, c) - abs(rng.standard_normal() * 0.5)
        vol = max(1000, int(rng.exponential(50000)))
        bars.append(Bar(
            symbol=symbol,
            exchange=exch,
            interval=interval,
            open=round(o, 2),
            high=round(h, 2),
            low=round(l, 2),
            close=round(c, 2),
            volume=float(vol),
            ts=current,
            source="synthetic",
        ))
        current += delta
    return bars


def _run_backtest_sync(job_id: str, body: dict) -> None:
    """Run backtest in sync (called from thread)."""
    global _backtest_jobs
    try:
        from src.api.routers.strategies import get_registry
        from src.backtesting.engine import BacktestEngine, BacktestConfig
        from src.backtesting.metrics import BacktestMetrics

        reg = get_registry()
        strategy_id = body.get("strategy_id")
        if not strategy_id:
            _backtest_jobs[job_id]["status"] = "failed"
            _backtest_jobs[job_id]["error"] = "strategy_id is required"
            return
        strategy = reg.get(strategy_id)
        if strategy is None:
            _backtest_jobs[job_id]["status"] = "failed"
            _backtest_jobs[job_id]["error"] = f"Strategy not found: {strategy_id}"
            return

        start = _parse_date(body["start"])
        end = _parse_date(body["end"])
        if start >= end:
            _backtest_jobs[job_id]["status"] = "failed"
            _backtest_jobs[job_id]["error"] = "start must be before end"
            return

        bars = _generate_synthetic_bars(
            body["symbol"],
            body.get("exchange", "NSE"),
            start,
            end,
            body.get("interval", "1d"),
        )
        if len(bars) < 30:
            _backtest_jobs[job_id]["status"] = "failed"
            _backtest_jobs[job_id]["error"] = "Not enough bars generated"
            return

        config = BacktestConfig(
            initial_capital=body.get("config", {}).get("initial_capital", 100_000.0),
        )
        engine = BacktestEngine(config=config)
        exch = Exchange(body.get("exchange", "NSE")) if body.get("exchange") in ("NSE", "BSE", "NYSE", "NASDAQ", "LSE", "FX") else Exchange.NSE
        result = engine.run(strategy, bars, body["symbol"], exchange=exch)

        metrics = result.metrics
        metrics_dict: Dict[str, Any] = {}
        if metrics:
            metrics_dict = {
                "total_return_pct": metrics.total_return_pct,
                "cagr_pct": metrics.cagr_pct,
                "sharpe": metrics.sharpe,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "win_rate_pct": metrics.win_rate_pct,
                "num_trades": len(result.trades),
            }
            if getattr(metrics, "risk_metrics", None):
                rm = metrics.risk_metrics
                metrics_dict["var_95"] = getattr(rm, "var_95", 0)
                metrics_dict["sharpe"] = getattr(rm, "sharpe", metrics.sharpe)

        _backtest_jobs[job_id]["status"] = "completed"
        _backtest_jobs[job_id]["result"] = {
            "equity_curve": result.equity_curve,
            "metrics": metrics_dict,
            "trades": result.trades[:100],
            "num_trades": len(result.trades),
        }
    except Exception as e:
        logger.exception("Backtest run failed: %s", e)
        _backtest_jobs[job_id]["status"] = "failed"
        _backtest_jobs[job_id]["error"] = str(e)


class BacktestRunRequest(BaseModel):
    strategy_id: str
    symbol: str
    exchange: str = "NSE"
    start: str
    end: str
    config: Optional[dict] = None
    interval: str = "1d"


@router.post("/run")
async def run_backtest(request: Request, body: BacktestRunRequest, background_tasks: BackgroundTasks):
    """Queue and run backtest; returns job_id. Runs in background."""
    try:
        _parse_date(body.start)
        _parse_date(body.end)
    except ValueError as e:
        raise HTTPException(400, str(e))

    job_id = f"bt_{uuid.uuid4().hex[:12]}"
    async with _backtest_lock:
        _backtest_jobs[job_id] = {
            "status": "running",
            "body": body.model_dump(),
            "result": None,
            "error": None,
        }

    async def _run_in_thread():
        if hasattr(asyncio, "to_thread"):
            await asyncio.to_thread(_run_backtest_sync, job_id, body.model_dump())
        else:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: _run_backtest_sync(job_id, body.model_dump()))

    background_tasks.add_task(_run_in_thread)
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs")
async def list_backtest_jobs():
    """List all backtest jobs (id, status)."""
    async with _backtest_lock:
        jobs = [
            {"job_id": jid, "status": data["status"], "body": data.get("body")}
            for jid, data in list(_backtest_jobs.items())
        ]
    return {"jobs": jobs[-50:]}


@router.get("/jobs/{job_id}")
async def get_backtest_job(job_id: str):
    """Get job status and metrics."""
    async with _backtest_lock:
        data = _backtest_jobs.get(job_id)
    if not data:
        raise HTTPException(404, "Job not found")
    out = {"job_id": job_id, "status": data["status"], "body": data.get("body")}
    if data.get("error"):
        out["error"] = data["error"]
    if data.get("result"):
        out["metrics"] = data["result"].get("metrics", {})
        out["num_trades"] = data["result"].get("num_trades", 0)
    return out


@router.get("/jobs/{job_id}/equity")
async def get_backtest_equity(job_id: str):
    """Get equity curve for job."""
    async with _backtest_lock:
        data = _backtest_jobs.get(job_id)
    if not data:
        raise HTTPException(404, "Job not found")
    if data["status"] != "completed" or not data.get("result"):
        return {"job_id": job_id, "status": data["status"], "equity_curve": []}
    return {
        "job_id": job_id,
        "status": "completed",
        "equity_curve": data["result"].get("equity_curve", []),
    }


@router.get("/jobs/{job_id}/trades")
async def get_backtest_trades(job_id: str, limit: int = 100):
    """Get trades for completed job."""
    async with _backtest_lock:
        data = _backtest_jobs.get(job_id)
    if not data:
        raise HTTPException(404, "Job not found")
    if data["status"] != "completed" or not data.get("result"):
        return {"job_id": job_id, "trades": []}
    trades = data["result"].get("trades", [])[:limit]
    return {"job_id": job_id, "trades": trades}
