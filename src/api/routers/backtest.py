"""
Backtest API: run backtests via BacktestEngine, store results, return job status and equity.
"""
import asyncio
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel

from src.api.auth import get_current_user

from src.core.events import Bar, Exchange

router = APIRouter()
logger = logging.getLogger(__name__)

# Job store: job_id -> { status, body, result (equity_curve, metrics, trades), error }
_backtest_jobs: Dict[str, Dict[str, Any]] = {}
_backtest_lock = threading.Lock()
MAX_JOBS = 100  # Limit stored jobs to prevent unbounded memory growth


def _evict_oldest_completed_jobs() -> None:
    """Remove oldest completed/failed jobs when the store exceeds MAX_JOBS.
    Must be called while _backtest_lock is held."""
    if len(_backtest_jobs) <= MAX_JOBS:
        return
    # Collect completed/failed job ids (insertion-ordered since Python 3.7)
    completed = [jid for jid, data in _backtest_jobs.items() if data["status"] in ("completed", "failed")]
    # Evict oldest completed first
    to_remove = len(_backtest_jobs) - MAX_JOBS
    for jid in completed[:to_remove]:
        _backtest_jobs.pop(jid, None)


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


def _fetch_bars_for_backtest(
    symbol: str,
    exchange: str,
    start: datetime,
    end: datetime,
    interval: str = "1d",
) -> List[Bar]:
    """Fetch OHLCV bars for backtest using Yahoo Finance. Returns empty list on failure."""
    exch = Exchange(exchange) if exchange in ("NSE", "BSE", "NYSE", "NASDAQ", "LSE", "FX") else Exchange.NSE
    try:
        from src.market_data.connectors.yahoo_finance import get_yahoo_connector
        yf = get_yahoo_connector()
        if yf is not None:
            raw_bars = yf.get_bars(
                symbol,
                exchange,
                interval,
                limit=2000,
                from_ts=start.strftime("%Y-%m-%d"),
                to_ts=end.strftime("%Y-%m-%d"),
            )
            if raw_bars:
                bars: List[Bar] = []
                for b in raw_bars:
                    try:
                        ts = b.get("ts", "")
                        if isinstance(ts, str):
                            # Parse ISO format; fall back to date-only
                            for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d"):
                                try:
                                    ts_dt = datetime.strptime(ts[:26].rstrip("+").rstrip("-"), fmt.rstrip("%z"))
                                    ts_dt = ts_dt.replace(tzinfo=timezone.utc)
                                    break
                                except ValueError:
                                    continue
                            else:
                                ts_dt = start  # fallback
                        else:
                            ts_dt = ts if hasattr(ts, "year") else start

                        bars.append(Bar(
                            symbol=symbol,
                            exchange=exch,
                            interval=interval,
                            open=float(b.get("open", 0)),
                            high=float(b.get("high", 0)),
                            low=float(b.get("low", 0)),
                            close=float(b.get("close", 0)),
                            volume=float(b.get("volume", 0)),
                            ts=ts_dt,
                            source="yahoo_finance",
                        ))
                    except Exception:
                        continue
                if bars:
                    logger.info("Fetched %d bars from Yahoo Finance for backtest %s", len(bars), symbol)
                    return bars
    except ImportError:
        logger.debug("Yahoo Finance connector not available for backtest")
    except Exception as e:
        logger.warning("Yahoo Finance bar fetch failed for backtest %s: %s", symbol, e)
    return []


def _run_backtest_sync(job_id: str, body: dict) -> None:
    """Run backtest in sync (called from thread)."""
    global _backtest_jobs
    try:
        from src.api.routers.strategies import get_registry
        from src.backtesting.engine import BacktestEngine, BacktestConfig

        reg = get_registry()
        strategy_id = body.get("strategy_id")
        if not strategy_id:
            with _backtest_lock:
                _backtest_jobs[job_id]["status"] = "failed"
                _backtest_jobs[job_id]["error"] = "strategy_id is required"
            return
        strategy = reg.get(strategy_id)
        if strategy is None:
            with _backtest_lock:
                _backtest_jobs[job_id]["status"] = "failed"
                _backtest_jobs[job_id]["error"] = f"Strategy not found: {strategy_id}"
            return

        start = _parse_date(body["start"])
        end = _parse_date(body["end"])
        if start >= end:
            with _backtest_lock:
                _backtest_jobs[job_id]["status"] = "failed"
                _backtest_jobs[job_id]["error"] = "start must be before end"
            return

        bars = _fetch_bars_for_backtest(
            body["symbol"],
            body.get("exchange", "NSE"),
            start,
            end,
            body.get("interval", "1d"),
        )
        if len(bars) < 30:
            with _backtest_lock:
                _backtest_jobs[job_id]["status"] = "failed"
                _backtest_jobs[job_id]["error"] = (
                    f"Insufficient market data for {body['symbol']}: got {len(bars)} bars, "
                    f"need at least 30. Ensure Yahoo Finance is available and the symbol/date range is valid."
                )
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
                "sharpe_ratio": metrics.sharpe,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "win_rate_pct": metrics.win_rate_pct,
                "win_rate": metrics.win_rate_pct,
                "num_trades": len(result.trades),
            }
            if getattr(metrics, "risk_metrics", None):
                rm = metrics.risk_metrics
                metrics_dict["var_95"] = getattr(rm, "var_95", 0)
                metrics_dict["sharpe"] = getattr(rm, "sharpe", metrics.sharpe)

        with _backtest_lock:
            _backtest_jobs[job_id]["status"] = "completed"
            _backtest_jobs[job_id]["result"] = {
                "equity_curve": result.equity_curve,
                "metrics": metrics_dict,
                "trades": result.trades[:100],
                "num_trades": len(result.trades),
            }
    except Exception as e:
        logger.exception("Backtest run failed")
        with _backtest_lock:
            _backtest_jobs[job_id]["status"] = "failed"
            _backtest_jobs[job_id]["error"] = "Backtest execution failed"


class BacktestRunRequest(BaseModel):
    strategy_id: str
    symbol: str
    exchange: str = "NSE"
    start: str
    end: str
    config: Optional[dict] = None
    interval: str = "1d"


@router.post("/run")
async def run_backtest(request: Request, body: BacktestRunRequest, background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user)):
    """Queue and run backtest; returns job_id. Runs in background."""
    try:
        _parse_date(body.start)
        _parse_date(body.end)
    except ValueError as e:
        raise HTTPException(400, str(e))

    job_id = f"bt_{uuid.uuid4().hex[:12]}"
    with _backtest_lock:
        _evict_oldest_completed_jobs()
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
async def list_backtest_jobs(current_user: dict = Depends(get_current_user)):
    """List all backtest jobs (id, status)."""
    with _backtest_lock:
        jobs = [
            {"job_id": jid, "status": data["status"], "body": data.get("body")}
            for jid, data in list(_backtest_jobs.items())
        ]
    return {"jobs": jobs[-50:]}


@router.get("/jobs/{job_id}")
async def get_backtest_job(job_id: str, current_user: dict = Depends(get_current_user)):
    """Get job status and metrics."""
    with _backtest_lock:
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
async def get_backtest_equity(job_id: str, current_user: dict = Depends(get_current_user)):
    """Get equity curve for job."""
    with _backtest_lock:
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
async def get_backtest_trades(job_id: str, limit: int = 100, current_user: dict = Depends(get_current_user)):
    """Get trades for completed job."""
    with _backtest_lock:
        data = _backtest_jobs.get(job_id)
    if not data:
        raise HTTPException(404, "Job not found")
    if data["status"] != "completed" or not data.get("result"):
        return {"job_id": job_id, "trades": []}
    trades = data["result"].get("trades", [])[:limit]
    return {"job_id": job_id, "trades": trades}
