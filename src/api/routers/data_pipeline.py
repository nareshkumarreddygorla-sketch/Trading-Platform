"""
Data Pipeline API: trigger data refresh, check data quality, manage OHLCV data,
real-time validation statistics, and data staleness monitoring.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request

from src.api.auth import get_current_user as require_auth

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/data", tags=["data"])


def _get_quality_monitor(request: Request):
    """Get the DataQualityMonitor from app state, or create a default."""
    monitor = getattr(request.app.state, "data_quality_monitor", None)
    if monitor is None:
        from src.data_pipeline.data_quality_monitor import DataQualityMonitor

        monitor = DataQualityMonitor()
        request.app.state.data_quality_monitor = monitor
    return monitor


def _get_reconciliator(request: Request):
    """Get the DataReconciliator from app state, or create a default."""
    reconciliator = getattr(request.app.state, "data_reconciliator", None)
    if reconciliator is None:
        from src.data_pipeline.data_reconciliation import DataReconciliator

        reconciliator = DataReconciliator()
        request.app.state.data_reconciliator = reconciliator
    return reconciliator


# ── Existing endpoints ──────────────────────────────────────────────────────


@router.post("/refresh")
async def trigger_data_refresh(
    background_tasks: BackgroundTasks,
    _user=Depends(require_auth),
):
    """Trigger nightly data refresh pipeline (runs in background)."""

    async def _run():
        from scripts.nightly_data_refresh import main

        await main()

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Data refresh pipeline started"}


@router.get("/quality")
async def check_data_quality(request: Request, _user=Depends(require_auth)):
    """
    Current data quality scores across all tracked symbols.

    Returns per-symbol quality scores (0-100), overall average, halted symbols,
    and quality level distribution. Falls back to OHLCV repo check if no
    live monitoring data is available.
    """
    try:
        monitor = _get_quality_monitor(request)
        summary = monitor.get_summary()
        all_scores = monitor.get_all_quality_scores()

        # If we have live monitoring data, return it
        if summary["total_symbols"] > 0:
            return {
                "status": "live_monitoring",
                "average_quality_score": summary["average_quality_score"],
                "total_symbols": summary["total_symbols"],
                "halted_symbols": summary["halted_symbols"],
                "halted_count": summary["halted_count"],
                "quality_levels": summary["quality_levels"],
                "quality_threshold": summary["quality_threshold"],
                "halt_threshold": summary["halt_threshold"],
                "symbols": all_scores,
            }

        # Fallback: check OHLCV repo for basic quality info
        try:
            from src.persistence.ohlcv_repo import OHLCVRepository

            repo = OHLCVRepository()
            symbols = repo.get_symbols_with_data(interval="1d", min_bars=50)
            return {
                "status": "static_check",
                "symbols_with_data": len(symbols),
                "symbols": symbols[:50],
                "quality": "good" if len(symbols) >= 20 else "insufficient",
                "average_quality_score": 80.0 if len(symbols) >= 20 else 40.0,
                "total_symbols": len(symbols),
            }
        except Exception:
            return {
                "status": "no_data",
                "average_quality_score": 0.0,
                "total_symbols": 0,
                "quality": "unknown",
            }

    except Exception:
        logger.exception("Data quality check failed")
        return {"error": "Data quality check failed", "quality": "unknown"}


@router.get("/quality/{symbol}")
async def get_symbol_quality(
    symbol: str,
    request: Request,
    _user=Depends(require_auth),
):
    """
    Per-symbol data quality details.

    Returns quality score, tick/bar acceptance rates, freshness timestamps,
    gap counts, missing bar counts, and trading halt status.
    """
    try:
        monitor = _get_quality_monitor(request)
        quality = monitor.get_symbol_quality(symbol.upper())

        # Add tick and bar validator details
        tick_stats = monitor.tick_validator.get_stats(symbol.upper())
        ohlc_stats = monitor.ohlc_validator.get_stats(symbol.upper())

        return {
            **quality,
            "tick_validation": tick_stats,
            "ohlc_validation": ohlc_stats,
            "alerts": [a for a in monitor.get_alerts(limit=20) if a.get("symbol") == symbol.upper()],
        }
    except Exception:
        logger.exception("Symbol quality check failed for %s", symbol)
        return {"error": f"Quality check failed for {symbol}", "symbol": symbol}


@router.get("/validation-stats")
async def get_validation_stats(request: Request, _user=Depends(require_auth)):
    """
    Validation statistics for tick and OHLC validators.

    Returns aggregate rejection counts by reason, acceptance rates, and
    per-symbol breakdowns.
    """
    try:
        monitor = _get_quality_monitor(request)
        return monitor.get_validation_stats()
    except Exception:
        logger.exception("Validation stats retrieval failed")
        return {"error": "Validation stats retrieval failed"}


@router.get("/staleness")
async def get_staleness_report(request: Request, _user=Depends(require_auth)):
    """
    Data freshness report across all symbols and sources.

    Returns per-symbol last data time, staleness in seconds, and whether
    each feed is considered stale. Also includes per-source freshness.
    """
    try:
        monitor = _get_quality_monitor(request)
        report = monitor.get_staleness_report()

        # Add reconciliation source info if available
        reconciliator = _get_reconciliator(request)
        report["source_reliability"] = reconciliator.get_source_reliability()
        report["reconciliation_summary"] = reconciliator.get_reconciliation_summary()

        return report
    except Exception:
        logger.exception("Staleness report failed")
        return {"error": "Staleness report failed"}


@router.get("/alerts")
async def get_quality_alerts(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    _user=Depends(require_auth),
):
    """Get recent data quality alerts."""
    try:
        monitor = _get_quality_monitor(request)
        return {
            "alerts": monitor.get_alerts(limit=limit),
            "total_alerts": len(monitor._alerts),
        }
    except Exception:
        logger.exception("Alert retrieval failed")
        return {"error": "Alert retrieval failed"}


@router.get("/reconciliation")
async def get_reconciliation_report(request: Request, _user=Depends(require_auth)):
    """Get data reconciliation summary across sources."""
    try:
        reconciliator = _get_reconciliator(request)
        return reconciliator.get_reconciliation_summary()
    except Exception:
        logger.exception("Reconciliation report failed")
        return {"error": "Reconciliation report failed"}


# ── Existing endpoints (preserved) ──────────────────────────────────────────


@router.get("/symbols")
async def get_available_symbols(
    interval: str = Query("1d"),
    min_bars: int = Query(50, ge=1),
    _user=Depends(require_auth),
):
    """Get symbols with sufficient historical data."""
    try:
        from src.persistence.ohlcv_repo import OHLCVRepository

        repo = OHLCVRepository()
        symbols = repo.get_symbols_with_data(interval=interval, min_bars=min_bars)
        return {"count": len(symbols), "symbols": symbols}
    except Exception:
        logger.exception("Failed to retrieve available symbols")
        return {"error": "Failed to retrieve available symbols"}


@router.get("/bars/{symbol}")
async def get_bars(
    symbol: str,
    interval: str = Query("1d"),
    limit: int = Query(100, ge=1, le=1000),
    _user=Depends(require_auth),
):
    """Get OHLCV bars for a symbol."""
    try:
        from src.persistence.ohlcv_repo import OHLCVRepository

        repo = OHLCVRepository()
        bars = repo.get_bars(symbol.upper(), interval=interval, limit=limit)
        return {
            "symbol": symbol.upper(),
            "interval": interval,
            "count": len(bars),
            "bars": [
                {
                    "timestamp": b.timestamp.isoformat() if hasattr(b.timestamp, "isoformat") else str(b.timestamp),
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                }
                for b in bars
            ],
        }
    except Exception:
        logger.exception("Failed to retrieve bars")
        return {"error": "Failed to retrieve bars"}


@router.get("/instrument-map")
async def get_instrument_info(
    symbol: str | None = None,
    _user=Depends(require_auth),
):
    """Get Angel One instrument info / symbol token mapping."""
    try:
        from src.market_data.symbol_token_map import get_symbol_token_map

        stm = get_symbol_token_map()
        if not stm.is_loaded:
            return {"loaded": False, "message": "Instrument map not loaded. Run data refresh first."}

        if symbol:
            info = stm.get_instrument_info(symbol.upper())
            return {"symbol": symbol.upper(), "info": info}

        return {
            "loaded": True,
            "instrument_count": stm.instrument_count,
            "nse_equity_count": len(stm.get_all_nse_equity_symbols()),
        }
    except Exception:
        logger.exception("Failed to retrieve instrument info")
        return {"error": "Failed to retrieve instrument info"}
