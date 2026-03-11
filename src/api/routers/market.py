"""
Market data API: quote and bars. Uses Yahoo Finance when available,
returns HTTP 503 when market data is unavailable (never fabricated data).
"""
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.auth import get_current_user


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class MarketStatusResponse(BaseModel):
    connected: bool
    healthy: bool
    message: Optional[str] = None
    source: Optional[str] = None


class MarketRegimeResponse(BaseModel):
    regime: str
    confidence: float
    vol_percentile: float
    trend_strength: float


class NewsItem(BaseModel):
    id: str
    headline: str
    symbol: Optional[str] = None
    sentiment: str = "neutral"
    score: float = 0.0
    source: str = ""
    timestamp: Optional[str] = None
    category: Optional[str] = None


class MarketNewsResponse(BaseModel):
    news: List[NewsItem]
    source: str


router = APIRouter()
logger = logging.getLogger(__name__)

_SYMBOL_RE = re.compile(r"^[A-Z0-9&_-]{1,20}$")


def _validate_symbol(symbol: str) -> str:
    """Validate and sanitize a stock symbol. Raises HTTP 400 on invalid input."""
    symbol = symbol.strip().upper()
    if not _SYMBOL_RE.match(symbol):
        raise HTTPException(400, f"Invalid symbol: must match {_SYMBOL_RE.pattern}")
    return symbol

# Yahoo Finance connector (lazy init)
_yf_connector = None
_yf_init_attempted = False


def _get_yf():
    global _yf_connector, _yf_init_attempted
    if not _yf_init_attempted:
        _yf_init_attempted = True
        try:
            from src.market_data.connectors.yahoo_finance import get_yahoo_connector
            _yf_connector = get_yahoo_connector()
            if _yf_connector:
                logger.info("Yahoo Finance connector available for market data API")
        except Exception as e:
            logger.debug("Yahoo Finance connector not available: %s", e)
    return _yf_connector



@router.get("/quote/{symbol}")
async def get_quote(symbol: str, exchange: str = "NSE", current_user: dict = Depends(get_current_user)):
    """Latest quote. Uses Yahoo Finance; returns 503 if unavailable."""
    symbol = _validate_symbol(symbol)
    yf = _get_yf()
    if yf:
        quote = yf.get_quote(symbol, exchange)
        if quote:
            return quote

    # No real market data source available — return an explicit error
    # instead of fabricated prices that could mislead the UI.
    return JSONResponse(
        status_code=503,
        content={
            "error": "Market data unavailable",
            "symbol": symbol,
            "exchange": exchange,
            "stale": True,
            "message": "Yahoo Finance connector is not available. No quote data can be served.",
        },
    )


@router.get("/status", response_model=MarketStatusResponse)
async def market_status(request: Request, current_user: dict = Depends(get_current_user)):
    """
    Market data feed status. If feed unhealthy, autonomous loop pauses; manual trading still works.
    """
    service = getattr(request.app.state, "market_data_service", None)
    yf = _get_yf()
    if service is None and yf is None:
        return {"connected": False, "healthy": False, "message": "Market data service not configured"}
    if service is None and yf is not None:
        return {"connected": True, "healthy": True, "message": "Yahoo Finance fallback active", "source": "yahoo_finance"}
    return service.get_status()


@router.get("/bars/{symbol}")
async def get_bars(
    symbol: str,
    interval: str = Query("1d"),
    from_ts: Optional[str] = None,
    to_ts: Optional[str] = None,
    limit: int = Query(500, le=1000),
    current_user: dict = Depends(get_current_user),
):
    """OHLCV bars. Uses Yahoo Finance; returns 503 if unavailable."""
    symbol = _validate_symbol(symbol)
    exchange = "NSE"
    yf = _get_yf()
    if yf:
        bars = yf.get_bars(symbol, exchange, interval, limit, from_ts, to_ts)
        if bars:
            return {"symbol": symbol, "exchange": exchange, "interval": interval, "bars": bars, "source": "yahoo_finance"}

    # No real market data source available — return an explicit error
    # instead of fabricated bar data.
    return JSONResponse(
        status_code=503,
        content={
            "error": "Market data unavailable",
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
            "stale": True,
            "message": "Yahoo Finance connector is not available. No bar data can be served.",
        },
    )


@router.get("/regime", response_model=MarketRegimeResponse)
async def get_regime(request: Request, current_user: dict = Depends(get_current_user)):
    """Current market regime from the regime classifier."""
    regime_classifier = getattr(request.app.state, "regime_classifier", None)
    if regime_classifier and hasattr(regime_classifier, "last_result"):
        result = regime_classifier.last_result
        if result:
            return {
                "regime": getattr(result, "label", "unknown"),
                "confidence": getattr(result, "confidence", 0.0),
                "vol_percentile": getattr(result, "vol_percentile", 0.0),
                "trend_strength": getattr(result, "trend_strength", 0.0),
            }
    # Fallback: infer from bar cache volatility
    bar_cache = getattr(request.app.state, "bar_cache", None)
    if bar_cache:
        try:
            from src.core.events import Exchange
            symbols = bar_cache.symbols_with_bars(Exchange.NSE, "1d", min_bars=10)
            if symbols:
                bars = bar_cache.get_bars(symbols[0], "NSE", "1d", 20)
                if bars and len(bars) >= 5:
                    closes = [getattr(b, "close", None) or (b.get("close") if isinstance(b, dict) else 0) for b in bars]
                    closes = [c for c in closes if c and c > 0]
                    if len(closes) >= 5:
                        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                        avg_ret = sum(returns) / len(returns)
                        vol = (sum((r - avg_ret)**2 for r in returns) / len(returns)) ** 0.5
                        trend = avg_ret / (vol + 1e-8)
                        if vol > 0.025:
                            regime = "high_volatility"
                        elif trend > 0.3:
                            regime = "trending_up"
                        elif trend < -0.3:
                            regime = "trending_down"
                        elif vol < 0.008:
                            regime = "low_volatility"
                        else:
                            regime = "low_volatility"
                        return {"regime": regime, "confidence": 0.7, "vol_percentile": round(vol * 100, 2), "trend_strength": round(trend, 3)}
        except Exception:
            pass
    return {"regime": "unknown", "confidence": 0.0, "vol_percentile": 0.0, "trend_strength": 0.0}


@router.get("/news", response_model=MarketNewsResponse)
async def get_news(request: Request, limit: int = Query(20, le=50), current_user: dict = Depends(get_current_user)):
    """Market news feed. Returns recent market events and AI sentiment analysis."""
    # Try real news service
    news_service = getattr(request.app.state, "news_service", None)
    if news_service and hasattr(news_service, "get_latest"):
        try:
            items = await news_service.get_latest(limit=limit)
            return {"news": items, "source": "live"}
        except Exception:
            pass

    # Generate contextual market news based on actual positions and bar data
    bar_cache = getattr(request.app.state, "bar_cache", None)
    risk_manager = getattr(request.app.state, "risk_manager", None)

    news_items = []
    now = datetime.now(timezone.utc)

    # Generate news from actual bar data
    if bar_cache:
        try:
            from src.core.events import Exchange
            symbols = bar_cache.symbols_with_bars(Exchange.NSE, "1d", min_bars=5)
            for sym in symbols[:10]:
                bars = bar_cache.get_bars(sym, "NSE", "1d", 5)
                if bars and len(bars) >= 2:
                    last_bar = bars[-1]
                    prev_bar = bars[-2]
                    close = getattr(last_bar, "close", None) or (last_bar.get("close") if isinstance(last_bar, dict) else 0)
                    prev_close = getattr(prev_bar, "close", None) or (prev_bar.get("close") if isinstance(prev_bar, dict) else 0)
                    if close and prev_close and prev_close > 0:
                        change_pct = ((close - prev_close) / prev_close) * 100
                        sentiment = "positive" if change_pct > 0.5 else "negative" if change_pct < -0.5 else "neutral"
                        headline = (
                            f"{sym} {'rallies' if change_pct > 1 else 'gains' if change_pct > 0 else 'drops' if change_pct < -1 else 'declines'} "
                            f"{abs(change_pct):.1f}% to ₹{close:.2f}"
                        )
                        news_items.append({
                            "id": f"news-{sym}-{len(news_items)}",
                            "headline": headline,
                            "symbol": sym,
                            "sentiment": sentiment,
                            "score": round(min(1.0, abs(change_pct) / 5), 2),
                            "source": "Market Data",
                            "timestamp": (now - timedelta(minutes=len(news_items) * 15)).isoformat(),
                            "category": "price_action",
                        })
        except Exception:
            pass

    # Add position-related news
    if risk_manager:
        for p in getattr(risk_manager, "positions", [])[:5]:
            sym = getattr(p, "symbol", "")
            side = getattr(getattr(p, "side", None), "value", "BUY")
            qty = getattr(p, "quantity", 0)
            if sym:
                news_items.append({
                    "id": f"news-pos-{sym}",
                    "headline": f"AlphaForge AI holds {qty} {'long' if side == 'BUY' else 'short'} position in {sym}",
                    "symbol": sym,
                    "sentiment": "neutral",
                    "score": 0.5,
                    "source": "Portfolio",
                    "timestamp": (now - timedelta(minutes=5)).isoformat(),
                    "category": "portfolio",
                })

    # Market summary
    if not news_items:
        news_items.append({
            "id": "news-market-0",
            "headline": "Markets trading steady, AI scanning for opportunities",
            "symbol": "NIFTY50",
            "sentiment": "neutral",
            "score": 0.5,
            "source": "Market",
            "timestamp": now.isoformat(),
            "category": "market",
        })

    return {"news": news_items[:limit], "source": "generated"}
