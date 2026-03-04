"""
Market data API: quote and bars. Uses Yahoo Finance when available,
falls back to in-memory stubs.
"""
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, Request

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory stub cache: (exchange, symbol) -> quote; (exchange, symbol, interval) -> bars
_quote_cache: Dict[str, Dict[str, Any]] = {}
_bars_cache: Dict[str, List[Dict[str, Any]]] = {}

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


def _seed_from_symbol(symbol: str, exchange: str) -> int:
    h = hashlib.sha256(f"{exchange}:{symbol}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


def _stub_quote(symbol: str, exchange: str) -> Dict[str, Any]:
    import numpy as np
    seed = _seed_from_symbol(symbol, exchange)
    rng = np.random.default_rng(seed)
    base = 100.0 + (seed % 500)
    price = base * (1 + rng.standard_normal() * 0.01)
    return {
        "symbol": symbol,
        "exchange": exchange,
        "last": round(price, 2),
        "bid": round(price - 0.05, 2),
        "ask": round(price + 0.05, 2),
        "volume": int(100000 + rng.integers(0, 50000)),
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": "stub",
    }


def _stub_bars(symbol: str, exchange: str, interval: str, from_ts: Optional[str], to_ts: Optional[str], limit: int = 500) -> List[Dict[str, Any]]:
    import numpy as np
    seed = _seed_from_symbol(symbol, exchange)
    rng = np.random.default_rng(seed)
    n = min(limit, 500)
    delta = timedelta(hours=1) if interval in ("1m", "1h") else timedelta(days=1)
    try:
        if to_ts:
            end = datetime.fromisoformat(to_ts.replace("Z", "+00:00"))
        else:
            end = datetime.now(timezone.utc)
        if from_ts:
            start = datetime.fromisoformat(from_ts.replace("Z", "+00:00"))
        else:
            start = end - n * delta
    except Exception:
        end = datetime.now(timezone.utc)
        start = end - n * delta
    price = 100.0 + (seed % 500)
    bars = []
    current = start
    for _ in range(n):
        ret = rng.standard_normal() * 0.015
        price = price * (1 + ret)
        o, c = price, price
        h = max(o, c) + abs(rng.standard_normal() * 0.5)
        l = min(o, c) - abs(rng.standard_normal() * 0.5)
        vol = max(1000, int(rng.exponential(30000)))
        bars.append({
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
            "volume": vol,
            "ts": current.isoformat(),
        })
        current += delta
        if current > end:
            break
    return bars


@router.get("/quote/{symbol}")
async def get_quote(symbol: str, exchange: str = "NSE"):
    """Latest quote. Tries Yahoo Finance first, falls back to stub."""
    yf = _get_yf()
    if yf:
        quote = yf.get_quote(symbol, exchange)
        if quote:
            return quote

    key = f"{exchange}:{symbol}"
    if key not in _quote_cache:
        _quote_cache[key] = _stub_quote(symbol, exchange)
    return _quote_cache[key]


@router.get("/status")
async def market_status(request: Request):
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
):
    """OHLCV bars. Tries Yahoo Finance first, falls back to stub."""
    exchange = "NSE"
    yf = _get_yf()
    if yf:
        bars = yf.get_bars(symbol, exchange, interval, limit, from_ts, to_ts)
        if bars:
            return {"symbol": symbol, "exchange": exchange, "interval": interval, "bars": bars, "source": "yahoo_finance"}

    bars = _stub_bars(symbol, exchange, interval, from_ts, to_ts, limit)
    return {"symbol": symbol, "exchange": exchange, "interval": interval, "bars": bars, "source": "stub"}


@router.get("/regime")
async def get_regime(request: Request):
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


@router.get("/news")
async def get_news(request: Request, limit: int = Query(20, le=50)):
    """Market news feed. Returns recent market events and AI sentiment analysis."""
    import random
    from datetime import datetime, timezone

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
