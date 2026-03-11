"""
Market context: fetch NIFTY50 index data + compute market-wide regime features.
Provides additional context to the AI model so it doesn't buy in crashing markets.

Features produced:
  - nifty_trend: 1-bar return of NIFTY50 index
  - nifty_rsi: RSI of NIFTY50
  - nifty_volatility: 20-bar rolling volatility of NIFTY50
  - market_breadth: proxy — NIFTY daily return direction
"""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

# Cache context data to avoid repeated API calls within same scan cycle
_cache: dict[str, any] = {}
_cache_ts: float = 0.0
_CACHE_TTL = 300  # 5 minutes


def _rsi_calc(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-period - 1 :])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss < 1e-12:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / (avg_loss + 1e-12)
    return float(100.0 - (100.0 / (1.0 + rs)))


def _rolling_vol(closes: np.ndarray, window: int = 20) -> float:
    if len(closes) < window + 1:
        return 0.0
    ret = np.diff(closes[-window - 1 :]) / (closes[-window - 1 : -1] + 1e-12)
    return float(np.std(ret))


def fetch_market_context(interval: str = "5m", period: str = "5d") -> dict[str, float]:
    """
    Fetch NIFTY50 index data and compute market context features.
    Returns dict with market-level features. Cached for 5 minutes.
    """
    global _cache, _cache_ts

    now = time.time()
    if _cache and (now - _cache_ts) < _CACHE_TTL:
        return _cache

    try:
        import yfinance as yf

        nifty = yf.download("^NSEI", period=period, interval=interval, progress=False)
        if nifty is None or nifty.empty:
            logger.debug("Market context: NIFTY50 data empty, using defaults")
            return _default_context()

        nifty = nifty.rename(columns=str.lower)
        if "close" not in nifty.columns:
            return _default_context()

        closes = nifty["close"].dropna().values.astype(float)
        if len(closes) < 30:
            return _default_context()

        ctx = {
            "nifty_return_1": float((closes[-1] - closes[-2]) / (closes[-2] + 1e-12)) if len(closes) >= 2 else 0.0,
            "nifty_return_5": float((closes[-1] - closes[-6]) / (closes[-6] + 1e-12)) if len(closes) >= 6 else 0.0,
            "nifty_rsi": _rsi_calc(closes, 14),
            "nifty_volatility": _rolling_vol(closes, 20),
            "nifty_trend": 1.0
            if closes[-1] > closes[-5]
            else (-1.0 if closes[-1] < closes[-5] else 0.0)
            if len(closes) >= 6
            else 0.0,
        }

        _cache = ctx
        _cache_ts = now
        logger.debug(
            "Market context refreshed: NIFTY RSI=%.1f vol=%.4f trend=%.0f",
            ctx["nifty_rsi"],
            ctx["nifty_volatility"],
            ctx["nifty_trend"],
        )
        return ctx

    except Exception as e:
        logger.debug("Market context fetch failed: %s", e)
        return _default_context()


def _default_context() -> dict[str, float]:
    """Neutral defaults when market data unavailable."""
    return {
        "nifty_return_1": 0.0,
        "nifty_return_5": 0.0,
        "nifty_rsi": 50.0,
        "nifty_volatility": 0.01,
        "nifty_trend": 0.0,
    }
