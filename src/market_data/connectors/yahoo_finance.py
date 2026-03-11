"""
Yahoo Finance connector: provides real market data (quotes + OHLCV bars)
for NSE/BSE stocks when Angel One is not configured.

Uses yfinance library. Falls back gracefully if yfinance is not installed.
In-memory cache with configurable TTL to avoid rate limits.
"""

import logging
import time
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Try to import yfinance; set flag if unavailable
try:
    import yfinance as yf

    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    logger.info("yfinance not installed — Yahoo Finance connector disabled. Install with: pip install yfinance")


class YahooFinanceConnector:
    """
    Fetch real market data from Yahoo Finance.
    - Auto-appends .NS suffix for NSE stocks, .BO for BSE.
    - Caches quotes for `cache_ttl_seconds` to avoid rate limiting.
    """

    MAX_CACHE_SIZE = 200  # Maximum entries per cache dict

    def __init__(self, cache_ttl_seconds: int = 60):
        self.cache_ttl = cache_ttl_seconds
        self._quote_cache: dict[str, tuple[float, dict[str, Any]]] = {}  # key -> (timestamp, data)
        self._bars_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}

    def _evict_stale(self, cache: dict[str, tuple[float, Any]]) -> None:
        """Remove stale entries and cap cache size."""
        now = time.time()
        # Remove expired entries
        expired = [k for k, (ts, _) in cache.items() if (now - ts) >= self.cache_ttl]
        for k in expired:
            del cache[k]
        # If still over limit, drop oldest entries
        if len(cache) > self.MAX_CACHE_SIZE:
            sorted_keys = sorted(cache, key=lambda k: cache[k][0])
            for k in sorted_keys[: len(cache) - self.MAX_CACHE_SIZE]:
                del cache[k]

    @staticmethod
    def _yf_symbol(symbol: str, exchange: str = "NSE") -> str:
        """Convert internal symbol to Yahoo Finance ticker."""
        symbol = symbol.upper().strip()
        if "." in symbol:  # Already has suffix
            return symbol
        suffix = ".NS" if exchange.upper() == "NSE" else ".BO" if exchange.upper() == "BSE" else ".NS"
        return f"{symbol}{suffix}"

    def get_quote(self, symbol: str, exchange: str = "NSE") -> dict[str, Any] | None:
        """Get latest quote for a symbol. Returns None on failure."""
        if not YF_AVAILABLE:
            return None

        cache_key = f"quote:{exchange}:{symbol}"
        cached = self._quote_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < self.cache_ttl:
            return cached[1]

        try:
            yf_sym = self._yf_symbol(symbol, exchange)
            ticker = yf.Ticker(yf_sym)
            info = ticker.fast_info
            price = float(getattr(info, "last_price", 0) or 0)
            if price <= 0:
                # Try history fallback
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])

            if price <= 0:
                return None

            prev_close = float(getattr(info, "previous_close", price) or price)
            day_high = float(getattr(info, "day_high", price) or price)
            day_low = float(getattr(info, "day_low", price) or price)
            volume = int(getattr(info, "last_volume", 0) or 0)

            # Yahoo Finance does not provide real-time bid/ask data.
            # Set bid and ask to the last traded price rather than fabricating a spread.
            quote = {
                "symbol": symbol,
                "exchange": exchange,
                "last": round(price, 2),
                "bid": round(price, 2),
                "ask": round(price, 2),
                "prev_close": round(prev_close, 2),
                "day_high": round(day_high, 2),
                "day_low": round(day_low, 2),
                "change": round(price - prev_close, 2),
                "change_pct": round(((price - prev_close) / prev_close * 100) if prev_close > 0 else 0, 2),
                "volume": volume,
                "ts": datetime.now(UTC).isoformat(),
                "source": "yahoo_finance",
            }
            self._quote_cache[cache_key] = (time.time(), quote)
            self._evict_stale(self._quote_cache)
            return quote
        except Exception as e:
            logger.debug("Yahoo Finance quote failed for %s: %s", symbol, e)
            return None

    def get_bars(
        self,
        symbol: str,
        exchange: str = "NSE",
        interval: str = "1d",
        limit: int = 100,
        from_ts: str | None = None,
        to_ts: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Get OHLCV bars. Returns None on failure."""
        if not YF_AVAILABLE:
            return None

        cache_key = f"bars:{exchange}:{symbol}:{interval}:{limit}"
        cached = self._bars_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < self.cache_ttl:
            return cached[1]

        # Map interval to yfinance format
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "1d": "1d",
            "1w": "1wk",
            "1M": "1mo",
        }
        yf_interval = interval_map.get(interval, "1d")

        # Determine period from limit
        if yf_interval in ("1m",):
            period = "7d"  # 1m data only available for 7 days
        elif yf_interval in ("5m", "15m", "30m"):
            period = "60d"
        elif yf_interval == "1h":
            period = "730d"
        else:
            period = "2y"

        try:
            yf_sym = self._yf_symbol(symbol, exchange)
            ticker = yf.Ticker(yf_sym)

            if from_ts and to_ts:
                hist = ticker.history(start=from_ts, end=to_ts, interval=yf_interval)
            else:
                hist = ticker.history(period=period, interval=yf_interval)

            if hist.empty:
                return None

            # Take last `limit` bars
            hist = hist.tail(limit)

            bars = []
            for idx, row in hist.iterrows():
                ts = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
                bars.append(
                    {
                        "open": round(float(row["Open"]), 2),
                        "high": round(float(row["High"]), 2),
                        "low": round(float(row["Low"]), 2),
                        "close": round(float(row["Close"]), 2),
                        "volume": int(row["Volume"]),
                        "ts": ts,
                    }
                )

            if bars:
                self._bars_cache[cache_key] = (time.time(), bars)
                self._evict_stale(self._bars_cache)
            return bars
        except Exception as e:
            logger.debug("Yahoo Finance bars failed for %s: %s", symbol, e)
            return None


# Module-level singleton
_connector: YahooFinanceConnector | None = None


def get_yahoo_connector() -> YahooFinanceConnector | None:
    """Get or create the Yahoo Finance connector singleton."""
    global _connector
    if not YF_AVAILABLE:
        return None
    if _connector is None:
        _connector = YahooFinanceConnector(cache_ttl_seconds=60)
    return _connector
