"""
Average Daily Volume (ADV) cache for market impact model and paper fill simulator.

Fetches ADV per symbol from yfinance with weekly refresh.
Thread-safe with in-memory cache (7-day TTL).
Provides `get_adv(symbol)` for market impact model and paper fills.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

# Default ADV when data unavailable (conservative: moderate liquidity)
DEFAULT_ADV = 500_000
CACHE_TTL_SECONDS = 7 * 86400  # 7 days
FETCH_TIMEOUT_SECONDS = 10


class ADVCache:
    """
    Cache for average daily volume per symbol.

    Usage:
        cache = ADVCache()
        adv = cache.get_adv("RELIANCE")  # Returns int volume
        cache.refresh(["RELIANCE", "INFY", "HDFCBANK"])  # Bulk refresh
    """

    def __init__(self, default_adv: int = DEFAULT_ADV, ttl_seconds: int = CACHE_TTL_SECONDS):
        self._cache: dict[str, int] = {}  # symbol -> ADV
        self._timestamps: dict[str, float] = {}  # symbol -> fetch time
        self._lock = threading.Lock()
        self._default_adv = default_adv
        self._ttl = ttl_seconds

    def get_adv(self, symbol: str, exchange: str = "NSE") -> int:
        """
        Get average daily volume for symbol. Returns cached value if fresh,
        otherwise attempts lazy fetch. Falls back to default on failure.
        """
        with self._lock:
            cached = self._cache.get(symbol)
            ts = self._timestamps.get(symbol, 0)
            if cached is not None and (time.time() - ts) < self._ttl:
                return cached

        # Try lazy fetch (non-blocking for callers if possible)
        adv = self._fetch_single(symbol, exchange)
        with self._lock:
            self._cache[symbol] = adv
            self._timestamps[symbol] = time.time()
        return adv

    def get_all(self) -> dict[str, int]:
        """Return all cached ADV values."""
        with self._lock:
            return dict(self._cache)

    def set_adv(self, symbol: str, adv: int) -> None:
        """Manually set ADV (e.g., from broker data)."""
        with self._lock:
            self._cache[symbol] = adv
            self._timestamps[symbol] = time.time()

    def refresh(self, symbols: list, exchange: str = "NSE") -> dict[str, int]:
        """
        Bulk refresh ADV for multiple symbols from yfinance.
        Returns dict of {symbol: adv}.
        """
        results = {}
        for sym in symbols:
            adv = self._fetch_single(sym, exchange)
            with self._lock:
                self._cache[sym] = adv
                self._timestamps[sym] = time.time()
            results[sym] = adv
        return results

    def _fetch_single(self, symbol: str, exchange: str = "NSE") -> int:
        """Fetch ADV for a single symbol from yfinance."""
        try:
            import yfinance as yf

            # Map NSE symbols to yfinance format
            yf_symbol = symbol
            if exchange.upper() in ("NSE", "BSE") and "." not in symbol:
                yf_symbol = f"{symbol}.NS"

            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period="30d", auto_adjust=True)
            if hist is not None and not hist.empty and "Volume" in hist.columns:
                volumes = hist["Volume"].dropna()
                if len(volumes) > 0:
                    adv = int(volumes.mean())
                    if adv > 0:
                        logger.debug("ADV fetched: %s = %d", symbol, adv)
                        return adv
        except ImportError:
            logger.debug("yfinance not installed — using default ADV for %s", symbol)
        except Exception as e:
            logger.debug("ADV fetch failed for %s: %s", symbol, e)

        return self._default_adv

    def start_background_refresh(self, symbols: list, exchange: str = "NSE", interval_hours: int = 24) -> None:
        """Start a background thread that refreshes ADV periodically."""

        def _refresh_loop():
            while True:
                try:
                    self.refresh(symbols, exchange)
                    logger.info("ADV cache refreshed: %d symbols", len(symbols))
                except Exception as e:
                    logger.warning("ADV background refresh failed: %s", e)
                time.sleep(interval_hours * 3600)

        t = threading.Thread(target=_refresh_loop, daemon=True, name="adv-cache-refresh")
        t.start()
        logger.info("ADV cache background refresh started (interval=%dh, symbols=%d)", interval_hours, len(symbols))
