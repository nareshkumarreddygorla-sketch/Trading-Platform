"""
Bar cache for autonomous loop: rolling bars per symbol, current bar timestamp.
In-memory implementation; can be backed by Redis for multi-pod consistency.
No trading allowed if market data unavailable (empty bars).
Thread-safe: all mutations and reads protected by threading.RLock (accessed from multiple threads).
"""

import logging
import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime

from src.core.events import Bar, Exchange
from src.data_pipeline.ohlc_validator import OHLCValidator

logger = logging.getLogger(__name__)

MAX_BARS_PER_SYMBOL = 500

# Symbols with no bar activity for this duration are pruned from the cache
_SYMBOL_IDLE_PRUNE_SECONDS = 24 * 3600  # 24 hours


class BarCache:
    """
    Rolling bar cache per (symbol, exchange, interval).
    Provides get_bars and current bar timestamp for idempotency.
    Thread-safe via threading.RLock (not asyncio -- accessed from multiple threads).

    Uses collections.deque(maxlen=N) for O(1) append with automatic eviction
    of the oldest bar when capacity is reached (no list.pop(0) O(n) penalty).
    """

    def __init__(
        self,
        max_bars: int = MAX_BARS_PER_SYMBOL,
        ohlc_validator: OHLCValidator | None = None,
    ):
        self._max_bars = max_bars
        self._bars: dict[str, deque[Bar]] = defaultdict(lambda: deque(maxlen=max_bars))
        # Track last bar arrival per key for idle-symbol pruning
        self._last_update: dict[str, float] = {}
        self._current_bar_ts: str | None = None
        self._lock = threading.RLock()
        # Use a generous stale threshold (7 days) to allow historical
        # backfill (e.g. yfinance initial 2-day load) while still catching
        # wildly stale data.  Live-feed freshness is enforced separately via
        # get_fresh_bars() and the feed health monitor.
        self._ohlc_validator = ohlc_validator or OHLCValidator(stale_seconds=7 * 86400.0)

    def _key(self, symbol: str, exchange: str, interval: str) -> str:
        return f"{exchange}:{symbol}:{interval}"

    def append_bar(self, bar: Bar) -> bool:
        """Validate and append a bar to the cache.

        Returns True if the bar was accepted, False if rejected by the
        OHLC validator or chronological ordering check.
        """
        with self._lock:
            k = self._key(
                bar.symbol, bar.exchange.value if hasattr(bar.exchange, "value") else str(bar.exchange), bar.interval
            )
            dq = self._bars[k]
            # Chronological validation: skip out-of-order bars
            if dq and hasattr(bar.ts, "timestamp") and hasattr(dq[-1].ts, "timestamp"):
                if bar.ts.timestamp() <= dq[-1].ts.timestamp():
                    return False  # Out-of-order bar, skip silently

            # OHLC data-integrity validation
            vr = self._ohlc_validator.validate_bar(
                symbol=bar.symbol,
                open_=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                timestamp=bar.ts,
                interval=bar.interval,
            )
            if not vr.is_valid:
                logger.debug(
                    "Bar rejected for %s: O=%.2f H=%.2f L=%.2f C=%.2f reasons=%s",
                    bar.symbol,
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    [r.value for r in vr.reject_reasons],
                )
                return False

            # deque(maxlen=N) automatically evicts the oldest bar on overflow
            dq.append(bar)
            ts = bar.ts.isoformat() if hasattr(bar.ts, "isoformat") else str(bar.ts)
            self._current_bar_ts = ts
            now = time.time()
            self._last_update[k] = now
            self._last_bar_time = now  # Track last bar arrival (any symbol)
            return True

    def get_bars(
        self,
        symbol: str,
        exchange: Exchange,
        interval: str = "1m",
        n: int = 100,
    ) -> list[Bar]:
        with self._lock:
            k = self._key(symbol, exchange.value if hasattr(exchange, "value") else str(exchange), interval)
            dq = self._bars.get(k)
            if dq is None:
                return []
            # Return list copy to prevent external mutation
            if n:
                # deque supports negative indexing; convert tail slice to list
                length = len(dq)
                start = max(0, length - n)
                return list(dq)[start:]
            return list(dq)

    def get_current_bar_ts(self) -> str | None:
        with self._lock:
            return self._current_bar_ts

    def get_fresh_bars(
        self,
        symbol: str,
        exchange: Exchange,
        interval: str = "1m",
        n: int = 100,
        max_age_seconds: float = 90.0,
    ) -> list[Bar] | None:
        """Return bars only if the most recent bar is within max_age_seconds.
        Returns None if data is stale — callers MUST NOT trade on None."""
        bars = self.get_bars(symbol, exchange, interval, n)
        if not bars:
            return None
        last_bar = bars[-1]
        bar_ts = getattr(last_bar, "ts", None)
        if bar_ts is None:
            return None  # No timestamp = untrusted data
        # Parse timestamp
        if isinstance(bar_ts, str):
            try:
                bar_ts = datetime.fromisoformat(bar_ts.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return None
        if hasattr(bar_ts, "tzinfo") and bar_ts.tzinfo is None:
            bar_ts = bar_ts.replace(tzinfo=UTC)
        try:
            age = (datetime.now(UTC) - bar_ts).total_seconds()
        except (TypeError, AttributeError):
            return None
        if age > max_age_seconds:
            logger.warning(
                "Stale data for %s: last bar %.0fs old (max=%.0fs) — blocking signal",
                symbol,
                age,
                max_age_seconds,
            )
            return None
        return bars

    def detect_gaps(
        self,
        symbol: str,
        exchange: Exchange,
        interval: str = "1m",
        n: int = 100,
    ) -> list[int]:
        """Return indices where bar gaps > 2.5x expected interval exist."""
        bars = self.get_bars(symbol, exchange, interval, n)
        if len(bars) < 2:
            return []
        interval_secs = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400}.get(interval, 60)
        gaps = []
        for i in range(1, len(bars)):
            ts_prev = getattr(bars[i - 1], "ts", None)
            ts_curr = getattr(bars[i], "ts", None)
            if ts_prev and ts_curr and hasattr(ts_prev, "timestamp") and hasattr(ts_curr, "timestamp"):
                delta = ts_curr.timestamp() - ts_prev.timestamp()
                if delta > interval_secs * 2.5:
                    gaps.append(i)
        return gaps

    def last_bar_timestamp(self) -> float | None:
        """Return timestamp (epoch seconds) of last bar arrival, or None."""
        return getattr(self, "_last_bar_time", None)

    def has_data(self, symbol: str, exchange: Exchange, interval: str = "1m", min_bars: int = 20) -> bool:
        return len(self.get_bars(symbol, exchange, interval, 0)) >= min_bars

    def symbols_with_bars(self, exchange: Exchange, interval: str = "1m", min_bars: int = 20) -> list[str]:
        with self._lock:
            out = []
            for k, dq in self._bars.items():
                if not dq:
                    continue
                parts = k.split(":", 2)
                if (
                    len(parts) >= 3
                    and parts[0] == (exchange.value if hasattr(exchange, "value") else str(exchange))
                    and parts[2] == interval
                ):
                    if len(dq) >= min_bars:
                        out.append(parts[1])
            return out

    def prune_idle_symbols(self, max_idle_seconds: float = _SYMBOL_IDLE_PRUNE_SECONDS) -> int:
        """Remove symbols that have not received a bar in *max_idle_seconds*.

        Returns the number of keys pruned.  Should be called periodically
        (e.g. once per hour or at market close) to prevent unbounded memory
        growth from symbols that are no longer in the active universe.
        """
        now = time.time()
        pruned = 0
        with self._lock:
            stale_keys = [k for k, ts in self._last_update.items() if (now - ts) > max_idle_seconds]
            for k in stale_keys:
                del self._bars[k]
                del self._last_update[k]
                pruned += 1
        if pruned:
            logger.info("BarCache: pruned %d idle symbol keys (idle > %ds)", pruned, int(max_idle_seconds))
        return pruned
