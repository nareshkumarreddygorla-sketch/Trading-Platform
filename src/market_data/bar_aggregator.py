"""
Tick-to-bar aggregator: aggregate ticks into OHLCV bars (1m minimum).
Feeds BarCache. Market data layer pushes ticks here; bars are emitted to cache.
No trading allowed if market data unavailable (bar cache empty).
"""
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from src.core.events import Bar, Exchange, Tick

logger = logging.getLogger(__name__)


class TickToBarAggregator:
    """
    Aggregate ticks into fixed-interval bars (e.g. 1m).
    Flush complete bars to the provided bar_cache.
    """

    def __init__(self, bar_cache, interval_seconds: int = 60):
        self.bar_cache = bar_cache
        self.interval_seconds = interval_seconds
        self._current: dict = defaultdict(lambda: {"open": None, "high": None, "low": None, "close": None, "volume": 0.0, "bucket_ts": None})

    def _bucket_ts(self, ts: datetime) -> int:
        t = ts.timestamp() if hasattr(ts, "timestamp") else float(ts)
        return int(t // self.interval_seconds) * self.interval_seconds

    def push_tick(self, tick: Tick) -> Optional[Bar]:
        """
        Push a tick. If it completes a new bar, return that Bar (and append to cache).
        """
        key = (tick.symbol, tick.exchange.value if hasattr(tick.exchange, "value") else str(tick.exchange))
        bucket = self._bucket_ts(tick.ts)
        cur = self._current[key]
        if cur["bucket_ts"] is None:
            cur["open"] = cur["high"] = cur["low"] = cur["close"] = tick.price
            cur["volume"] = tick.size
            cur["bucket_ts"] = bucket
            return None
        if bucket > cur["bucket_ts"]:
            bar = Bar(
                symbol=tick.symbol,
                exchange=tick.exchange if isinstance(tick.exchange, Exchange) else Exchange(tick.exchange) if isinstance(tick.exchange, str) else Exchange.NSE,
                interval="1m",
                open=cur["open"],
                high=cur["high"],
                low=cur["low"],
                close=cur["close"],
                volume=cur["volume"],
                ts=datetime.fromtimestamp(cur["bucket_ts"], tz=timezone.utc),
                source="aggregator",
            )
            self.bar_cache.append_bar(bar)
            cur["open"] = cur["high"] = cur["low"] = cur["close"] = tick.price
            cur["volume"] = tick.size
            cur["bucket_ts"] = bucket
            return bar
        cur["high"] = max(cur["high"], tick.price)
        cur["low"] = min(cur["low"], tick.price)
        cur["close"] = tick.price
        cur["volume"] += tick.size
        return None
