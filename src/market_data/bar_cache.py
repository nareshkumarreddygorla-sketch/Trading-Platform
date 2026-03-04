"""
Bar cache for autonomous loop: rolling bars per symbol, current bar timestamp.
In-memory implementation; can be backed by Redis for multi-pod consistency.
No trading allowed if market data unavailable (empty bars).
Thread-safe: all mutations and reads protected by threading.Lock (accessed from multiple threads).
"""
import logging
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.core.events import Bar, Exchange

logger = logging.getLogger(__name__)

MAX_BARS_PER_SYMBOL = 500


class BarCache:
    """
    Rolling bar cache per (symbol, exchange, interval).
    Provides get_bars and current bar timestamp for idempotency.
    Thread-safe via threading.Lock (not asyncio — accessed from multiple threads).
    """

    def __init__(self, max_bars: int = MAX_BARS_PER_SYMBOL):
        self._max_bars = max_bars
        self._bars: Dict[str, List[Bar]] = defaultdict(list)
        self._current_bar_ts: Optional[str] = None
        self._lock = threading.Lock()

    def _key(self, symbol: str, exchange: str, interval: str) -> str:
        return f"{exchange}:{symbol}:{interval}"

    def append_bar(self, bar: Bar) -> None:
        with self._lock:
            k = self._key(bar.symbol, bar.exchange.value if hasattr(bar.exchange, "value") else str(bar.exchange), bar.interval)
            lst = self._bars[k]
            # Chronological validation (Sprint 10.7): skip out-of-order bars
            if lst and hasattr(bar.ts, 'timestamp') and hasattr(lst[-1].ts, 'timestamp'):
                if bar.ts.timestamp() <= lst[-1].ts.timestamp():
                    return  # Out-of-order bar, skip silently
            lst.append(bar)
            if len(lst) > self._max_bars:
                lst.pop(0)
            ts = bar.ts.isoformat() if hasattr(bar.ts, "isoformat") else str(bar.ts)
            self._current_bar_ts = ts
            self._last_bar_time = __import__("time").time()  # Track last bar arrival

    def get_bars(
        self,
        symbol: str,
        exchange: Exchange,
        interval: str = "1m",
        n: int = 100,
    ) -> List[Bar]:
        with self._lock:
            k = self._key(symbol, exchange.value if hasattr(exchange, "value") else str(exchange), interval)
            lst = self._bars.get(k, [])
            # Return copies to prevent external mutation
            return list(lst[-n:]) if n else list(lst)

    def get_current_bar_ts(self) -> Optional[str]:
        with self._lock:
            return self._current_bar_ts

    def last_bar_timestamp(self) -> Optional[float]:
        """Return timestamp (epoch seconds) of last bar arrival, or None."""
        return getattr(self, "_last_bar_time", None)

    def has_data(self, symbol: str, exchange: Exchange, interval: str = "1m", min_bars: int = 20) -> bool:
        return len(self.get_bars(symbol, exchange, interval, 0)) >= min_bars

    def symbols_with_bars(self, exchange: Exchange, interval: str = "1m", min_bars: int = 20) -> List[str]:
        with self._lock:
            out = []
            for k, lst in self._bars.items():
                if not lst:
                    continue
                parts = k.split(":", 2)
                if len(parts) >= 3 and parts[0] == (exchange.value if hasattr(exchange, "value") else str(exchange)) and parts[2] == interval:
                    if len(lst) >= min_bars:
                        out.append(parts[1])
            return out
