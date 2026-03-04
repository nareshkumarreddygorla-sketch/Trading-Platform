"""
Phase 13: Order flood protection — max N orders per minute; reject beyond that.
"""
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class RateLimitConfig:
    max_orders_per_minute: int = 60
    window_seconds: float = 60.0


class OrderRateLimiter:
    """
    Sliding window: count orders in last window_seconds; reject if count >= max_orders_per_minute.
    """

    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self._timestamps: Deque[float] = deque()

    def allow(self) -> bool:
        """Return True if order allowed; else False (flood)."""
        now = time.monotonic()
        window = self.config.window_seconds
        while self._timestamps and self._timestamps[0] < now - window:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.config.max_orders_per_minute:
            return False
        self._timestamps.append(now)
        return True

    def reset(self) -> None:
        self._timestamps.clear()
