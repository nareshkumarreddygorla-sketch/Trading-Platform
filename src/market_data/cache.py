"""Redis cache for latest quote and order book (L1)."""
import json
import logging
from typing import Optional

from src.core.events import Bar, Tick

logger = logging.getLogger(__name__)


class QuoteCache:
    """Real-time quote cache. Keys: quote:{exchange}:{symbol}, bar:{exchange}:{symbol}:{interval}."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._client = None

    async def get_client(self):
        if self._client is None:
            try:
                import redis.asyncio as redis
                self._client = redis.from_url(self.redis_url, decode_responses=True)
            except ImportError:
                logger.warning("redis not installed; QuoteCache no-op")
        return self._client

    def _quote_key(self, exchange: str, symbol: str) -> str:
        return f"quote:{exchange}:{symbol}"

    def _bar_key(self, exchange: str, symbol: str, interval: str) -> str:
        return f"bar:{exchange}:{symbol}:{interval}"

    async def set_tick(self, tick: Tick, ttl_seconds: int = 60) -> None:
        client = await self.get_client()
        if not client:
            return
        key = self._quote_key(tick.exchange.value, tick.symbol)
        payload = {
            "symbol": tick.symbol,
            "price": tick.price,
            "size": tick.size,
            "ts": tick.ts.isoformat(),
        }
        await client.setex(key, ttl_seconds, json.dumps(payload))

    async def get_quote(self, exchange: str, symbol: str) -> Optional[dict]:
        client = await self.get_client()
        if not client:
            return None
        key = self._quote_key(exchange, symbol)
        raw = await client.get(key)
        if raw:
            return json.loads(raw)
        return None

    async def set_bar(self, bar: Bar, ttl_seconds: int = 3600) -> None:
        client = await self.get_client()
        if not client:
            return
        key = self._bar_key(bar.exchange.value, bar.symbol, bar.interval)
        payload = {
            "symbol": bar.symbol,
            "open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close,
            "volume": bar.volume, "ts": bar.ts.isoformat(),
        }
        await client.setex(key, ttl_seconds, json.dumps(payload))

    async def get_bar(self, exchange: str, symbol: str, interval: str) -> Optional[dict]:
        client = await self.get_client()
        if not client:
            return None
        key = self._bar_key(exchange, symbol, interval)
        raw = await client.get(key)
        if raw:
            return json.loads(raw)
        return None
