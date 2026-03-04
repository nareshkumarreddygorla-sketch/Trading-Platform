"""
Production-grade idempotency: Redis (or DB) with TTL.
Duplicate request returns same order_id; broker is NOT called again.
Falls back to in-memory store for paper trading when Redis is unavailable.
"""
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

IDEMPOTENCY_PREFIX = "idem:"
DEFAULT_TTL_SECONDS = 86400 * 2  # 2 days


class IdempotencyStore:
    """
    Store: idempotency_key -> { order_id, broker_order_id, status, ts }.
    Uses Redis when available; falls back to in-memory dict for paper trading.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self._redis = None
        self._redis_checked = False
        self._redis_available = False
        # In-memory fallback for paper trading (no Redis)
        self._mem_store: dict = {}

    async def _get_redis(self):
        if self._redis is None and not self._redis_checked:
            self._redis_checked = True
            try:
                import redis.asyncio as redis
                client = redis.from_url(self.redis_url, decode_responses=True)
                await client.ping()
                self._redis = client
                self._redis_available = True
                logger.info("IdempotencyStore: Redis connected")
            except ImportError:
                logger.info("redis not installed; using in-memory idempotency (paper mode)")
            except Exception:
                logger.info("Redis unavailable; using in-memory idempotency (paper mode)")
        return self._redis

    async def is_available(self) -> bool:
        """Always returns True — uses Redis when available, in-memory otherwise."""
        await self._get_redis()
        return True

    @property
    def redis_connected(self) -> bool:
        """True only if Redis is the active backend (multi-pod safe dedup).
        When False, idempotency falls back to in-memory which is volatile and per-process."""
        return self._redis_available

    def _key(self, idempotency_key: str) -> str:
        return f"{IDEMPOTENCY_PREFIX}{idempotency_key}"

    def _mem_cleanup(self) -> None:
        """Evict expired keys from in-memory store."""
        now = time.time()
        expired = [k for k, (exp, _) in self._mem_store.items() if now >= exp]
        for k in expired:
            del self._mem_store[k]

    async def get(self, idempotency_key: str) -> Optional[dict]:
        """Return stored result if any. None if missing or expired."""
        client = await self._get_redis()
        if client:
            try:
                raw = await client.get(self._key(idempotency_key))
                if raw:
                    return json.loads(raw)
            except Exception as e:
                logger.exception("IdempotencyStore.get failed: %s", e)
            return None
        # In-memory fallback
        key = self._key(idempotency_key)
        entry = self._mem_store.get(key)
        if entry:
            expiry, data = entry
            if time.time() < expiry:
                return data
            del self._mem_store[key]
        return None

    async def set(self, idempotency_key: str, order_id: str, broker_order_id: Optional[str] = None, status: str = "PENDING") -> bool:
        """Store result. Returns False if key already exists (NX semantics)."""
        payload = {"order_id": order_id, "broker_order_id": broker_order_id or "", "status": status, "ts": datetime.now(timezone.utc).isoformat()}
        client = await self._get_redis()
        if client:
            key = self._key(idempotency_key)
            try:
                ok = await client.set(key, json.dumps(payload), nx=True, ex=self.ttl_seconds)
                return bool(ok)
            except Exception as e:
                logger.exception("IdempotencyStore.set failed: %s", e)
                return False
        # In-memory fallback (NX: only set if not exists)
        key = self._key(idempotency_key)
        existing = self._mem_store.get(key)
        if existing and time.time() < existing[0]:
            return False  # key exists, not expired
        self._mem_store[key] = (time.time() + self.ttl_seconds, payload)
        self._mem_cleanup()
        return True

    async def update(self, idempotency_key: str, order_id: str, broker_order_id: Optional[str] = None, status: str = "PENDING") -> None:
        """Overwrite existing key."""
        payload = {"order_id": order_id, "broker_order_id": broker_order_id or "", "status": status, "ts": datetime.now(timezone.utc).isoformat()}
        client = await self._get_redis()
        if client:
            key = self._key(idempotency_key)
            try:
                await client.set(key, json.dumps(payload), ex=self.ttl_seconds)
            except Exception as e:
                logger.exception("IdempotencyStore.update failed: %s", e)
            return
        # In-memory fallback
        key = self._key(idempotency_key)
        self._mem_store[key] = (time.time() + self.ttl_seconds, payload)

    async def set_if_new_or_get(self, idempotency_key: str, order_id: str, broker_order_id: Optional[str], status: str) -> tuple[bool, Optional[dict]]:
        """
        If key does not exist: set and return (True, None).
        If key exists: return (False, stored_value) so caller can return stored order_id.
        """
        existing = await self.get(idempotency_key)
        if existing:
            return False, existing
        ok = await self.set(idempotency_key, order_id, broker_order_id, status)
        return ok, None

    @staticmethod
    def derive_key(signal_id: str, symbol: str, side: str, quantity: int, price: Optional[float], ts_iso: str) -> str:
        """Derive idempotency key from request if client did not send one."""
        raw = f"{signal_id}|{symbol}|{side}|{quantity}|{price}|{ts_iso}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    @staticmethod
    def derive_key_bar_stable(bar_ts_iso: str, strategy_id: str, symbol: str, side: str) -> str:
        """Stable idempotency key for autonomous loop: bar_ts + strategy + symbol + side. Same bar → same key → no duplicate order."""
        raw = f"{bar_ts_iso}|{strategy_id}|{symbol}|{side}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]
