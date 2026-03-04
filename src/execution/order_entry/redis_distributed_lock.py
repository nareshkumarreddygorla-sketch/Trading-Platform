"""
Redis distributed lock for order submission. Ensures cluster-wide single-writer for critical section.
Lock expiry prevents deadlock on pod crash.
Falls back to asyncio.Lock for single-process paper trading when Redis is unavailable.
"""
import asyncio
import logging
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_LOCK_KEY = "trading:order_submit_lock"
DEFAULT_TTL_SECONDS = 25


class RedisDistributedLock:
    """
    Acquire a Redis lock (SET NX EX). Release with DELETE.
    Use as async context manager; supports lock timeout for acquire.
    Falls back to asyncio.Lock when Redis is unavailable (paper trading).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        lock_key: str = DEFAULT_LOCK_KEY,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        acquire_timeout_seconds: float = 15.0,
    ):
        self.redis_url = redis_url
        self.lock_key = lock_key
        self.ttl_seconds = ttl_seconds
        self.acquire_timeout_seconds = acquire_timeout_seconds
        self._redis = None
        self._redis_checked = False
        self._token: Optional[str] = None
        # In-memory fallback lock for paper trading
        self._local_lock = asyncio.Lock()
        self._using_local = False

    async def _get_redis(self):
        if self._redis is None and not self._redis_checked:
            self._redis_checked = True
            try:
                import redis.asyncio as redis
                client = redis.from_url(self.redis_url, decode_responses=True)
                await client.ping()
                self._redis = client
            except ImportError:
                logger.info("redis not installed; using local lock (paper mode)")
            except Exception:
                logger.info("Redis unavailable; using local lock (paper mode)")
        return self._redis

    async def acquire(self) -> bool:
        """Acquire lock. Returns True if acquired, False on timeout."""
        client = await self._get_redis()
        if not client:
            # Fallback to local asyncio.Lock
            self._using_local = True
            try:
                await asyncio.wait_for(self._local_lock.acquire(), timeout=self.acquire_timeout_seconds)
                return True
            except asyncio.TimeoutError:
                return False
        self._using_local = False
        self._token = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.acquire_timeout_seconds
        while loop.time() < deadline:
            try:
                ok = await client.set(self.lock_key, self._token, nx=True, ex=self.ttl_seconds)
                if ok:
                    return True
            except Exception as e:
                logger.warning("Redis lock acquire failed: %s", e)
                return False
            await asyncio.sleep(0.2)
        return False

    async def release(self) -> None:
        """Release lock."""
        if self._using_local:
            try:
                self._local_lock.release()
            except RuntimeError:
                pass
            self._using_local = False
            return
        if not self._token:
            return
        client = await self._get_redis()
        if not client:
            return
        try:
            script = """
            if redis.call('get', KEYS[1]) == ARGV[1] then
                return redis.call('del', KEYS[1])
            end
            return 0
            """
            await client.eval(script, 1, self.lock_key, self._token)
        except Exception as e:
            logger.warning("Redis lock release failed: %s", e)
        self._token = None

    async def __aenter__(self) -> "RedisDistributedLock":
        ok = await self.acquire()
        if not ok:
            raise RuntimeError("Failed to acquire distributed submission lock (timeout)")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()
        return None
