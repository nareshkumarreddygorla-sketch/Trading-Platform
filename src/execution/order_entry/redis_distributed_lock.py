"""
Redis distributed lock for order submission. Ensures cluster-wide single-writer for critical section.
Lock expiry prevents deadlock on pod crash.
Falls back to asyncio.Lock for single-process paper trading when Redis is unavailable.
"""

import asyncio
import logging
import uuid

logger = logging.getLogger(__name__)

DEFAULT_LOCK_KEY = "trading:order_submit_lock"
DEFAULT_TTL_SECONDS = 30
WATCHDOG_INTERVAL_SECONDS = 10  # Re-extend TTL every 10s while lock is held


class RedisDistributedLock:
    """
    Acquire a Redis lock (SET NX EX). Release with DELETE.
    Use as async context manager; supports lock timeout for acquire.

    Safety features:
    - TTL on every lock (default 30s) prevents cluster-wide deadlock on pod crash.
    - Watchdog task auto-extends TTL while the holder is alive.
    - force_release() for operator intervention when needed.

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
        self._token: str | None = None
        # In-memory fallback lock for paper trading
        self._local_lock = asyncio.Lock()
        self._using_local = False
        # Watchdog task that auto-extends TTL
        self._watchdog_task: asyncio.Task | None = None

    async def _get_redis(self):
        if self._redis is None and not self._redis_checked:
            self._redis_checked = True
            try:
                import redis.asyncio as redis

                client = redis.from_url(self.redis_url, decode_responses=True)
                await client.ping()
                self._redis = client
            except ImportError:
                logger.warning(
                    "DistributedLock DEGRADED: redis package not installed; using local asyncio.Lock (paper mode only)"
                )
            except Exception:
                logger.warning(
                    "DistributedLock DEGRADED: Redis unavailable; using local asyncio.Lock (paper mode only)"
                )
        return self._redis

    async def _watchdog(self) -> None:
        """Periodically extend the lock TTL while the holder is still alive.
        Runs as a background task; cancelled on release."""
        client = self._redis
        token = self._token
        if not client or not token:
            return
        # Lua script: only extend if we still own the lock
        extend_script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('expire', KEYS[1], ARGV[2])
        end
        return 0
        """
        try:
            while True:
                await asyncio.sleep(WATCHDOG_INTERVAL_SECONDS)
                try:
                    result = await client.eval(extend_script, 1, self.lock_key, token, str(self.ttl_seconds))
                    if not result:
                        logger.warning(
                            "Lock watchdog: lock %s lost (token mismatch); stopping watchdog",
                            self.lock_key,
                        )
                        return
                    logger.debug("Lock watchdog: extended TTL for %s", self.lock_key)
                except Exception as e:
                    logger.warning("Lock watchdog: failed to extend TTL: %s", e)
        except asyncio.CancelledError:
            return

    async def acquire(self) -> bool:
        """Acquire lock. Returns True if acquired, False on timeout."""
        client = await self._get_redis()
        if not client:
            # Fallback to local asyncio.Lock
            self._using_local = True
            try:
                await asyncio.wait_for(self._local_lock.acquire(), timeout=self.acquire_timeout_seconds)
                return True
            except TimeoutError:
                logger.error(
                    "Failed to acquire local fallback lock (timeout=%.1fs)",
                    self.acquire_timeout_seconds,
                )
                return False
        self._using_local = False
        self._token = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.acquire_timeout_seconds
        while loop.time() < deadline:
            try:
                ok = await client.set(self.lock_key, self._token, nx=True, ex=self.ttl_seconds)
                if ok:
                    # Start watchdog to auto-extend TTL
                    self._watchdog_task = asyncio.create_task(self._watchdog(), name=f"lock-watchdog-{self.lock_key}")
                    return True
            except Exception as e:
                logger.warning("Redis lock acquire failed: %s", e)
                return False
            await asyncio.sleep(0.2)
        # Timed out waiting to acquire
        logger.error(
            "Failed to acquire distributed lock key=%s after %.1fs timeout; "
            "another holder may be stuck or TTL too long",
            self.lock_key,
            self.acquire_timeout_seconds,
        )
        self._token = None
        return False

    async def release(self) -> None:
        """Release lock. Cancels watchdog and deletes key if we still own it."""
        # Stop watchdog first
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None

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
            # Lua: only delete if we still own the lock (compare token)
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

    async def force_release(self) -> bool:
        """Unconditionally delete the lock key — for operator intervention only.
        Returns True if a key was deleted, False otherwise.
        WARNING: This does NOT check ownership. Use only when you are certain
        the current holder is dead/stuck."""
        client = await self._get_redis()
        if not client:
            # For local lock, force-release by creating a new lock instance
            if self._local_lock.locked():
                try:
                    self._local_lock.release()
                except RuntimeError:
                    pass
                logger.warning("force_release: released local fallback lock")
                return True
            return False
        try:
            result = await client.delete(self.lock_key)
            if result:
                logger.warning(
                    "force_release: deleted lock key %s — operator intervention",
                    self.lock_key,
                )
                return True
            else:
                logger.info("force_release: lock key %s did not exist", self.lock_key)
                return False
        except Exception as e:
            logger.error("force_release failed for key %s: %s", self.lock_key, e)
            return False

    async def __aenter__(self) -> "RedisDistributedLock":
        ok = await self.acquire()
        if not ok:
            raise RuntimeError(
                f"Failed to acquire distributed lock key={self.lock_key} (timeout={self.acquire_timeout_seconds}s)"
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()
        return None
