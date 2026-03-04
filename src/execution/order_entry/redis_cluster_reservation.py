"""
Cluster-wide reservation count in Redis. Ensures max_open_positions not exceeded across pods.
Reserve: only if (current_count + 1) <= max_allowed; then INCR and SADD order_id.
Release: SREM order_id, DECR (only if member was present).
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

RESERVATION_COUNT_KEY = "trading:cluster:reservation_count"
RESERVATION_IDS_KEY = "trading:cluster:reservation_ids"


class RedisClusterReservation:
    """
    Shared reservation state in Redis. Call reserve() before local reserve; release() on broker fail/timeout.

    Parameters
    ----------
    fail_open : bool
        If True (default), allow reservation when Redis is unavailable (safe for single-pod).
        Set to False in multi-pod/cluster deployments to prevent position limit bypass.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", fail_open: bool = True):
        self.redis_url = redis_url
        self._redis = None
        self._fail_open = fail_open

    async def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(self.redis_url, decode_responses=True)
            except ImportError:
                return None
        return self._redis

    async def reserve(self, order_id: str, max_allowed: int) -> bool:
        """
        Reserve one slot cluster-wide. Returns True if reserved.
        max_allowed = max_open_positions - (local positions + local active orders).
        """
        if max_allowed < 1:
            return False
        client = await self._get_redis()
        if not client:
            if self._fail_open:
                logger.warning("Redis unavailable; allowing reservation (single-pod mode, fail_open=True)")
                return True
            logger.error("Redis unavailable; BLOCKING reservation (multi-pod safety, fail_open=False)")
            return False
        try:
            script = """
            local count = tonumber(redis.call('GET', KEYS[1]) or 0)
            if count + 1 <= tonumber(ARGV[1]) then
                redis.call('INCR', KEYS[1])
                redis.call('SADD', KEYS[2], ARGV[2])
                return 1
            end
            return 0
            """
            result = await client.eval(script, 2, RESERVATION_COUNT_KEY, RESERVATION_IDS_KEY, str(max_allowed), order_id)
            return result == 1
        except Exception as e:
            logger.warning("Redis cluster reserve failed: %s", e)
            if self._fail_open:
                return True
            return False

    async def release(self, order_id: str) -> None:
        """Release reservation (SREM + DECR if was present)."""
        client = await self._get_redis()
        if not client:
            return
        try:
            script = """
            if redis.call('SREM', KEYS[1], ARGV[1]) == 1 then
                redis.call('DECR', KEYS[2])
            end
            """
            await client.eval(script, 2, RESERVATION_IDS_KEY, RESERVATION_COUNT_KEY, order_id)
        except Exception as e:
            logger.warning("Redis cluster release failed: %s", e)
