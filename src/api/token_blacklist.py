"""
Token blacklist: revoke JWTs on logout / refresh rotation.
Tries Redis first (distributed, survives restarts). Falls back to a
thread-safe in-memory set with automatic expiry cleanup every 60 s.
"""
import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory fallback
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_blacklist: dict[str, float] = {}  # token -> expires_at (epoch)
_cleanup_thread: Optional[threading.Thread] = None
_cleanup_running = False


def _cleanup_loop() -> None:
    """Background daemon that purges expired tokens every 60 s."""
    global _cleanup_running
    while _cleanup_running:
        try:
            now = time.time()
            with _lock:
                expired = [t for t, exp in _blacklist.items() if exp <= now]
                for t in expired:
                    del _blacklist[t]
                if expired:
                    logger.debug("Token blacklist cleanup: removed %d expired tokens", len(expired))
        except Exception as exc:
            logger.warning("Token blacklist cleanup error: %s", exc)
        # Sleep in small increments so the thread can exit quickly
        for _ in range(60):
            if not _cleanup_running:
                break
            time.sleep(1)


def _ensure_cleanup_thread() -> None:
    global _cleanup_thread, _cleanup_running
    if _cleanup_thread is not None and _cleanup_thread.is_alive():
        return
    _cleanup_running = True
    _cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True, name="token-blacklist-cleanup")
    _cleanup_thread.start()
    logger.debug("Token blacklist cleanup thread started")


def stop_cleanup_thread() -> None:
    """Stop the background cleanup thread (call on app shutdown)."""
    global _cleanup_running
    _cleanup_running = False


# ---------------------------------------------------------------------------
# Redis backend (optional)
# ---------------------------------------------------------------------------
_redis_client = None
_redis_available = False


def _get_redis():
    """Lazily connect to Redis for token blacklisting."""
    global _redis_client, _redis_available
    if _redis_client is not None:
        return _redis_client if _redis_available else None
    redis_url = os.environ.get("REDIS_URL") or os.environ.get("RATE_LIMIT_REDIS_URL")
    if not redis_url:
        _redis_available = False
        return None
    try:
        import redis as _redis_mod
        _redis_client = _redis_mod.Redis.from_url(redis_url, socket_connect_timeout=2, decode_responses=True)
        _redis_client.ping()
        _redis_available = True
        logger.info("Token blacklist: Redis-backed (%s)", redis_url)
        return _redis_client
    except Exception as exc:
        logger.info("Token blacklist: Redis unavailable (%s), using in-memory fallback", exc)
        _redis_client = True  # sentinel so we don't retry
        _redis_available = False
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def blacklist_token(token: str, expires_at: float) -> None:
    """
    Add a token to the blacklist.

    Args:
        token: The raw JWT string to revoke.
        expires_at: Unix epoch when the token naturally expires (used for TTL).
    """
    ttl = max(int(expires_at - time.time()), 1)
    # Try Redis first
    r = _get_redis()
    if r is not None:
        try:
            r.setex(f"token_bl:{token}", ttl, "1")
            logger.debug("Token blacklisted in Redis (TTL=%ds)", ttl)
            return
        except Exception as exc:
            logger.warning("Redis blacklist write failed (%s), falling back to in-memory", exc)

    # In-memory fallback
    _ensure_cleanup_thread()
    with _lock:
        _blacklist[token] = expires_at
    logger.debug("Token blacklisted in-memory (expires_at=%.0f)", expires_at)


def is_blacklisted(token: str) -> bool:
    """
    Check whether a token has been revoked.

    Returns True if the token is in the blacklist and has not yet expired.
    """
    # Try Redis first
    r = _get_redis()
    if r is not None:
        try:
            return r.exists(f"token_bl:{token}") > 0
        except Exception as exc:
            logger.warning("Redis blacklist read failed (%s), falling back to in-memory", exc)

    # In-memory fallback
    with _lock:
        exp = _blacklist.get(token)
        if exp is None:
            return False
        if exp <= time.time():
            # Expired, clean up eagerly
            del _blacklist[token]
            return False
        return True
