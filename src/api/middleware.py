"""
Security middleware: headers, rate limiting (Redis-backed with in-memory fallback), request logging.
"""
import logging
import math
import os
import time
from collections import defaultdict
from typing import Callable, Optional, Tuple

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        # HSTS only in production
        if os.environ.get("ENV", "").lower() == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


# Per-endpoint rate limits (requests per minute)
_ENDPOINT_LIMITS = {
    "/api/v1/orders": 30,
    "/api/v1/health": 600,
    "/api/v1/auth": 20,
}


def _get_limit_for_path(path: str, default_limit: int, auth_limit: int) -> int:
    """Determine rate limit based on path. More specific paths checked first."""
    if "/auth/" in path:
        return auth_limit
    for prefix, limit in _ENDPOINT_LIMITS.items():
        if path.startswith(prefix):
            return limit
    return default_limit


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiter per IP with Redis-backed counters (falls back to in-memory).

    Features:
    - Redis-backed distributed counting when REDIS_URL is available
    - Per-endpoint rate limits (orders=30/min, health=600/min, auth=20/min)
    - X-RateLimit-* response headers
    - Automatic fallback to in-memory when Redis is unavailable
    """

    def __init__(self, app, requests_per_minute: int = 120, auth_requests_per_minute: int = 20,
                 trust_proxy: bool = False, max_tracked_ips: int = 10_000):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.auth_requests_per_minute = auth_requests_per_minute
        self.trust_proxy = trust_proxy
        self._max_tracked_ips = max_tracked_ips
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._last_cleanup: float = 0.0

        # Try to connect to Redis for distributed rate limiting
        self._redis = None
        self._redis_available = False
        redis_url = os.environ.get("REDIS_URL") or os.environ.get("RATE_LIMIT_REDIS_URL")
        if redis_url:
            try:
                import redis
                self._redis = redis.Redis.from_url(redis_url, socket_connect_timeout=2, decode_responses=True)
                self._redis.ping()
                self._redis_available = True
                logger.info("Rate limiting: Redis-backed (%s)", redis_url)
            except Exception as e:
                logger.info("Rate limiting: Redis unavailable (%s), using in-memory fallback", e)
                self._redis = None

    def _get_client_ip(self, request: Request) -> str:
        # Only trust X-Forwarded-For when explicitly configured behind a known proxy
        if self.trust_proxy:
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _cleanup_stale_ips(self, now: float) -> None:
        """Prune IPs with no requests in the last 5 minutes to prevent unbounded growth."""
        if now - self._last_cleanup < 300:
            return
        self._last_cleanup = now
        cutoff = now - 300
        stale = [ip for ip, ts_list in self._requests.items() if not ts_list or ts_list[-1] < cutoff]
        for ip in stale:
            del self._requests[ip]

    def _check_rate_memory(self, client_ip: str, limit: int) -> Tuple[bool, int]:
        """In-memory rate check. Returns (is_limited, remaining)."""
        now = time.time()
        window_start = now - 60
        self._cleanup_stale_ips(now)
        # Safety valve: don't track new IPs when at capacity
        if len(self._requests) > self._max_tracked_ips and client_ip not in self._requests:
            return False, limit
        # Clean old entries for this IP
        self._requests[client_ip] = [t for t in self._requests[client_ip] if t > window_start]
        count = len(self._requests[client_ip])
        if count >= limit:
            return True, 0
        self._requests[client_ip].append(now)
        return False, max(0, limit - count - 1)

    def _check_rate_redis(self, client_ip: str, limit: int, path: str) -> Tuple[bool, int]:
        """Redis-backed rate check with sliding window. Returns (is_limited, remaining)."""
        try:
            key = f"rl:{client_ip}:{path.split('/')[3] if len(path.split('/')) > 3 else 'default'}"
            pipe = self._redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, 60)
            results = pipe.execute()
            count = results[0]
            remaining = max(0, limit - count)
            return count > limit, remaining
        except Exception:
            # Fallback to in-memory on Redis error
            return self._check_rate_memory(client_ip, limit)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = self._get_client_ip(request)
        path = request.url.path

        # Determine per-endpoint limit
        limit = _get_limit_for_path(path, self.requests_per_minute, self.auth_requests_per_minute)

        # Check rate limit (Redis or in-memory)
        if self._redis_available and self._redis:
            is_limited, remaining = self._check_rate_redis(client_ip, limit, path)
        else:
            is_limited, remaining = self._check_rate_memory(client_ip, limit)

        if is_limited:
            logger.warning("Rate limited: %s on %s", client_ip, path)
            reset_time = int(time.time()) + 60
            return Response(
                content='{"detail":"Rate limit exceeded. Try again later."}',
                status_code=429,
                media_type="application/json",
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                },
            )

        response = await call_next(request)

        # Add rate limit headers to successful responses
        reset_time = int(time.time()) + 60
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)

        return response
