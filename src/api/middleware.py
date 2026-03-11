"""
Security middleware: headers, rate limiting (Redis-backed with in-memory fallback), request logging.
"""

import logging
import os
import secrets
import time
from collections import defaultdict
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        # X-XSS-Protection is deprecated; CSP provides superior XSS protection
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        # Prevent caching of API responses containing sensitive data
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"
        # Nonce-based CSP: eliminates unsafe-inline/unsafe-eval for XSS protection
        nonce = secrets.token_urlsafe(16)
        request.state.csp_nonce = nonce
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            f"script-src 'self' 'nonce-{nonce}'; "
            f"style-src 'self' 'nonce-{nonce}'; "
            "img-src 'self' data: blob:; "
            "font-src 'self' data:; "
            "connect-src 'self' ws: wss:; "
            "frame-ancestors 'none'"
        )
        # HSTS only in production
        if os.environ.get("ENV", os.environ.get("ENVIRONMENT", "")).lower() == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


# Per-endpoint rate limits (requests per minute).
# More specific paths are checked first to ensure login/register get the
# strictest limit (5/min) rather than the generic auth limit (10/min).
_ENDPOINT_LIMITS = {
    "/api/v1/auth/login": 5,  # Brute-force protection: 5 login attempts/min per IP
    "/api/v1/auth/register": 5,  # Prevent mass account creation
    "/api/v1/auth/refresh": 10,  # Token refresh is less sensitive
    "/api/v1/auth": 10,  # Other auth endpoints
    "/api/v1/orders": 30,
    "/api/v1/health": 600,
}


def _get_limit_for_path(path: str, default_limit: int, auth_limit: int) -> int:
    """Determine rate limit based on path. More specific paths checked first."""
    # Check exact/prefix matches from most-specific to least-specific
    for prefix in sorted(_ENDPOINT_LIMITS.keys(), key=len, reverse=True):
        if path.startswith(prefix):
            return _ENDPOINT_LIMITS[prefix]
    if "/auth/" in path or "/auth" == path.rstrip("/").split("?")[0].rsplit("/", 1)[-1]:
        return auth_limit
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

    def __init__(
        self,
        app,
        requests_per_minute: int = 120,
        auth_requests_per_minute: int = 10,
        trust_proxy: bool = False,
        max_tracked_ips: int = 10_000,
    ):
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

    @staticmethod
    def _rate_key_for_path(path: str) -> str:
        """Derive a rate-limit bucket key from the request path.

        Auth-sensitive endpoints (login, register, refresh) get their own
        per-IP bucket so the strict 5/min limit is enforced independently.
        Other paths share a bucket per top-level resource.
        """
        parts = path.rstrip("/").split("/")
        # /api/v1/auth/login -> "auth/login", /api/v1/auth/register -> "auth/register"
        if len(parts) >= 5 and parts[3] == "auth":
            return f"auth/{parts[4]}"
        if len(parts) > 3:
            return parts[3]
        return "default"

    def _check_rate_memory(self, client_ip: str, limit: int, path: str = "") -> tuple[bool, int]:
        """In-memory rate check. Returns (is_limited, remaining)."""
        now = time.time()
        window_start = now - 60
        self._cleanup_stale_ips(now)
        rate_key = f"{client_ip}:{self._rate_key_for_path(path)}"
        # Safety valve: don't track new IPs when at capacity
        if len(self._requests) > self._max_tracked_ips and rate_key not in self._requests:
            return False, limit
        # Clean old entries for this key
        self._requests[rate_key] = [t for t in self._requests[rate_key] if t > window_start]
        count = len(self._requests[rate_key])
        if count >= limit:
            return True, 0
        self._requests[rate_key].append(now)
        return False, max(0, limit - count - 1)

    def _check_rate_redis(self, client_ip: str, limit: int, path: str) -> tuple[bool, int]:
        """Redis-backed rate check using a fixed-window counter. Returns (is_limited, remaining).

        This is a fixed-window rate limiter: each key gets a 60-second TTL via EXPIRE.
        A client could theoretically burst up to 2x the limit across a window boundary.
        This trade-off is acceptable for simplicity and low Redis overhead. For stricter
        guarantees, consider a sliding-window log or token-bucket algorithm.
        """
        try:
            key = f"rl:{client_ip}:{self._rate_key_for_path(path)}"
            pipe = self._redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, 60)
            results = pipe.execute()
            count = results[0]
            remaining = max(0, limit - count)
            return count >= limit, remaining
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
            is_limited, remaining = self._check_rate_memory(client_ip, limit, path)

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


# Paths to skip in request logging (too noisy / health-check traffic)
_LOG_SKIP_PATHS = frozenset({"/health", "/api/v1/health", "/metrics"})


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with method, path, status, duration, and client IP.

    - Skips /health and /metrics to reduce noise.
    - Uses INFO for 2xx/3xx, WARNING for 4xx, ERROR for 5xx.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        import uuid

        from src.api.logging_config import set_correlation_id

        correlation_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        set_correlation_id(correlation_id)

        path = request.url.path
        if path in _LOG_SKIP_PATHS:
            response = await call_next(request)
            response.headers["X-Request-ID"] = correlation_id
            return response

        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        start = time.time()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.time() - start) * 1000, 1)
            logger.error(
                "%s %s 500 %.1fms ip=%s (unhandled exception)",
                method,
                path,
                duration_ms,
                client_ip,
            )
            raise

        duration_ms = round((time.time() - start) * 1000, 1)
        status = response.status_code

        response.headers["X-Request-ID"] = correlation_id

        if status >= 500:
            logger.error(
                "%s %s %d %.1fms ip=%s",
                method,
                path,
                status,
                duration_ms,
                client_ip,
            )
        elif status >= 400:
            logger.warning(
                "%s %s %d %.1fms ip=%s",
                method,
                path,
                status,
                duration_ms,
                client_ip,
            )
        else:
            logger.info(
                "%s %s %d %.1fms ip=%s",
                method,
                path,
                status,
                duration_ms,
                client_ip,
            )

        return response
