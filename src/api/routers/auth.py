"""
Auth: login, register, token refresh, logout. Issues JWT for get_current_user.
When DATABASE_URL is set, users are persisted (UserRepository). Else hashed in-memory fallback.
Access tokens: 1 hour. Refresh tokens: 7 days.
Logout blacklists the token. Refresh rotation blacklists old refresh tokens.
"""

import hmac
import logging
import os
import re
import threading
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from ..token_blacklist import blacklist_token, is_blacklisted

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Auth"])
_bearer_scheme = HTTPBearer(auto_error=False)

# Access token: short-lived (1 hour). Refresh token: long-lived (7 days).
ACCESS_TOKEN_EXPIRY_SECONDS = 60 * 60  # 1 hour
REFRESH_TOKEN_EXPIRY_SECONDS = 7 * 24 * 3600  # 7 days
MIN_PASSWORD_LENGTH = 12  # Production-grade minimum

# In-memory fallback (dev only) - passwords are hashed, never stored plaintext
_users: dict[str, dict] = {}
_users_lock = threading.Lock()

# Account lockout: 5 failures in 15 minutes
import time as _time  # noqa: E402
from collections import defaultdict  # noqa: E402

_failed_attempts: dict = defaultdict(list)
_MAX_FAILED_ATTEMPTS = 5
_LOCKOUT_DURATION_SECONDS = 900  # 15 minutes
_MAX_TRACKED_USERNAMES = 10_000  # Bound the dict to prevent unbounded memory growth
_last_cleanup_time: float = 0.0


def _cleanup_failed_attempts() -> None:
    """Remove entries where all timestamps are older than the lockout window."""
    global _last_cleanup_time
    now = _time.time()
    # Run cleanup at most once per 60 seconds
    if now - _last_cleanup_time < 60:
        return
    _last_cleanup_time = now
    cutoff = now - _LOCKOUT_DURATION_SECONDS
    stale_keys = [k for k, ts_list in _failed_attempts.items() if not ts_list or all(t <= cutoff for t in ts_list)]
    for k in stale_keys:
        _failed_attempts.pop(k, None)
    # Safety valve: if still over capacity, evict oldest entries
    if len(_failed_attempts) > _MAX_TRACKED_USERNAMES:
        sorted_keys = sorted(
            _failed_attempts.keys(), key=lambda uname: max(_failed_attempts[uname]) if _failed_attempts[uname] else 0
        )
        evict_count = len(_failed_attempts) - _MAX_TRACKED_USERNAMES
        for k in sorted_keys[0:evict_count]:
            _failed_attempts.pop(k, None)


def _check_lockout(username: str) -> bool:
    """Return True if account is locked out."""
    now = _time.time()
    cutoff = now - _LOCKOUT_DURATION_SECONDS
    _failed_attempts[username] = [t for t in _failed_attempts[username] if t > cutoff]
    _cleanup_failed_attempts()
    return len(_failed_attempts[username]) >= _MAX_FAILED_ATTEMPTS


def _record_failed_attempt(username: str) -> None:
    _failed_attempts[username].append(_time.time())


def _clear_failed_attempts(username: str) -> None:
    _failed_attempts.pop(username, None)


async def _blacklist_token_persistent(token: str, ttl: int = 1800):
    """Persist token to Redis blacklist for cross-restart survival."""
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"), socket_connect_timeout=2)
        await r.set(f"blacklist:{token}", "1", ex=ttl)
        await r.aclose()
    except Exception:
        pass  # Redis not available, in-memory only


def _hash_inmemory(password: str) -> str:
    """Hash password for in-memory store using bcrypt (salted, secure)."""
    try:
        from passlib.hash import bcrypt as _bcrypt

        # bcrypt has a 72-byte limit; truncate safely
        return _bcrypt.hash(password[:72])
    except Exception:
        # Fallback: use PBKDF2 from stdlib (works everywhere)
        import hashlib as _hl
        import os

        salt = os.urandom(16).hex()
        return f"pbkdf2:{salt}:{_hl.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()}"


def _verify_inmemory(password: str, hashed: str) -> bool:
    """Verify password against bcrypt or PBKDF2 hash."""
    if hashed.startswith("pbkdf2:"):
        _, salt, expected = hashed.split(":", 2)
        import hashlib as _hl

        actual = _hl.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()
        import hmac

        return hmac.compare_digest(actual, expected)
    try:
        from passlib.hash import bcrypt as _bcrypt

        return _bcrypt.verify(password, hashed)
    except Exception:
        return False


def _get_secret() -> str:
    """Return JWT secret via the centralized auth module (never None)."""
    from ..auth import _get_secret as _central_get_secret

    return _central_get_secret()


def _validate_password_strength(password: str) -> str | None:
    """
    Validate password meets production security requirements.
    Returns an error message if invalid, None if valid.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters"
    if not re.search(r"[A-Z]", password):
        return "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return "Password must contain at least one lowercase letter"
    if not re.search(r"[0-9]", password):
        return "Password must contain at least one digit"
    if not re.search(r"[^A-Za-z0-9]", password):
        return "Password must contain at least one special character"
    return None


def _issue_token(username: str, roles: list[str], token_type: str = "access") -> str:  # noqa: S107
    import time
    import uuid as _uuid

    import jwt

    secret = _get_secret()
    expiry = ACCESS_TOKEN_EXPIRY_SECONDS if token_type == "access" else REFRESH_TOKEN_EXPIRY_SECONDS  # noqa: S105
    now = int(time.time())
    payload = {
        "sub": username,
        "user_id": username,
        "roles": roles,
        "type": token_type,
        "exp": now + expiry,
        "iat": now,
        "nbf": now,  # Not valid before issuance time
        "jti": _uuid.uuid4().hex,  # Unique token ID for revocation tracking
    }
    return jwt.encode(payload, secret, algorithm="HS256")


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9._@-]+$")
    password: str = Field(..., min_length=1, max_length=256)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=128, pattern=r"^[a-zA-Z0-9._@-]+$")
    email: str | None = Field(None, max_length=256, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(
        ...,
        min_length=MIN_PASSWORD_LENGTH,
        max_length=256,
        description="Min 12 chars, must include upper, lower, digit, and special char",
    )


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/login")
def login(request: Request, body: LoginRequest):
    """Login with username and password. Returns access_token (1h) and refresh_token (7d)."""
    # Account lockout check
    if _check_lockout(body.username):
        raise HTTPException(
            status_code=429,
            detail=f"Account locked for {_LOCKOUT_DURATION_SECONDS // 60} minutes due to repeated failed attempts",
        )
    # Env-based single user (optional)
    env_user = os.environ.get("AUTH_USERNAME")
    env_pass = os.environ.get("AUTH_PASSWORD")
    # Use constant-time comparison to prevent timing attacks on password
    if env_user and env_pass and hmac.compare_digest(body.username, env_user):
        if hmac.compare_digest(body.password, env_pass):
            roles = ["user", "admin"] if os.environ.get("AUTH_ADMIN") == "1" else ["user"]
            _clear_failed_attempts(body.username)
            return {
                "access_token": _issue_token(body.username, roles, "access"),
                "refresh_token": _issue_token(body.username, roles, "refresh"),
                "token_type": "bearer",
                "expires_in": ACCESS_TOKEN_EXPIRY_SECONDS,
            }
        # Username matched env credentials but password didn't — reject immediately
        # so the request doesn't fall through to DB check with a different password.
        _record_failed_attempt(body.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    # Persisted users (when DATABASE_URL set)
    user_repo = getattr(request.app.state, "user_repo", None)
    if user_repo is not None:
        verified = user_repo.verify_password(body.username, body.password)
        if verified:
            _clear_failed_attempts(body.username)
            return {
                "access_token": _issue_token(verified["username"], verified["roles"], "access"),
                "refresh_token": _issue_token(verified["username"], verified["roles"], "refresh"),
                "token_type": "bearer",
                "expires_in": ACCESS_TOKEN_EXPIRY_SECONDS,
            }
    # In-memory fallback (dev only, hashed)
    with _users_lock:
        user_data = _users.get(body.username)
    if user_data and _verify_inmemory(body.password, user_data.get("password_hash", "")):
        roles = user_data.get("roles", ["user"])
        _clear_failed_attempts(body.username)
        return {
            "access_token": _issue_token(body.username, roles, "access"),
            "refresh_token": _issue_token(body.username, roles, "refresh"),
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRY_SECONDS,
        }
    _record_failed_attempt(body.username)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid username or password",
    )


@router.post("/refresh")
async def refresh_token(body: RefreshRequest):
    """
    Exchange a valid refresh_token for a new access_token AND a new refresh_token.
    The old refresh token is blacklisted immediately (rotation) to prevent replay attacks.
    """
    # Check blacklist BEFORE decoding
    if is_blacklisted(body.refresh_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked")
    secret = _get_secret()
    try:
        import jwt

        payload = jwt.decode(body.refresh_token, secret, algorithms=["HS256"])
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token"
        ) from None
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not a refresh token")
    username = payload.get("sub") or payload.get("user_id")
    roles = payload.get("roles", ["user"])
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    # Blacklist the OLD refresh token (rotation — prevents replay)
    old_exp = payload.get("exp", time.time() + REFRESH_TOKEN_EXPIRY_SECONDS)
    blacklist_token(body.refresh_token, float(old_exp))
    await _blacklist_token_persistent(body.refresh_token, ttl=int(old_exp - time.time()))

    # Issue both new access AND new refresh tokens
    new_access = _issue_token(username, roles, "access")
    new_refresh = _issue_token(username, roles, "refresh")
    return {
        "access_token": new_access,
        "refresh_token": new_refresh,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRY_SECONDS,
    }


@router.post("/register")
def register(request: Request, body: RegisterRequest):
    """Register a new user (persisted when DATABASE_URL set). Then use /auth/login."""
    if not body.username.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username required")
    password_error = _validate_password_strength(body.password)
    if password_error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=password_error)
    user_repo = getattr(request.app.state, "user_repo", None)
    if user_repo is not None:
        created = user_repo.create(
            username=body.username,
            password=body.password,
            email=body.email,
            roles=["user"],
        )
        if not created:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already taken")
        logger.info("Registered user (persisted): %s", body.username)
        return {"message": "Registered. Use /auth/login to get a token."}
    # In-memory fallback (dev only) - store hashed password
    with _users_lock:
        if body.username in _users:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already taken")
        _users[body.username] = {"password_hash": _hash_inmemory(body.password), "email": body.email, "roles": ["user"]}
    logger.info("Registered user (in-memory): %s", body.username)
    return {"message": "Registered. Use /auth/login to get a token."}


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
):
    """
    Logout: blacklist the current access token so it cannot be reused.
    The token is added to an in-memory (or Redis-backed) blacklist with a TTL
    matching the token's remaining lifetime.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    # Decode to get expiry (but don't fail if token is already expired — still blacklist it)
    secret = _get_secret()
    try:
        import jwt

        payload = jwt.decode(token, secret, algorithms=["HS256"], options={"verify_exp": False})
        expires_at = float(payload.get("exp", time.time() + ACCESS_TOKEN_EXPIRY_SECONDS))
    except Exception:
        # Even if we can't decode, blacklist the raw token with a default TTL
        expires_at = time.time() + ACCESS_TOKEN_EXPIRY_SECONDS

    blacklist_token(token, expires_at)
    await _blacklist_token_persistent(token, ttl=max(int(expires_at - time.time()), 1))
    logger.info("Token blacklisted via /auth/logout")
    return {"message": "Logged out successfully"}
