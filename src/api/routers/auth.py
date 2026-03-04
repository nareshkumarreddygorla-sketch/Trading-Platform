"""
Auth: login, register, token refresh. Issues JWT for get_current_user.
When DATABASE_URL is set, users are persisted (UserRepository). Else hashed in-memory fallback.
Access tokens: 30 minutes. Refresh tokens: 7 days.
"""
import os
import hashlib
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Auth"])

# Access token: short-lived (30 min). Refresh token: long-lived (7 days).
ACCESS_TOKEN_EXPIRY_SECONDS = 30 * 60  # 30 minutes
REFRESH_TOKEN_EXPIRY_SECONDS = 7 * 24 * 3600  # 7 days
MIN_PASSWORD_LENGTH = 8

# In-memory fallback (dev only) - passwords are hashed, never stored plaintext
_users: dict[str, dict] = {}


def _hash_inmemory(password: str) -> str:
    """Hash password for in-memory store (dev only). Not for production."""
    return hashlib.sha256(f"trading-inmem-{password}".encode()).hexdigest()


def _get_secret() -> Optional[str]:
    return os.environ.get("JWT_SECRET") or os.environ.get("AUTH_SECRET")


def _issue_token(username: str, roles: list[str], token_type: str = "access") -> str:
    import jwt
    import time
    secret = _get_secret()
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth not configured (set JWT_SECRET)",
        )
    expiry = ACCESS_TOKEN_EXPIRY_SECONDS if token_type == "access" else REFRESH_TOKEN_EXPIRY_SECONDS
    payload = {
        "sub": username,
        "user_id": username,
        "roles": roles,
        "type": token_type,
        "exp": int(time.time()) + expiry,
        "iat": int(time.time()),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=1, max_length=256)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    email: Optional[str] = Field(None, max_length=256)
    password: str = Field(..., min_length=MIN_PASSWORD_LENGTH, max_length=256)


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/login")
def login(request: Request, body: LoginRequest):
    """Login with username and password. Returns access_token (30m) and refresh_token (7d)."""
    secret = _get_secret()
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth not configured (set JWT_SECRET)",
        )
    # Env-based single user (optional)
    env_user = os.environ.get("AUTH_USERNAME")
    env_pass = os.environ.get("AUTH_PASSWORD")
    if env_user and env_pass and body.username == env_user and body.password == env_pass:
        roles = ["user", "admin"] if os.environ.get("AUTH_ADMIN") == "1" else ["user"]
        return {
            "access_token": _issue_token(body.username, roles, "access"),
            "refresh_token": _issue_token(body.username, roles, "refresh"),
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRY_SECONDS,
        }
    # Persisted users (when DATABASE_URL set)
    user_repo = getattr(request.app.state, "user_repo", None)
    if user_repo is not None:
        verified = user_repo.verify_password(body.username, body.password)
        if verified:
            return {
                "access_token": _issue_token(verified["username"], verified["roles"], "access"),
                "refresh_token": _issue_token(verified["username"], verified["roles"], "refresh"),
                "token_type": "bearer",
                "expires_in": ACCESS_TOKEN_EXPIRY_SECONDS,
            }
    # In-memory fallback (dev only, hashed)
    if body.username in _users and _users[body.username].get("password_hash") == _hash_inmemory(body.password):
        roles = _users[body.username].get("roles", ["user"])
        return {
            "access_token": _issue_token(body.username, roles, "access"),
            "refresh_token": _issue_token(body.username, roles, "refresh"),
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRY_SECONDS,
        }
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid username or password",
    )


@router.post("/refresh")
def refresh_token(body: RefreshRequest):
    """Exchange a valid refresh_token for a new access_token."""
    secret = _get_secret()
    if not secret:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Auth not configured")
    try:
        import jwt
        payload = jwt.decode(body.refresh_token, secret, algorithms=["HS256"])
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token")
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not a refresh token")
    username = payload.get("sub") or payload.get("user_id")
    roles = payload.get("roles", ["user"])
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    return {
        "access_token": _issue_token(username, roles, "access"),
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRY_SECONDS,
    }


@router.post("/register")
def register(request: Request, body: RegisterRequest):
    """Register a new user (persisted when DATABASE_URL set). Then use /auth/login."""
    if not body.username.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username required")
    if len(body.password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
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
    if body.username in _users:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already taken")
    _users[body.username] = {"password_hash": _hash_inmemory(body.password), "email": body.email, "roles": ["user"]}
    logger.info("Registered user (in-memory): %s", body.username)
    return {"message": "Registered. Use /auth/login to get a token."}
