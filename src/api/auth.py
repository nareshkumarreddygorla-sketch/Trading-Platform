"""
JWT authentication and RBAC. All critical actions traceable to actor.
JWT_SECRET is REQUIRED. If not set, a random secret is generated at startup
and a warning is logged. Auth is NEVER bypassed.
"""

import logging
import os
import secrets

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

HTTP_BEARER = HTTPBearer(auto_error=False)

# Generate a fallback secret at module load if JWT_SECRET is not configured.
# This ensures auth is NEVER bypassed, but tokens won't survive restarts.
_FALLBACK_SECRET: str | None = None


def _get_secret() -> str:
    """Return JWT secret. Always returns a value - never None."""
    global _FALLBACK_SECRET
    secret = os.environ.get("JWT_SECRET") or os.environ.get("AUTH_SECRET")
    if secret:
        return secret
    # No env var set: generate a random secret so auth is still enforced.
    if _FALLBACK_SECRET is None:
        _FALLBACK_SECRET = secrets.token_hex(32)
        logger.warning(
            "JWT_SECRET is not set! A random secret has been generated. "
            "Tokens will NOT survive restarts. Set JWT_SECRET for production use."
        )
    return _FALLBACK_SECRET


def _decode_token(token: str) -> dict | None:
    # Check blacklist BEFORE spending CPU on JWT validation
    from .token_blacklist import is_blacklisted

    if is_blacklisted(token):
        logger.debug("JWT rejected: token is blacklisted")
        return None

    secret = _get_secret()
    try:
        import jwt

        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return payload
    except Exception as e:
        logger.debug("JWT decode failed: %s", e)
        return None


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(HTTP_BEARER),
) -> dict:
    """
    Resolve current user from JWT. Auth is ALWAYS enforced.
    Returns {"user_id": str, "roles": list, "tenant_id": str|None}.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = _decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = payload.get("sub") or payload.get("user_id") or "unknown"
    roles = payload.get("roles") or payload.get("role") or ["user"]
    if isinstance(roles, str):
        roles = [roles]
    tenant_id = payload.get("tenant_id")
    return {"user_id": user_id, "roles": list(roles), "tenant_id": tenant_id}


def require_roles(allowed: list[str]):
    """Dependency: require at least one of allowed roles (e.g. admin)."""

    async def _check(
        current_user: dict = Depends(get_current_user),
    ) -> dict:
        if not current_user.get("roles"):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
        if any(r in allowed for r in current_user["roles"]):
            return current_user
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")

    return _check


def get_actor(request: Request, current_user: dict | None = None) -> str:
    """Return actor string for audit: user_id from JWT or 'api'."""
    if current_user and current_user.get("user_id") and current_user.get("user_id") != "system":
        return str(current_user["user_id"])
    return "api"
