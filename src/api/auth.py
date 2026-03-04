"""
JWT authentication and RBAC. All critical actions traceable to actor.
When JWT_SECRET is set, protected endpoints require valid Bearer token.
"""
import os
import logging
from typing import List, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

HTTP_BEARER = HTTPBearer(auto_error=False)


def _get_secret() -> Optional[str]:
    return os.environ.get("JWT_SECRET") or os.environ.get("AUTH_SECRET")


def _decode_token(token: str) -> Optional[dict]:
    secret = _get_secret()
    if not secret:
        return None
    try:
        import jwt
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return payload
    except Exception as e:
        logger.debug("JWT decode failed: %s", e)
        return None


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTP_BEARER),
) -> dict:
    """
    Resolve current user from JWT or allow anonymous when JWT_SECRET not set.
    Returns {"user_id": str, "roles": list, "tenant_id": str|None}.
    """
    secret = _get_secret()
    if not secret:
        logger.warning("JWT_SECRET not set — authentication disabled. Set JWT_SECRET for production use.")
        return {"user_id": "system", "roles": ["admin"], "tenant_id": None}

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


def require_roles(allowed: List[str]):
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


def get_actor(request: Request, current_user: Optional[dict] = None) -> str:
    """Return actor string for audit: user_id from JWT or 'api'."""
    if current_user and current_user.get("user_id") and current_user.get("user_id") != "system":
        return str(current_user["user_id"])
    return "api"
