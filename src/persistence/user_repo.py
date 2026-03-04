"""User repository: create and lookup users by username. Passwords stored hashed."""
import logging
from typing import List, Optional

from sqlalchemy.orm import Session

from .database import session_scope
from .models import UserModel

logger = logging.getLogger(__name__)


try:
    from passlib.context import CryptContext
    _pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
except ImportError:
    _pwd_context = None
    logger.warning("passlib[bcrypt] not installed. Using PBKDF2 fallback. Install passlib[bcrypt] for production.")


def _hash_password(password: str) -> str:
    if _pwd_context is not None:
        return _pwd_context.hash(password)
    # Fallback: PBKDF2-SHA256 with random salt (stdlib)
    import hashlib
    import os
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return f"pbkdf2:sha256:100000${salt.hex()}${dk.hex()}"


def _verify_password(plain: str, hashed: str) -> bool:
    if _pwd_context is not None:
        try:
            return _pwd_context.verify(plain, hashed)
        except Exception:
            pass
    # Handle PBKDF2 fallback format
    if hashed.startswith("pbkdf2:sha256:"):
        import hashlib
        parts = hashed.split("$")
        if len(parts) == 3:
            salt = bytes.fromhex(parts[1])
            expected = parts[2]
            dk = hashlib.pbkdf2_hmac("sha256", plain.encode(), salt, 100_000)
            return dk.hex() == expected
    # Legacy SHA256 migration path
    import hashlib
    return hashlib.sha256(f"trading-platform-salt-{plain}".encode()).hexdigest() == hashed


class UserRepository:
    """Sync repository for users. Use from async via run_in_executor if needed."""

    def get_by_username(self, username: str) -> Optional[dict]:
        """Return user dict with keys: username, password_hash, email, roles (list)."""
        with session_scope() as session:
            m = session.query(UserModel).filter(UserModel.username == username).first()
            if m is None:
                return None
            roles = [r.strip() for r in (m.roles or "user").split(",") if r.strip()]
            return {
                "username": m.username,
                "password_hash": m.password_hash,
                "email": m.email,
                "roles": roles,
            }

    def create(self, username: str, password: str, email: Optional[str] = None, roles: Optional[List[str]] = None) -> bool:
        """Create user with hashed password. Returns True if created, False if username exists."""
        with session_scope() as session:
            if session.query(UserModel).filter(UserModel.username == username).first():
                return False
            roles_str = ",".join(roles or ["user"])
            session.add(UserModel(
                username=username,
                password_hash=_hash_password(password),
                email=email,
                roles=roles_str,
            ))
            return True

    def verify_password(self, username: str, password: str) -> Optional[dict]:
        """If user exists and password matches, return user dict (username, roles). Else None."""
        user = self.get_by_username(username)
        if user is None:
            return None
        if not _verify_password(password, user["password_hash"]):
            return None
        return {"username": user["username"], "roles": user["roles"]}
