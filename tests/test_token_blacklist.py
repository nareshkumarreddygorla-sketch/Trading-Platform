"""
Token blacklist tests: blacklisting, lookup, expiry cleanup, and
logout integration.

Tests the in-memory fallback path (no Redis) which is the default in tests.
"""
import os
import time
import uuid
from typing import Optional

import jwt
import pytest
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
os.environ["JWT_SECRET"] = "test-secret-minimum-32-characters-long!!"
os.environ["AUTH_USERNAME"] = "testadmin"
os.environ["AUTH_PASSWORD"] = "TestP@ss2026!!"
os.environ["EXEC_PAPER"] = "true"
os.environ["ENV"] = "development"

from src.api.app import create_app  # noqa: E402

JWT_SECRET = os.environ["JWT_SECRET"]
API = "/api/v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_token(
    sub: str = "testadmin",
    roles: Optional[list] = None,
    token_type: str = "access",
    exp_delta: int = 1800,
) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "sub": sub,
            "user_id": sub,
            "roles": roles or ["user", "admin"],
            "type": token_type,
            "iat": now,
            "exp": now + exp_delta,
        },
        JWT_SECRET,
        algorithm="HS256",
    )


# =========================================================================
# 1. Direct blacklist module tests (no FastAPI)
# =========================================================================

class TestBlacklistModule:
    """Test src.api.token_blacklist directly."""

    def setup_method(self):
        """Reset the in-memory blacklist before each test to avoid leaks."""
        from src.api.token_blacklist import _blacklist, _lock
        with _lock:
            _blacklist.clear()

    def test_blacklist_token(self):
        """After blacklisting a token, is_blacklisted returns True."""
        from src.api.token_blacklist import blacklist_token, is_blacklisted

        token = f"test-token-{uuid.uuid4()}"
        expires_at = time.time() + 3600  # 1 hour from now

        assert is_blacklisted(token) is False
        blacklist_token(token, expires_at)
        assert is_blacklisted(token) is True

    def test_non_blacklisted_token(self):
        """A random token that was never blacklisted returns False."""
        from src.api.token_blacklist import is_blacklisted

        token = f"never-blacklisted-{uuid.uuid4()}"
        assert is_blacklisted(token) is False

    def test_expired_tokens_cleaned_up(self):
        """Tokens whose expires_at is in the past are treated as not blacklisted.

        The is_blacklisted function eagerly cleans up expired entries on read.
        """
        from src.api.token_blacklist import blacklist_token, is_blacklisted, _blacklist, _lock

        token = f"expired-token-{uuid.uuid4()}"
        # Set expires_at to 2 seconds in the past
        past_time = time.time() - 2
        blacklist_token(token, past_time)

        # Despite being in the dict, is_blacklisted should return False
        # because it's expired, and it should be cleaned up.
        assert is_blacklisted(token) is False

        # Verify the entry was removed from the dict
        with _lock:
            assert token not in _blacklist

    def test_multiple_tokens_independent(self):
        """Blacklisting one token does not affect others."""
        from src.api.token_blacklist import blacklist_token, is_blacklisted

        token_a = f"token-a-{uuid.uuid4()}"
        token_b = f"token-b-{uuid.uuid4()}"

        blacklist_token(token_a, time.time() + 3600)

        assert is_blacklisted(token_a) is True
        assert is_blacklisted(token_b) is False

    def test_blacklist_with_minimal_ttl(self):
        """A token with TTL of 1 second is still valid immediately after blacklisting."""
        from src.api.token_blacklist import blacklist_token, is_blacklisted

        token = f"short-ttl-{uuid.uuid4()}"
        blacklist_token(token, time.time() + 1)
        assert is_blacklisted(token) is True

    def test_blacklist_same_token_twice(self):
        """Blacklisting the same token twice does not raise and is idempotent."""
        from src.api.token_blacklist import blacklist_token, is_blacklisted

        token = f"double-bl-{uuid.uuid4()}"
        expires_at = time.time() + 3600

        blacklist_token(token, expires_at)
        blacklist_token(token, expires_at)  # No error
        assert is_blacklisted(token) is True


# =========================================================================
# 2. Logout invalidates token (integration via FastAPI)
# =========================================================================

class TestLogoutBlacklist:
    """Test that POST /auth/logout blacklists the token so subsequent
    requests with the same token are rejected."""

    @pytest.mark.asyncio
    async def test_logout_invalidates_token(self):
        """After POST /auth/logout, the old token is rejected on protected endpoints."""
        app = create_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            # Login to get a token
            login_resp = await client.post(
                f"{API}/auth/login",
                json={"username": "testadmin", "password": "TestP@ss2026!!"},
            )
            assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
            token = login_resp.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

            # Verify the token works before logout
            pre_resp = await client.get(f"{API}/trading/mode", headers=headers)
            assert pre_resp.status_code == 200

            # Logout
            logout_resp = await client.post(
                f"{API}/auth/logout",
                headers=headers,
            )
            assert logout_resp.status_code == 200
            assert "logged out" in logout_resp.json()["message"].lower()

            # Now the same token should be rejected (blacklisted)
            post_resp = await client.get(f"{API}/orders", headers=headers)
            assert post_resp.status_code == 401

    @pytest.mark.asyncio
    async def test_logout_without_token_fails(self):
        """POST /auth/logout without a token returns 401."""
        app = create_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(f"{API}/auth/logout")
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_token_rotation_blacklists_old(self):
        """After refreshing, the old refresh token is blacklisted and cannot be reused."""
        app = create_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            # Login to get tokens
            login_resp = await client.post(
                f"{API}/auth/login",
                json={"username": "testadmin", "password": "TestP@ss2026!!"},
            )
            assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
            old_refresh = login_resp.json()["refresh_token"]

            # First refresh: should succeed
            first_refresh = await client.post(
                f"{API}/auth/refresh",
                json={"refresh_token": old_refresh},
            )
            assert first_refresh.status_code == 200

            # Second refresh with the SAME old refresh token: should fail (blacklisted)
            second_refresh = await client.post(
                f"{API}/auth/refresh",
                json={"refresh_token": old_refresh},
            )
            assert second_refresh.status_code == 401
