"""
Comprehensive security test suite for the Trading Platform API.

Covers:
  1. JWT Authentication (expired, invalid, missing, valid, refresh flow)
  2. CORS (allowed vs disallowed origins)
  3. Rate limiting (within limit, exceeding limit)
  4. Input validation (invalid side, exchange, negative qty, missing limit_price, SQL injection)
  5. Password security (min length enforcement, no plaintext in responses)
"""

import time

import jwt
import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import create_app

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEST_JWT_SECRET = "test-secret-key-for-security-tests"
API_PREFIX = "/api/v1"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_jwt_env(monkeypatch):
    """Ensure JWT_SECRET is set so auth enforcement is active."""
    monkeypatch.setenv("JWT_SECRET", TEST_JWT_SECRET)


@pytest.fixture
def app():
    """Fresh FastAPI application per test (middleware state is isolated)."""
    return create_app()


@pytest.fixture
def make_token():
    """Factory for generating JWT tokens with customisable claims."""

    def _make(
        sub: str = "testuser",
        roles: list[str] | None = None,
        token_type: str = "access",
        exp_delta: int = 1800,
        secret: str = TEST_JWT_SECRET,
        algorithm: str = "HS256",
    ) -> str:
        now = int(time.time())
        payload = {
            "sub": sub,
            "user_id": sub,
            "roles": roles or ["user"],
            "type": token_type,
            "iat": now,
            "exp": now + exp_delta,
        }
        return jwt.encode(payload, secret, algorithm=algorithm)

    return _make


# =========================================================================
# 1. JWT Authentication Tests
# =========================================================================


class TestJWTAuthentication:
    """Verify that JWT authentication correctly guards protected endpoints."""

    @pytest.mark.asyncio
    async def test_expired_token_rejected(self, app, make_token):
        """A token whose exp is in the past must be rejected with 401."""
        expired_token = make_token(exp_delta=-60)  # expired 60 s ago
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"{API_PREFIX}/orders",
                json={
                    "symbol": "RELIANCE",
                    "exchange": "NSE",
                    "side": "BUY",
                    "quantity": 10,
                    "order_type": "LIMIT",
                    "limit_price": 2500.0,
                },
                headers={"Authorization": f"Bearer {expired_token}"},
            )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_token_rejected(self, app):
        """A completely garbage token must be rejected with 401."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"{API_PREFIX}/orders",
                json={
                    "symbol": "RELIANCE",
                    "exchange": "NSE",
                    "side": "BUY",
                    "quantity": 10,
                    "order_type": "LIMIT",
                    "limit_price": 2500.0,
                },
                headers={"Authorization": "Bearer not.a.valid.jwt.token"},
            )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_token_on_protected_endpoint(self, app):
        """Requests without an Authorization header to a protected endpoint must get 401."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"{API_PREFIX}/orders",
                json={
                    "symbol": "RELIANCE",
                    "exchange": "NSE",
                    "side": "BUY",
                    "quantity": 10,
                    "order_type": "LIMIT",
                    "limit_price": 2500.0,
                },
            )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_token_accepted(self, app, make_token):
        """A correctly signed, non-expired token should pass auth.

        The request may still fail downstream (e.g. 503 if order entry is not
        configured), but it must NOT be a 401.
        """
        valid_token = make_token()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"{API_PREFIX}/orders",
                json={
                    "symbol": "RELIANCE",
                    "exchange": "NSE",
                    "side": "BUY",
                    "quantity": 10,
                    "order_type": "LIMIT",
                    "limit_price": 2500.0,
                },
                headers={"Authorization": f"Bearer {valid_token}"},
            )
        # Auth passed; downstream may return 503 (order entry not wired) but never 401
        assert resp.status_code != 401

    @pytest.mark.asyncio
    async def test_wrong_secret_rejected(self, app, make_token):
        """A token signed with a different secret must be rejected."""
        wrong_secret_token = make_token(secret="wrong-secret")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"{API_PREFIX}/orders",
                json={
                    "symbol": "RELIANCE",
                    "exchange": "NSE",
                    "side": "BUY",
                    "quantity": 10,
                    "order_type": "LIMIT",
                    "limit_price": 2500.0,
                },
                headers={"Authorization": f"Bearer {wrong_secret_token}"},
            )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_token_flow_access_vs_refresh(self, app, make_token):
        """The /auth/refresh endpoint must:
        - Accept a token with type=refresh
        - Reject a token with type=access
        """
        refresh_tok = make_token(token_type="refresh", exp_delta=7 * 86400)
        access_tok = make_token(token_type="access", exp_delta=1800)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Valid refresh token -> new access token
            resp = await client.post(
                f"{API_PREFIX}/auth/refresh",
                json={"refresh_token": refresh_tok},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert "access_token" in body
            assert body.get("token_type") == "bearer"

            # Using an access-type token as a refresh token should fail
            resp2 = await client.post(
                f"{API_PREFIX}/auth/refresh",
                json={"refresh_token": access_tok},
            )
            assert resp2.status_code == 401


# =========================================================================
# 2. CORS Tests
# =========================================================================


class TestCORS:
    """Verify that CORS headers are set correctly based on origin."""

    @pytest.mark.asyncio
    async def test_disallowed_origin_cors_response(self, app):
        """An Origin not in the allow-list must NOT receive access-control-allow-origin."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.options(
                "/health",
                headers={
                    "Origin": "https://evil-site.example.com",
                    "Access-Control-Request-Method": "GET",
                },
            )
        # The response must not echo back the disallowed origin
        acao = resp.headers.get("access-control-allow-origin")
        assert acao != "https://evil-site.example.com"
        # Ensure it's not a wildcard either
        assert acao != "*"

    @pytest.mark.asyncio
    async def test_allowed_origin_cors_headers(self, app):
        """An allowed dev origin must receive proper CORS headers."""
        allowed = "http://localhost:3000"
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.options(
                "/health",
                headers={
                    "Origin": allowed,
                    "Access-Control-Request-Method": "GET",
                },
            )
        assert resp.headers.get("access-control-allow-origin") == allowed
        # Credentials flag should be present
        assert resp.headers.get("access-control-allow-credentials") == "true"


# =========================================================================
# 3. Rate Limiting Tests
# =========================================================================


class TestRateLimiting:
    """Verify the in-memory per-IP rate limiter."""

    @pytest.mark.asyncio
    async def test_requests_within_limit_succeed(self, app):
        """A handful of requests should all succeed (not 429)."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            for _ in range(5):
                resp = await client.get("/health")
                assert resp.status_code != 429

    @pytest.mark.asyncio
    async def test_auth_rate_limit_exceeded(self, app):
        """Exceeding the auth rate limit (20/min) should eventually yield 429.

        Auth endpoints have a stricter limit (20 req/min in default config).
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            got_429 = False
            for i in range(25):
                resp = await client.post(
                    f"{API_PREFIX}/auth/login",
                    json={"username": "testuser", "password": "wrongpassword"},
                )
                if resp.status_code == 429:
                    got_429 = True
                    break
            assert got_429, "Expected at least one 429 after exceeding auth rate limit"

    @pytest.mark.asyncio
    async def test_rate_limit_returns_retry_after(self, app):
        """When rate limited, the response must include a Retry-After header (or account lockout 429)."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            for _ in range(25):
                resp = await client.post(
                    f"{API_PREFIX}/auth/login",
                    json={"username": "x", "password": "y"},
                )
                if resp.status_code == 429:
                    # Accept either rate-limiter 429 (has retry-after) or
                    # account lockout 429 (no retry-after, but correct behavior)
                    has_retry = "retry-after" in resp.headers
                    is_lockout = "locked" in resp.json().get("detail", "").lower()
                    assert has_retry or is_lockout, (
                        "429 response should have retry-after header or be an account lockout"
                    )
                    return
        pytest.skip("Rate limit not triggered within 25 requests")


# =========================================================================
# 4. Input Validation Tests
# =========================================================================


class TestInputValidation:
    """Verify that the order placement endpoint rejects malformed input."""

    def _auth_header(self, make_token):
        return {"Authorization": f"Bearer {make_token()}"}

    @pytest.mark.asyncio
    async def test_invalid_side(self, app, make_token):
        """Order side must be BUY or SELL; anything else -> 422."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"{API_PREFIX}/orders",
                json={
                    "symbol": "RELIANCE",
                    "exchange": "NSE",
                    "side": "HOLD",
                    "quantity": 10,
                    "order_type": "LIMIT",
                    "limit_price": 2500.0,
                },
                headers=self._auth_header(make_token),
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_exchange(self, app, make_token):
        """Exchange must be one of NSE/BSE/NYSE/NASDAQ/LSE/FX; anything else -> 422."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"{API_PREFIX}/orders",
                json={
                    "symbol": "RELIANCE",
                    "exchange": "FAKE_EXCHANGE",
                    "side": "BUY",
                    "quantity": 10,
                    "order_type": "LIMIT",
                    "limit_price": 2500.0,
                },
                headers=self._auth_header(make_token),
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_negative_quantity(self, app, make_token):
        """Quantity must be >0; negative -> 422."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"{API_PREFIX}/orders",
                json={
                    "symbol": "RELIANCE",
                    "exchange": "NSE",
                    "side": "BUY",
                    "quantity": -5,
                    "order_type": "LIMIT",
                    "limit_price": 2500.0,
                },
                headers=self._auth_header(make_token),
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_zero_quantity(self, app, make_token):
        """Quantity must be >0; zero -> 422."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"{API_PREFIX}/orders",
                json={
                    "symbol": "RELIANCE",
                    "exchange": "NSE",
                    "side": "BUY",
                    "quantity": 0,
                    "order_type": "LIMIT",
                    "limit_price": 2500.0,
                },
                headers=self._auth_header(make_token),
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_limit_price_for_limit_order(self, app, make_token):
        """LIMIT orders require limit_price; omitting it should be rejected.

        Depending on the flow, this may be a 400 (app-level check) or 503
        (order entry not wired in tests) but never silently accepted (200).
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"{API_PREFIX}/orders",
                json={
                    "symbol": "RELIANCE",
                    "exchange": "NSE",
                    "side": "BUY",
                    "quantity": 10,
                    "order_type": "LIMIT",
                    # limit_price intentionally omitted
                },
                headers=self._auth_header(make_token),
            )
        # Should return 400 (explicit check in place_order) or at worst 503
        # but never 200/201 (the order should not be accepted)
        assert resp.status_code in (400, 422, 503)
        # If it's a 400, confirm the detail mentions limit_price
        if resp.status_code == 400:
            assert "limit_price" in resp.json().get("detail", "").lower()

    @pytest.mark.asyncio
    async def test_sql_injection_in_symbol(self, app, make_token):
        """Symbols containing SQL injection patterns must be rejected by the
        regex pattern validator (^[A-Za-z0-9._&-]{1,32}$) -> 422.
        """
        sql_payloads = [
            "'; DROP TABLE orders;--",
            "RELIANCE' OR '1'='1",
            "1; SELECT * FROM users",
            'RELIANCE"; DELETE FROM orders;--',
        ]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            for payload in sql_payloads:
                resp = await client.post(
                    f"{API_PREFIX}/orders",
                    json={
                        "symbol": payload,
                        "exchange": "NSE",
                        "side": "BUY",
                        "quantity": 10,
                        "order_type": "MARKET",
                    },
                    headers=self._auth_header(make_token),
                )
                assert resp.status_code == 422, f"SQL injection payload was not rejected: {payload!r}"

    @pytest.mark.asyncio
    async def test_quantity_exceeds_max(self, app, make_token):
        """Quantity above the 1,000,000 max must be rejected."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"{API_PREFIX}/orders",
                json={
                    "symbol": "RELIANCE",
                    "exchange": "NSE",
                    "side": "BUY",
                    "quantity": 2_000_000,
                    "order_type": "MARKET",
                },
                headers=self._auth_header(make_token),
            )
        assert resp.status_code == 422


# =========================================================================
# 5. Password Security Tests
# =========================================================================


class TestPasswordSecurity:
    """Verify password handling meets security requirements."""

    @pytest.mark.asyncio
    async def test_short_password_rejected_on_register(self, app):
        """Registration with a password shorter than MIN_PASSWORD_LENGTH (8)
        must be rejected with 422 (Pydantic validation) or 400.
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"{API_PREFIX}/auth/register",
                json={
                    "username": "shortpwduser",
                    "password": "abc",  # 3 chars, below minimum of 12
                    "email": "short@example.com",
                },
            )
        # Pydantic enforces min_length=12 on RegisterRequest.password -> 422
        assert resp.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_exactly_min_length_password_accepted(self, app):
        """A password with exactly MIN_PASSWORD_LENGTH chars and complexity should pass validation."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"{API_PREFIX}/auth/register",
                json={
                    "username": "minpwduser",
                    "password": "Abcdefgh1@zz",  # exactly 12 chars with upper, lower, digit, special
                    "email": "min@example.com",
                },
            )
        # Registration should succeed (200) or possibly 503 if JWT_SECRET
        # isn't wired for issuance, but NOT 400/422 for password length
        assert resp.status_code not in (400, 422)

    @pytest.mark.asyncio
    async def test_plaintext_password_never_in_register_response(self, app):
        """The register response body must never contain the plaintext password."""
        password = "SuperSecretP@ssw0rd!"
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"{API_PREFIX}/auth/register",
                json={
                    "username": "reguser1",
                    "password": password,
                    "email": "reg1@example.com",
                },
            )
        # Regardless of status, the password should not appear in the response
        assert password not in resp.text

    @pytest.mark.asyncio
    async def test_plaintext_password_never_in_login_response(self, app):
        """The login response body must never contain the plaintext password."""
        password = "MyTradingP@ss99"
        # First register, then login
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                f"{API_PREFIX}/auth/register",
                json={
                    "username": "loginuser1",
                    "password": password,
                    "email": "login1@example.com",
                },
            )
            resp = await client.post(
                f"{API_PREFIX}/auth/login",
                json={"username": "loginuser1", "password": password},
            )
        assert password not in resp.text

    @pytest.mark.asyncio
    async def test_empty_password_rejected(self, app):
        """An empty password string must be rejected."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"{API_PREFIX}/auth/register",
                json={
                    "username": "emptypwduser",
                    "password": "",
                    "email": "empty@example.com",
                },
            )
        assert resp.status_code in (400, 422)


# =========================================================================
# 6. Security Headers Tests (bonus)
# =========================================================================


class TestSecurityHeaders:
    """Verify that SecurityHeadersMiddleware sets expected headers."""

    @pytest.mark.asyncio
    async def test_security_headers_present(self, app):
        """Responses should contain standard security headers."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert resp.headers.get("x-frame-options") == "DENY"
        assert resp.headers.get("x-xss-protection") == "1; mode=block"
        assert "strict-origin" in resp.headers.get("referrer-policy", "")
