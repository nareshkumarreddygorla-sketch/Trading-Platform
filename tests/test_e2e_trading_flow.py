"""
End-to-end trading flow tests covering the full lifecycle:
  login -> auth -> strategies -> positions -> risk -> orders -> metrics.

Uses the FastAPI TestClient (via httpx AsyncClient + ASGITransport) so
no external services (Redis, Postgres, broker) are required.
"""

import os
import time

import jwt
import pytest
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Environment setup: must happen BEFORE importing the app so that the auth
# module picks up these values at module load time.
# ---------------------------------------------------------------------------
os.environ["JWT_SECRET"] = "test-secret-minimum-32-characters-long!!"
os.environ["AUTH_USERNAME"] = "testadmin"
os.environ["AUTH_PASSWORD"] = "TestP@ss2026!!"
os.environ["AUTH_ADMIN"] = "1"  # Grant admin role so admin-protected endpoints work
os.environ["EXEC_PAPER"] = "true"
os.environ["ENV"] = "development"

from src.api.app import create_app  # noqa: E402
from src.risk_engine.manager import RiskManager  # noqa: E402

API = "/api/v1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """Create a fresh FastAPI app for each test function."""
    _app = create_app()
    # Inject RiskManager so /risk/* endpoints don't return 503 in CI.
    _app.state.risk_manager = RiskManager(equity=1_000_000, load_persisted_state=False)
    return _app


@pytest.fixture
async def client(app):
    """Yield an async httpx client bound to the ASGI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


@pytest.fixture
async def auth_token(client) -> str:
    """Login with env-based credentials and return the access_token."""
    resp = await client.post(
        f"{API}/auth/login",
        json={"username": "testadmin", "password": "TestP@ss2026!!"},
    )
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    body = resp.json()
    assert "access_token" in body
    return body["access_token"]


@pytest.fixture
async def auth_headers(auth_token) -> dict:
    return {"Authorization": f"Bearer {auth_token}"}


# ---------------------------------------------------------------------------
# Helper: create a valid JWT directly (for cases where login is not tested)
# ---------------------------------------------------------------------------


def _make_token(
    sub: str = "testadmin",
    roles: list | None = None,
    token_type: str = "access",
    exp_delta: int = 1800,
) -> str:
    secret = os.environ["JWT_SECRET"]
    now = int(time.time())
    payload = {
        "sub": sub,
        "user_id": sub,
        "roles": roles or ["user", "admin"],
        "type": token_type,
        "iat": now,
        "exp": now + exp_delta,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


# =========================================================================
# 1. Login returns tokens
# =========================================================================


class TestLoginFlow:
    @pytest.mark.asyncio
    async def test_login_returns_tokens(self, client):
        """POST /auth/login returns access_token + refresh_token."""
        resp = await client.post(
            f"{API}/auth/login",
            json={"username": "testadmin", "password": "TestP@ss2026!!"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert "refresh_token" in body
        assert body["token_type"] == "bearer"
        assert body["expires_in"] > 0

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, client):
        """Invalid credentials must return 401."""
        resp = await client.post(
            f"{API}/auth/login",
            json={"username": "testadmin", "password": "WrongPassword!!"},
        )
        assert resp.status_code == 401


# =========================================================================
# 2. Protected endpoint requires auth
# =========================================================================


class TestAuthProtection:
    @pytest.mark.asyncio
    async def test_protected_endpoint_requires_auth(self, client):
        """/api/v1/trading/mode returns 401 without a token."""
        resp = await client.get(f"{API}/orders")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_protected_endpoint_with_auth(self, client, auth_headers):
        """/api/v1/trading/mode returns 200 with a valid token."""
        resp = await client.get(f"{API}/trading/mode", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "mode" in body


# =========================================================================
# 3. Strategies
# =========================================================================


class TestStrategies:
    @pytest.mark.asyncio
    async def test_get_strategies(self, client, auth_headers):
        """GET /api/v1/strategies returns a list of strategies."""
        resp = await client.get(f"{API}/strategies", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "strategies" in body
        assert isinstance(body["strategies"], list)
        assert len(body["strategies"]) > 0
        # Each strategy should have at least id and name
        first = body["strategies"][0]
        assert "id" in first
        assert "name" in first


# =========================================================================
# 4. Positions
# =========================================================================


class TestPositions:
    @pytest.mark.asyncio
    async def test_get_positions(self, client, auth_headers):
        """GET /api/v1/positions returns an array (possibly empty)."""
        resp = await client.get(f"{API}/positions", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "positions" in body
        assert isinstance(body["positions"], list)


# =========================================================================
# 5. Risk state
# =========================================================================


class TestRiskState:
    @pytest.mark.asyncio
    async def test_get_risk_state(self, client, auth_headers):
        """GET /api/v1/risk/state returns risk data."""
        resp = await client.get(f"{API}/risk/state", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "circuit_open" in body
        assert "daily_pnl" in body
        assert "open_positions" in body


# =========================================================================
# 6. Trading mode
# =========================================================================


class TestTradingMode:
    @pytest.mark.asyncio
    async def test_trading_mode_shows_paper(self, client, auth_headers):
        """GET /api/v1/trading/mode shows paper mode when EXEC_PAPER=true."""
        resp = await client.get(f"{API}/trading/mode", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        # In test environment the gateway defaults to paper mode
        assert body["mode"] == "paper"


# =========================================================================
# 7. Autonomous toggle
# =========================================================================


class TestAutonomousToggle:
    @pytest.mark.asyncio
    async def test_autonomous_toggle(self, client, auth_headers):
        """PUT /api/v1/trading/autonomous returns 503 when loop is not configured
        (no autonomous loop in test env), confirming the endpoint exists."""
        resp = await client.put(
            f"{API}/trading/autonomous",
            json={"enabled": True},
            headers=auth_headers,
        )
        # 503 is expected: autonomous_loop is None in test env
        assert resp.status_code in (200, 503)
        if resp.status_code == 503:
            body = resp.json()
            assert "autonomous_loop_not_configured" in body.get("error", "")


# =========================================================================
# 8. Health endpoint
# =========================================================================


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """GET /health returns 200 with status=ok."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# =========================================================================
# 9. Ready endpoint
# =========================================================================


class TestReady:
    @pytest.mark.asyncio
    async def test_ready_endpoint(self, client):
        """GET /ready returns 200 or 503 with checks dict."""
        resp = await client.get("/ready")
        assert resp.status_code in (200, 503)
        body = resp.json()
        assert "checks" in body


# =========================================================================
# 10. Metrics endpoint
# =========================================================================


class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, client):
        """GET /metrics returns prometheus text or 503 when prometheus_client is missing."""
        resp = await client.get("/metrics")
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            # Prometheus metrics are text/plain with metric lines
            assert "text/" in resp.headers.get("content-type", "")


# =========================================================================
# 11. Refresh token flow
# =========================================================================


class TestRefreshTokenFlow:
    @pytest.mark.asyncio
    async def test_refresh_token_flow(self, client):
        """POST /auth/refresh with a valid refresh token returns new tokens."""
        # First login to get a refresh token
        login_resp = await client.post(
            f"{API}/auth/login",
            json={"username": "testadmin", "password": "TestP@ss2026!!"},
        )
        assert login_resp.status_code == 200
        refresh_tok = login_resp.json()["refresh_token"]

        # Exchange refresh token for new tokens
        refresh_resp = await client.post(
            f"{API}/auth/refresh",
            json={"refresh_token": refresh_tok},
        )
        assert refresh_resp.status_code == 200
        body = refresh_resp.json()
        assert "access_token" in body
        assert "refresh_token" in body
        assert body["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_refresh_with_access_token_fails(self, client):
        """Using an access token as a refresh token must fail."""
        access_tok = _make_token(token_type="access")
        resp = await client.post(
            f"{API}/auth/refresh",
            json={"refresh_token": access_tok},
        )
        assert resp.status_code == 401


# =========================================================================
# 12. Order submission (paper mode)
# =========================================================================


class TestOrderSubmission:
    @pytest.mark.asyncio
    async def test_order_submission_paper(self, client, auth_headers):
        """POST /api/v1/orders with valid order in paper mode succeeds or
        returns 503 (order entry not wired), but never 401/422."""
        order = {
            "symbol": "RELIANCE",
            "exchange": "NSE",
            "side": "BUY",
            "quantity": 10,
            "order_type": "LIMIT",
            "limit_price": 2500.0,
        }
        resp = await client.post(
            f"{API}/orders",
            json=order,
            headers=auth_headers,
        )
        # Auth should pass; downstream may return 200/503 depending on
        # whether OrderEntryService was wired during test-app lifespan.
        assert resp.status_code != 401
        assert resp.status_code != 422

    @pytest.mark.asyncio
    async def test_invalid_order_rejected(self, client, auth_headers):
        """POST /api/v1/orders with negative qty returns 422."""
        bad_order = {
            "symbol": "RELIANCE",
            "exchange": "NSE",
            "side": "BUY",
            "quantity": -5,
            "order_type": "MARKET",
        }
        resp = await client.post(
            f"{API}/orders",
            json=bad_order,
            headers=auth_headers,
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_side_rejected(self, client, auth_headers):
        """POST /api/v1/orders with invalid side returns 422."""
        bad_order = {
            "symbol": "RELIANCE",
            "exchange": "NSE",
            "side": "HOLD",
            "quantity": 10,
            "order_type": "MARKET",
        }
        resp = await client.post(
            f"{API}/orders",
            json=bad_order,
            headers=auth_headers,
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_zero_quantity_rejected(self, client, auth_headers):
        """POST /api/v1/orders with quantity=0 returns 422."""
        bad_order = {
            "symbol": "RELIANCE",
            "exchange": "NSE",
            "side": "BUY",
            "quantity": 0,
            "order_type": "MARKET",
        }
        resp = await client.post(
            f"{API}/orders",
            json=bad_order,
            headers=auth_headers,
        )
        assert resp.status_code == 422
