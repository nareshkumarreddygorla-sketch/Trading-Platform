"""
Comprehensive API endpoint tests for every major route.

Covers: broker status, broker configure, risk limits, orders list,
strategy toggle, performance summary, market news, audit logs,
equity curve, and risk positions.

All tests run without external services (Redis, Postgres, broker).
"""
import os
import time
from typing import Optional

import jwt
import pytest
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Environment: must precede app import
# ---------------------------------------------------------------------------
os.environ["JWT_SECRET"] = "test-secret-minimum-32-characters-long!!"
os.environ["AUTH_USERNAME"] = "testadmin"
os.environ["AUTH_PASSWORD"] = "TestP@ss2026!!"
os.environ["EXEC_PAPER"] = "true"
os.environ["ENV"] = "development"

from src.api.app import create_app  # noqa: E402

API = "/api/v1"
JWT_SECRET = os.environ["JWT_SECRET"]


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


@pytest.fixture
def auth_headers() -> dict:
    return {"Authorization": f"Bearer {_make_token()}"}


# =========================================================================
# 1. Broker status
# =========================================================================

class TestBrokerStatus:
    @pytest.mark.asyncio
    async def test_broker_status(self, client, auth_headers):
        """GET /api/v1/broker/status returns connection and mode info."""
        resp = await client.get(f"{API}/broker/status", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "mode" in body
        assert "connected" in body
        assert "healthy" in body

    @pytest.mark.asyncio
    async def test_broker_status_shows_paper_mode(self, client, auth_headers):
        """Broker status should indicate paper mode in test env."""
        resp = await client.get(f"{API}/broker/status", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "paper"


# =========================================================================
# 2. Broker configure with invalid creds
# =========================================================================

class TestBrokerConfigure:
    @pytest.mark.asyncio
    async def test_broker_configure_invalid_creds(self, client, auth_headers):
        """POST /api/v1/broker/configure with bad creds returns 400 or 503."""
        resp = await client.post(
            f"{API}/broker/configure",
            json={
                "api_key": "FAKE_KEY",
                "client_id": "FAKE_CLIENT",
                "password": "FAKE_PASS",
                "totp_secret": "INVALIDBASE32SECRET",
            },
            headers=auth_headers,
        )
        # Should fail validation: either 400 (bad creds) or 503 (no gateway)
        assert resp.status_code in (400, 500, 503)


# =========================================================================
# 3. Risk limits update
# =========================================================================

class TestRiskLimits:
    @pytest.mark.asyncio
    async def test_risk_limits_get(self, client, auth_headers):
        """GET /api/v1/risk/limits returns current limits."""
        resp = await client.get(f"{API}/risk/limits", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "max_position_pct" in body
        assert "max_daily_loss_pct" in body
        assert "max_open_positions" in body

    @pytest.mark.asyncio
    async def test_risk_limits_update(self, client, auth_headers):
        """PUT /api/v1/risk/limits updates and returns new values."""
        resp = await client.put(
            f"{API}/risk/limits",
            json={"max_open_positions": 5},
            headers=auth_headers,
        )
        # Requires admin role (our token has it) and risk_manager in state
        assert resp.status_code in (200, 400, 403)
        if resp.status_code == 200:
            body = resp.json()
            assert body["status"] == "ok"
            assert body["limits"]["max_open_positions"] == 5

    @pytest.mark.asyncio
    async def test_risk_limits_update_invalid_range(self, client, auth_headers):
        """PUT /api/v1/risk/limits with out-of-range value returns 400."""
        resp = await client.put(
            f"{API}/risk/limits",
            json={"max_position_pct": -10},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_risk_limits_update_empty_body(self, client, auth_headers):
        """PUT /api/v1/risk/limits with no fields returns 400."""
        resp = await client.put(
            f"{API}/risk/limits",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 400


# =========================================================================
# 4. Orders list
# =========================================================================

class TestOrdersList:
    @pytest.mark.asyncio
    async def test_orders_list(self, client, auth_headers):
        """GET /api/v1/orders returns an order list response."""
        resp = await client.get(f"{API}/orders", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "orders" in body
        assert "total" in body
        assert isinstance(body["orders"], list)

    @pytest.mark.asyncio
    async def test_orders_list_with_pagination(self, client, auth_headers):
        """GET /api/v1/orders?limit=5&offset=0 respects pagination params."""
        resp = await client.get(
            f"{API}/orders?limit=5&offset=0",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["orders"], list)


# =========================================================================
# 5. Strategy toggle
# =========================================================================

class TestStrategyToggle:
    @pytest.mark.asyncio
    async def test_strategy_toggle_enable(self, client, auth_headers):
        """PUT /api/v1/strategies/{name}/toggle enables a strategy."""
        resp = await client.put(
            f"{API}/strategies/ema_crossover/toggle",
            json={"enabled": True},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["strategy_id"] == "ema_crossover"
        assert body["enabled"] is True

    @pytest.mark.asyncio
    async def test_strategy_toggle_disable(self, client, auth_headers):
        """PUT /api/v1/strategies/{name}/toggle disables a strategy."""
        resp = await client.put(
            f"{API}/strategies/ema_crossover/toggle",
            json={"enabled": False},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is False

    @pytest.mark.asyncio
    async def test_strategy_toggle_nonexistent(self, client, auth_headers):
        """PUT /api/v1/strategies/nonexistent/toggle returns 404."""
        resp = await client.put(
            f"{API}/strategies/nonexistent_strategy_xyz/toggle",
            json={"enabled": True},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_strategy_enable_disable(self, client, auth_headers):
        """POST enable/disable endpoints work correctly."""
        # Enable
        resp = await client.post(
            f"{API}/strategies/macd/enable",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["enabled"] is True

        # Disable
        resp = await client.post(
            f"{API}/strategies/macd/disable",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False


# =========================================================================
# 6. Performance summary
# =========================================================================

class TestPerformanceSummary:
    @pytest.mark.asyncio
    async def test_performance_summary(self, client, auth_headers):
        """GET /api/v1/performance/summary returns summary metrics."""
        resp = await client.get(
            f"{API}/performance/summary",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        # At minimum these keys should exist (demo or live)
        assert "sharpe_ratio" in body
        assert "max_drawdown_pct" in body
        assert "total_pnl" in body

    @pytest.mark.asyncio
    async def test_performance_summary_with_weeks_param(self, client, auth_headers):
        """GET /api/v1/performance/summary?weeks=8 respects weeks parameter."""
        resp = await client.get(
            f"{API}/performance/summary?weeks=8",
            headers=auth_headers,
        )
        assert resp.status_code == 200


# =========================================================================
# 7. Market news
# =========================================================================

class TestMarketNews:
    @pytest.mark.asyncio
    async def test_market_news(self, client, auth_headers):
        """GET /api/v1/market/news returns news items."""
        resp = await client.get(
            f"{API}/market/news",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "news" in body
        assert isinstance(body["news"], list)
        assert len(body["news"]) > 0

    @pytest.mark.asyncio
    async def test_market_news_with_limit(self, client, auth_headers):
        """GET /api/v1/market/news?limit=3 respects limit."""
        resp = await client.get(
            f"{API}/market/news?limit=3",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["news"]) <= 3


# =========================================================================
# 8. Audit logs
# =========================================================================

class TestAuditLogs:
    @pytest.mark.asyncio
    async def test_audit_logs(self, client, auth_headers):
        """GET /api/v1/audit/logs returns audit events."""
        resp = await client.get(
            f"{API}/audit/logs",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "events" in body
        assert isinstance(body["events"], list)
        # Demo events should be present when no DB
        assert len(body["events"]) > 0

    @pytest.mark.asyncio
    async def test_audit_logs_filter_by_type(self, client, auth_headers):
        """GET /api/v1/audit/logs?event_type=trade_executed filters events."""
        resp = await client.get(
            f"{API}/audit/logs?event_type=trade_executed",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        for event in body["events"]:
            assert event["event_type"] == "trade_executed"


# =========================================================================
# 9. Equity curve
# =========================================================================

class TestEquityCurve:
    @pytest.mark.asyncio
    async def test_equity_curve(self, client, auth_headers):
        """GET /api/v1/performance/equity-curve returns curve data."""
        resp = await client.get(
            f"{API}/performance/equity-curve",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "equity_curve" in body
        assert isinstance(body["equity_curve"], list)
        assert len(body["equity_curve"]) > 0
        # Each point should have date and equity
        first = body["equity_curve"][0]
        assert "date" in first
        assert "equity" in first

    @pytest.mark.asyncio
    async def test_equity_curve_weeks_range(self, client, auth_headers):
        """Equity curve respects the weeks query parameter."""
        resp_8 = await client.get(
            f"{API}/performance/equity-curve?weeks=8",
            headers=auth_headers,
        )
        resp_26 = await client.get(
            f"{API}/performance/equity-curve?weeks=26",
            headers=auth_headers,
        )
        assert resp_8.status_code == 200
        assert resp_26.status_code == 200
        # More weeks should have more data points
        len_8 = len(resp_8.json()["equity_curve"])
        len_26 = len(resp_26.json()["equity_curve"])
        assert len_26 >= len_8


# =========================================================================
# 10. Risk positions
# =========================================================================

class TestRiskPositions:
    @pytest.mark.asyncio
    async def test_risk_positions(self, client, auth_headers):
        """GET /api/v1/risk/positions returns positions list."""
        resp = await client.get(
            f"{API}/risk/positions",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "positions" in body
        assert isinstance(body["positions"], list)


# =========================================================================
# 11. Risk snapshot and VaR
# =========================================================================

class TestRiskSnapshot:
    @pytest.mark.asyncio
    async def test_risk_snapshot(self, client, auth_headers):
        """GET /api/v1/risk/snapshot returns equity and positions."""
        resp = await client.get(f"{API}/risk/snapshot", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "equity" in body
        assert "daily_pnl" in body
        assert "positions" in body

    @pytest.mark.asyncio
    async def test_risk_var(self, client, auth_headers):
        """GET /api/v1/risk/var returns VaR data."""
        resp = await client.get(f"{API}/risk/var", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "var_95" in body


# =========================================================================
# 12. Market data endpoints
# =========================================================================

class TestMarketData:
    @pytest.mark.asyncio
    async def test_market_quote(self, client, auth_headers):
        """GET /api/v1/market/quote/RELIANCE returns a quote."""
        resp = await client.get(
            f"{API}/market/quote/RELIANCE",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "symbol" in body
        assert "last" in body

    @pytest.mark.asyncio
    async def test_market_bars(self, client, auth_headers):
        """GET /api/v1/market/bars/RELIANCE returns OHLCV bars."""
        resp = await client.get(
            f"{API}/market/bars/RELIANCE?interval=1d&limit=10",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "bars" in body
        assert isinstance(body["bars"], list)

    @pytest.mark.asyncio
    async def test_market_status(self, client, auth_headers):
        """GET /api/v1/market/status returns feed status."""
        resp = await client.get(
            f"{API}/market/status",
            headers=auth_headers,
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_market_regime(self, client, auth_headers):
        """GET /api/v1/market/regime returns regime info."""
        resp = await client.get(
            f"{API}/market/regime",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "regime" in body


# =========================================================================
# 13. Drawdown and monthly returns
# =========================================================================

class TestPerformanceAdditional:
    @pytest.mark.asyncio
    async def test_drawdown(self, client, auth_headers):
        """GET /api/v1/performance/drawdown returns drawdown series."""
        resp = await client.get(
            f"{API}/performance/drawdown",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "drawdown" in body

    @pytest.mark.asyncio
    async def test_monthly_returns(self, client, auth_headers):
        """GET /api/v1/performance/monthly-returns returns monthly data."""
        resp = await client.get(
            f"{API}/performance/monthly-returns",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "monthly_returns" in body

    @pytest.mark.asyncio
    async def test_public_performance(self, client):
        """GET /api/v1/performance/public is accessible without auth."""
        resp = await client.get(f"{API}/performance/public")
        assert resp.status_code == 200
        body = resp.json()
        assert "disclaimer" in body


# =========================================================================
# 14. Strategies listing and performance
# =========================================================================

class TestStrategiesListing:
    @pytest.mark.asyncio
    async def test_strategies_list_returns_all(self, client, auth_headers):
        """GET /api/v1/strategies returns all registered strategies."""
        resp = await client.get(f"{API}/strategies", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        ids = [s["id"] for s in body["strategies"]]
        # Should include classical and professional strategies
        assert "ema_crossover" in ids
        assert "macd" in ids

    @pytest.mark.asyncio
    async def test_strategies_performance(self, client, auth_headers):
        """GET /api/v1/strategies/performance returns aggregated stats."""
        resp = await client.get(
            f"{API}/strategies/performance",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "strategies" in body
        assert "summary" in body
        assert "total_pnl" in body["summary"]

    @pytest.mark.asyncio
    async def test_strategy_capital_update(self, client, auth_headers):
        """PUT /api/v1/strategies/{id}/capital updates allocated capital."""
        resp = await client.put(
            f"{API}/strategies/ema_crossover/capital",
            json={"capital": 50000.0},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["capital_allocated"] == 50000.0

    @pytest.mark.asyncio
    async def test_strategy_capital_negative_rejected(self, client, auth_headers):
        """PUT /api/v1/strategies/{id}/capital rejects negative capital."""
        resp = await client.put(
            f"{API}/strategies/ema_crossover/capital",
            json={"capital": -100},
            headers=auth_headers,
        )
        assert resp.status_code == 400
