"""
Comprehensive auth security tests: verify that ALL previously-unprotected
endpoints now require JWT authentication (return 401 without Authorization header).

Two verification strategies:
  1. Signature inspection -- assert that endpoint handler functions declare a
     `current_user` parameter backed by get_current_user / require_roles.
  2. HTTP integration -- fire actual requests without an Authorization header
     against the running ASGI app and assert 401 is returned.

Both approaches are tested so regressions are caught even if routing paths change.
"""
import inspect
import os
import time
from typing import List, Optional

import jwt
import pytest
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Environment setup (must precede app import)
# ---------------------------------------------------------------------------
os.environ.setdefault("JWT_SECRET", "test-secret-minimum-32-characters-long!!")
os.environ.setdefault("AUTH_USERNAME", "testadmin")
os.environ.setdefault("AUTH_PASSWORD", "TestP@ss2026!!")
os.environ.setdefault("EXEC_PAPER", "true")
os.environ.setdefault("ENV", "development")

from src.api.app import create_app  # noqa: E402

JWT_SECRET = os.environ["JWT_SECRET"]
API = "/api/v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_token(
    sub: str = "testadmin",
    roles: Optional[List[str]] = None,
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


def _has_auth_dependency(func) -> bool:
    """Return True if the function signature includes a current_user parameter."""
    sig = inspect.signature(func)
    for param in sig.parameters.values():
        if "current_user" in param.name:
            return True
    return False


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
# PART 1: Signature-based verification (no HTTP required)
#
# Assert that endpoint handler functions declare a current_user dependency.
# This catches regressions even when the full app lifespan is not running.
# =========================================================================

class TestSignatureHasAuthDependency:
    """Inspect every sensitive endpoint handler for a current_user parameter."""

    # ── Strategies router ──
    def test_list_strategies_has_auth(self):
        from src.api.routers.strategies import list_strategies
        assert _has_auth_dependency(list_strategies), \
            "list_strategies (GET /strategies) missing auth dependency"

    def test_enable_strategy_has_auth(self):
        from src.api.routers.strategies import enable_strategy
        assert _has_auth_dependency(enable_strategy), \
            "enable_strategy (POST /strategies/{id}/enable) missing auth dependency"

    def test_disable_strategy_has_auth(self):
        from src.api.routers.strategies import disable_strategy
        assert _has_auth_dependency(disable_strategy), \
            "disable_strategy (POST /strategies/{id}/disable) missing auth dependency"

    def test_toggle_strategy_has_auth(self):
        from src.api.routers.strategies import toggle_strategy
        assert _has_auth_dependency(toggle_strategy), \
            "toggle_strategy (PUT /strategies/{id}/toggle) missing auth dependency"

    def test_update_capital_has_auth(self):
        from src.api.routers.strategies import update_capital
        assert _has_auth_dependency(update_capital), \
            "update_capital (PUT /strategies/{id}/capital) missing auth dependency"

    def test_get_signals_has_auth(self):
        from src.api.routers.strategies import get_signals
        assert _has_auth_dependency(get_signals), \
            "get_signals (GET /strategies/signals) missing auth dependency"

    def test_strategies_performance_has_auth(self):
        from src.api.routers.strategies import strategies_performance
        assert _has_auth_dependency(strategies_performance), \
            "strategies_performance (GET /strategies/performance) missing auth dependency"

    # ── Audit router ──
    def test_get_audit_logs_has_auth(self):
        from src.api.routers.audit import get_audit_logs
        assert _has_auth_dependency(get_audit_logs), \
            "get_audit_logs (GET /audit/logs) missing auth dependency"

    # ── Agents router ──
    def test_get_all_agent_status_has_auth(self):
        from src.api.routers.agents import get_all_agent_status
        assert _has_auth_dependency(get_all_agent_status), \
            "get_all_agent_status (GET /agents/status) missing auth dependency"

    def test_get_agent_status_has_auth(self):
        from src.api.routers.agents import get_agent_status
        assert _has_auth_dependency(get_agent_status), \
            "get_agent_status (GET /agents/{name}/status) missing auth dependency"

    def test_stop_agent_has_auth(self):
        from src.api.routers.agents import stop_agent
        assert _has_auth_dependency(stop_agent), \
            "stop_agent (POST /agents/{name}/stop) missing auth dependency"

    def test_start_agent_has_auth(self):
        from src.api.routers.agents import start_agent
        assert _has_auth_dependency(start_agent), \
            "start_agent (POST /agents/{name}/start) missing auth dependency"

    # ── Training router ──
    def test_get_training_status_has_auth(self):
        from src.api.routers.training import get_training_status
        assert _has_auth_dependency(get_training_status), \
            "get_training_status (GET /training/status) missing auth dependency"

    def test_start_training_has_auth(self):
        from src.api.routers.training import start_training
        assert _has_auth_dependency(start_training), \
            "start_training (POST /training/start) missing auth dependency"

    def test_stop_training_has_auth(self):
        from src.api.routers.training import stop_training
        assert _has_auth_dependency(stop_training), \
            "stop_training (POST /training/stop) missing auth dependency"

    def test_get_training_logs_has_auth(self):
        from src.api.routers.training import get_training_logs
        assert _has_auth_dependency(get_training_logs), \
            "get_training_logs (GET /training/logs) missing auth dependency"

    # ── Risk router ──
    def test_risk_snapshot_has_auth(self):
        from src.api.routers.risk import risk_snapshot
        assert _has_auth_dependency(risk_snapshot), \
            "risk_snapshot (GET /risk/snapshot) missing auth dependency"

    def test_risk_state_has_auth(self):
        from src.api.routers.risk import risk_state
        assert _has_auth_dependency(risk_state), \
            "risk_state (GET /risk/state) missing auth dependency"

    def test_risk_positions_has_auth(self):
        from src.api.routers.risk import risk_positions
        assert _has_auth_dependency(risk_positions), \
            "risk_positions (GET /risk/positions) missing auth dependency"

    def test_get_limits_has_auth(self):
        from src.api.routers.risk import get_limits
        assert _has_auth_dependency(get_limits), \
            "get_limits (GET /risk/limits) missing auth dependency"

    def test_update_limits_has_auth(self):
        from src.api.routers.risk import update_limits
        assert _has_auth_dependency(update_limits), \
            "update_limits (PUT /risk/limits) missing auth dependency"

    def test_risk_var_has_auth(self):
        from src.api.routers.risk import risk_var
        assert _has_auth_dependency(risk_var), \
            "risk_var (GET /risk/var) missing auth dependency"

    def test_risk_sectors_has_auth(self):
        from src.api.routers.risk import risk_sectors
        assert _has_auth_dependency(risk_sectors), \
            "risk_sectors (GET /risk/sectors) missing auth dependency"

    def test_risk_tail_has_auth(self):
        from src.api.routers.risk import risk_tail
        assert _has_auth_dependency(risk_tail), \
            "risk_tail (GET /risk/tail) missing auth dependency"

    def test_risk_vol_targeting_has_auth(self):
        from src.api.routers.risk import risk_vol_targeting
        assert _has_auth_dependency(risk_vol_targeting), \
            "risk_vol_targeting (GET /risk/vol-targeting) missing auth dependency"

    def test_risk_correlation_has_auth(self):
        from src.api.routers.risk import risk_correlation
        assert _has_auth_dependency(risk_correlation), \
            "risk_correlation (GET /risk/correlation) missing auth dependency"

    def test_model_weights_has_auth(self):
        from src.api.routers.risk import model_weights
        assert _has_auth_dependency(model_weights), \
            "model_weights (GET /risk/models/weights) missing auth dependency"

    def test_model_drift_has_auth(self):
        from src.api.routers.risk import model_drift
        assert _has_auth_dependency(model_drift), \
            "model_drift (GET /risk/models/drift) missing auth dependency"

    def test_alert_history_has_auth(self):
        from src.api.routers.risk import alert_history
        assert _has_auth_dependency(alert_history), \
            "alert_history (GET /risk/alerts/history) missing auth dependency"

    # ── Trading router ──
    def test_trading_mode_has_auth(self):
        from src.api.routers.trading import trading_mode
        assert _has_auth_dependency(trading_mode), \
            "trading_mode (GET /trading/mode) missing auth dependency"

    def test_trading_ready_is_open(self):
        """trading/ready is a K8s probe -- it must NOT require auth."""
        from src.api.routers.trading import trading_ready
        assert not _has_auth_dependency(trading_ready), \
            "trading_ready (GET /trading/ready) should NOT have auth (K8s probe)"

    def test_trading_stop_has_auth(self):
        from src.api.routers.trading import trading_stop
        assert _has_auth_dependency(trading_stop), \
            "trading_stop (POST /trading/stop) missing auth dependency"

    def test_trading_start_has_auth(self):
        from src.api.routers.trading import trading_start
        assert _has_auth_dependency(trading_start), \
            "trading_start (POST /trading/start) missing auth dependency"

    def test_set_exposure_multiplier_has_auth(self):
        from src.api.routers.trading import set_exposure_multiplier
        assert _has_auth_dependency(set_exposure_multiplier), \
            "set_exposure_multiplier (PUT /trading/exposure_multiplier) missing auth"

    def test_set_autonomous_mode_has_auth(self):
        from src.api.routers.trading import set_autonomous_mode
        assert _has_auth_dependency(set_autonomous_mode), \
            "set_autonomous_mode (PUT /trading/autonomous) missing auth"

    # ── Broker router ──
    def test_broker_status_has_auth(self):
        from src.api.routers.broker import broker_status
        assert _has_auth_dependency(broker_status), \
            "broker_status (GET /broker/status) missing auth dependency"

    def test_configure_broker_has_auth(self):
        from src.api.routers.broker import configure_broker
        assert _has_auth_dependency(configure_broker), \
            "configure_broker (POST /broker/configure) missing auth dependency"

    def test_confirm_live_mode_has_auth(self):
        from src.api.routers.broker import confirm_live_mode
        assert _has_auth_dependency(confirm_live_mode), \
            "confirm_live_mode (POST /broker/confirm-live) missing auth dependency"

    def test_disconnect_broker_has_auth(self):
        from src.api.routers.broker import disconnect_broker
        assert _has_auth_dependency(disconnect_broker), \
            "disconnect_broker (POST /broker/disconnect) missing auth dependency"

    def test_validate_broker_credentials_has_auth(self):
        from src.api.routers.broker import validate_broker_credentials
        assert _has_auth_dependency(validate_broker_credentials), \
            "validate_broker_credentials (POST /broker/validate) missing auth dependency"

    # ── Orders router ──
    def test_list_orders_has_auth(self):
        from src.api.routers.orders import list_orders
        assert _has_auth_dependency(list_orders), \
            "list_orders (GET /orders) missing auth dependency"

    def test_get_order_has_auth(self):
        from src.api.routers.orders import get_order
        assert _has_auth_dependency(get_order), \
            "get_order (GET /orders/{id}) missing auth dependency"

    def test_cancel_order_has_auth(self):
        from src.api.routers.orders import cancel_order
        assert _has_auth_dependency(cancel_order), \
            "cancel_order (POST /orders/{id}/cancel) missing auth dependency"

    def test_list_positions_has_auth(self):
        from src.api.routers.orders import list_positions
        assert _has_auth_dependency(list_positions), \
            "list_positions (GET /positions) missing auth dependency"

    def test_place_order_has_auth(self):
        from src.api.routers.orders import place_order
        assert _has_auth_dependency(place_order), \
            "place_order (POST /orders) missing auth dependency"

    def test_kill_switch_arm_has_auth(self):
        from src.api.routers.orders import kill_switch_arm
        assert _has_auth_dependency(kill_switch_arm), \
            "kill_switch_arm (POST /admin/kill_switch/arm) missing auth dependency"

    def test_kill_switch_disarm_has_auth(self):
        from src.api.routers.orders import kill_switch_disarm
        assert _has_auth_dependency(kill_switch_disarm), \
            "kill_switch_disarm (POST /admin/kill_switch/disarm) missing auth dependency"

    def test_safe_mode_clear_has_auth(self):
        from src.api.routers.orders import safe_mode_clear
        assert _has_auth_dependency(safe_mode_clear), \
            "safe_mode_clear (POST /admin/safe_mode/clear) missing auth dependency"


# =========================================================================
# PART 2: HTTP integration tests -- verify 401 when no auth header
#
# These tests actually hit the ASGI app and confirm the middleware + deps
# reject unauthenticated requests with 401 or 403.
# =========================================================================

class TestEndpointsRequireAuth:
    """Fire requests without Authorization header; expect 401."""

    # ── Strategies ──
    @pytest.mark.asyncio
    async def test_get_strategies_requires_auth(self, client):
        resp = await client.get(f"{API}/strategies")
        assert resp.status_code == 401, f"GET /strategies should return 401, got {resp.status_code}"

    @pytest.mark.asyncio
    async def test_enable_strategy_requires_auth(self, client):
        resp = await client.post(f"{API}/strategies/test_id/enable")
        assert resp.status_code == 401, f"POST /strategies/test_id/enable should return 401, got {resp.status_code}"

    @pytest.mark.asyncio
    async def test_disable_strategy_requires_auth(self, client):
        resp = await client.post(f"{API}/strategies/test_id/disable")
        assert resp.status_code == 401, f"POST /strategies/test_id/disable should return 401, got {resp.status_code}"

    # ── Audit ──
    @pytest.mark.asyncio
    async def test_get_audit_logs_requires_auth(self, client):
        resp = await client.get(f"{API}/audit/logs")
        assert resp.status_code == 401, f"GET /audit/logs should return 401, got {resp.status_code}"

    # ── Risk ──
    @pytest.mark.asyncio
    async def test_risk_snapshot_requires_auth(self, client):
        resp = await client.get(f"{API}/risk/snapshot")
        assert resp.status_code == 401, f"GET /risk/snapshot should return 401, got {resp.status_code}"

    @pytest.mark.asyncio
    async def test_risk_state_requires_auth(self, client):
        resp = await client.get(f"{API}/risk/state")
        assert resp.status_code == 401, f"GET /risk/state should return 401, got {resp.status_code}"

    @pytest.mark.asyncio
    async def test_risk_positions_requires_auth(self, client):
        resp = await client.get(f"{API}/risk/positions")
        assert resp.status_code == 401, f"GET /risk/positions should return 401, got {resp.status_code}"

    @pytest.mark.asyncio
    async def test_risk_limits_get_requires_auth(self, client):
        resp = await client.get(f"{API}/risk/limits")
        assert resp.status_code == 401, f"GET /risk/limits should return 401, got {resp.status_code}"

    @pytest.mark.asyncio
    async def test_risk_var_requires_auth(self, client):
        resp = await client.get(f"{API}/risk/var")
        assert resp.status_code == 401, f"GET /risk/var should return 401, got {resp.status_code}"

    # ── Trading ──
    @pytest.mark.asyncio
    async def test_trading_mode_requires_auth(self, client):
        resp = await client.get(f"{API}/trading/mode")
        assert resp.status_code == 401, f"GET /trading/mode should return 401, got {resp.status_code}"

    @pytest.mark.asyncio
    async def test_trading_ready_does_not_require_auth(self, client):
        """K8s readiness probe -- should be OPEN (no auth). Status may be 200 or 503."""
        resp = await client.get(f"{API}/trading/ready")
        assert resp.status_code in (200, 503), \
            f"GET /trading/ready should NOT return 401 (K8s probe), got {resp.status_code}"

    # ── Broker ──
    @pytest.mark.asyncio
    async def test_broker_status_requires_auth(self, client):
        resp = await client.get(f"{API}/broker/status")
        assert resp.status_code == 401, f"GET /broker/status should return 401, got {resp.status_code}"

    # ── Orders / Positions ──
    @pytest.mark.asyncio
    async def test_list_orders_requires_auth(self, client):
        resp = await client.get(f"{API}/orders")
        assert resp.status_code == 401, f"GET /orders should return 401, got {resp.status_code}"

    @pytest.mark.asyncio
    async def test_list_positions_requires_auth(self, client):
        resp = await client.get(f"{API}/positions")
        assert resp.status_code == 401, f"GET /positions should return 401, got {resp.status_code}"


# =========================================================================
# PART 3: Verify authenticated requests succeed (200, not 401/403)
# =========================================================================

class TestAuthenticatedAccessSucceeds:
    """With a valid JWT, protected endpoints must NOT return 401."""

    @pytest.mark.asyncio
    async def test_strategies_with_auth(self, client, auth_headers):
        resp = await client.get(f"{API}/strategies", headers=auth_headers)
        assert resp.status_code != 401, "GET /strategies should accept valid auth"
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_audit_logs_with_auth(self, client, auth_headers):
        resp = await client.get(f"{API}/audit/logs", headers=auth_headers)
        assert resp.status_code != 401, "GET /audit/logs should accept valid auth"
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_risk_snapshot_with_auth(self, client, auth_headers):
        resp = await client.get(f"{API}/risk/snapshot", headers=auth_headers)
        assert resp.status_code != 401, "GET /risk/snapshot should accept valid auth"
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_risk_state_with_auth(self, client, auth_headers):
        resp = await client.get(f"{API}/risk/state", headers=auth_headers)
        assert resp.status_code != 401, "GET /risk/state should accept valid auth"
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_broker_status_with_auth(self, client, auth_headers):
        resp = await client.get(f"{API}/broker/status", headers=auth_headers)
        assert resp.status_code != 401, "GET /broker/status should accept valid auth"
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_trading_mode_with_auth(self, client, auth_headers):
        resp = await client.get(f"{API}/trading/mode", headers=auth_headers)
        assert resp.status_code != 401, "GET /trading/mode should accept valid auth"
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_positions_with_auth(self, client, auth_headers):
        resp = await client.get(f"{API}/positions", headers=auth_headers)
        assert resp.status_code != 401, "GET /positions should accept valid auth"
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_orders_with_auth(self, client, auth_headers):
        resp = await client.get(f"{API}/orders", headers=auth_headers)
        assert resp.status_code != 401, "GET /orders should accept valid auth"
        assert resp.status_code == 200


# =========================================================================
# PART 4: Token validation edge cases
# =========================================================================

class TestTokenEdgeCases:
    """Verify auth rejection for expired, wrong-type, and missing tokens."""

    @pytest.mark.asyncio
    async def test_expired_token_returns_401(self, client):
        expired = _make_token(exp_delta=-60)
        resp = await client.get(
            f"{API}/strategies",
            headers={"Authorization": f"Bearer {expired}"},
        )
        assert resp.status_code == 401, "Expired token should be rejected"

    @pytest.mark.asyncio
    async def test_wrong_secret_returns_401(self, client):
        now = int(time.time())
        bad_token = jwt.encode(
            {"sub": "user", "roles": ["user"], "type": "access",
             "iat": now, "exp": now + 1800},
            "wrong-secret-key-definitely-not-matching",
            algorithm="HS256",
        )
        resp = await client.get(
            f"{API}/strategies",
            headers={"Authorization": f"Bearer {bad_token}"},
        )
        assert resp.status_code == 401, "Token signed with wrong secret should be rejected"

    @pytest.mark.asyncio
    async def test_malformed_bearer_returns_401(self, client):
        resp = await client.get(
            f"{API}/strategies",
            headers={"Authorization": "Bearer not.a.valid.jwt"},
        )
        assert resp.status_code == 401, "Malformed JWT should be rejected"

    @pytest.mark.asyncio
    async def test_empty_bearer_returns_401(self, client):
        resp = await client.get(
            f"{API}/strategies",
            headers={"Authorization": "Bearer "},
        )
        assert resp.status_code == 401, "Empty bearer should be rejected"

    @pytest.mark.asyncio
    async def test_no_auth_header_returns_401(self, client):
        resp = await client.get(f"{API}/strategies")
        assert resp.status_code == 401, "No auth header should be rejected"


# =========================================================================
# PART 5: Role-based access control
# =========================================================================

class TestRoleBasedAccess:
    """Verify that admin-only endpoints reject non-admin users."""

    @pytest.mark.asyncio
    async def test_enable_strategy_requires_admin(self, client):
        """A user with only 'user' role should get 403 on admin endpoints."""
        user_token = _make_token(roles=["user"])
        resp = await client.post(
            f"{API}/strategies/test_id/enable",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert resp.status_code == 403, \
            f"Non-admin should get 403 on enable_strategy, got {resp.status_code}"

    @pytest.mark.asyncio
    async def test_disable_strategy_requires_admin(self, client):
        user_token = _make_token(roles=["user"])
        resp = await client.post(
            f"{API}/strategies/test_id/disable",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert resp.status_code == 403, \
            f"Non-admin should get 403 on disable_strategy, got {resp.status_code}"

    @pytest.mark.asyncio
    async def test_admin_can_access_admin_endpoints(self, client, auth_headers):
        """An admin user should not get 403."""
        resp = await client.post(
            f"{API}/strategies/nonexistent/enable",
            headers=auth_headers,
        )
        # Should be 404 (strategy not found), NOT 403
        assert resp.status_code != 403, \
            f"Admin should not get 403 on enable_strategy, got {resp.status_code}"
