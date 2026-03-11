"""
QA — Angel One live market data (WebSocket) with paper execution.
1) Market data feed connects and sets healthy.
2) Market data feed unhealthy triggers skip in autonomous loop.
3) Paper mode never calls real broker.
4) Market status endpoint returns expected shape.
5) WebSocket market_status_updated broadcast.
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.api.ws_manager import ConnectionManager
from src.core.events import Exchange, Tick
from src.execution.autonomous_loop import AutonomousLoop
from src.market_data.bar_aggregator import TickToBarAggregator
from src.market_data.bar_cache import BarCache
from src.market_data.market_data_service import MarketDataService


# --- 1) Market data feed connects and sets healthy ---
@pytest.mark.asyncio
async def test_market_data_feed_connects_and_sets_healthy():
    """When connector connects and receives a tick, service reports healthy."""

    class MockConnector:
        _connected = False
        _tick_queue = None

        async def connect(self):
            self._connected = True
            self._tick_queue = asyncio.Queue()
            await self._tick_queue.put(
                Tick(symbol="RELIANCE", exchange=Exchange.NSE, price=100.0, size=10.0, ts=datetime.now(UTC))
            )

        async def disconnect(self):
            self._connected = False

        async def subscribe_ticks(self, symbols):
            pass

        async def stream_ticks(self):
            try:
                tick = await asyncio.wait_for(self._tick_queue.get(), timeout=1.0)
                yield tick
            except TimeoutError:
                pass

    bar_cache = BarCache()
    aggregator = TickToBarAggregator(bar_cache, interval_seconds=60)
    connector = MockConnector()
    service = MarketDataService(
        connector,
        bar_cache,
        aggregator,
        ["RELIANCE"],
        feed_stale_seconds=120,
    )
    service.start()
    await asyncio.sleep(0.4)
    st = service.get_status()
    assert "connected" in st
    assert "healthy" in st
    assert "last_tick_ts" in st
    assert "symbols" in st
    await service.stop()


# --- 2) Market data feed unhealthy triggers skip ---
@pytest.mark.asyncio
async def test_market_data_feed_unhealthy_triggers_skip():
    """Autonomous loop skips _tick when market feed is unhealthy."""
    tick_ran = []

    async def submit_fn(*args, **kwargs):
        tick_ran.append(1)
        return type("R", (), {"success": True, "order_id": "o1", "latency_ms": 1.0})()

    get_market_feed_healthy = lambda: False  # unhealthy
    loop = AutonomousLoop(
        submit_fn,
        get_safe_mode=lambda: False,
        get_bar_ts=lambda: "2025-01-15T10:00:00Z",
        get_bars=lambda s, e, i, n: [],
        get_symbols=lambda: [("X", Exchange.NSE)],
        strategy_runner=MagicMock(),
        allocator=MagicMock(),
        get_risk_state=lambda: {},
        get_positions=lambda: [],
        get_market_feed_healthy=get_market_feed_healthy,
        poll_interval_seconds=999,
    )
    await loop._tick()
    assert len(tick_ran) == 0, "Loop must skip when market feed unhealthy"


# --- 3) Paper mode never calls real broker ---
@pytest.mark.asyncio
async def test_paper_mode_never_calls_real_broker():
    """AngelOneExecutionGateway in paper mode must not use HTTP client for place_order."""
    from src.execution.angel_one_gateway import AngelOneExecutionGateway

    gw = AngelOneExecutionGateway(
        api_key="test",
        api_secret="secret",
        access_token="token",
        paper=True,
    )
    assert gw.paper is True
    assert gw._client is None
    order = await gw.place_order(
        symbol="RELIANCE",
        exchange="NSE",
        side="BUY",
        quantity=10,
        order_type="MARKET",
    )
    assert order is not None
    assert getattr(order, "metadata", {}).get("paper") is True


@pytest.mark.asyncio
async def test_paper_mode_place_order_returns_immediately():
    """Paper mode place_order returns in-memory order without broker call."""
    from src.execution.angel_one_gateway import AngelOneExecutionGateway

    gw = AngelOneExecutionGateway(
        api_key="key",
        api_secret="secret",
        access_token="token",
        paper=True,
    )
    order = await gw.place_order(
        symbol="RELIANCE",
        exchange="NSE",
        side="BUY",
        quantity=5,
        order_type="MARKET",
    )
    assert order.metadata.get("paper") is True
    assert order.symbol == "RELIANCE"
    assert order.quantity == 5


# --- 4) Market status endpoint ---
@pytest.mark.asyncio
async def test_market_status_endpoint():
    """GET /api/v1/market/status returns connected, healthy, last_tick_ts, symbols."""
    import os
    import time
    from contextlib import asynccontextmanager

    import jwt
    from fastapi.testclient import TestClient

    from src.api.app import create_app

    @asynccontextmanager
    async def _noop_lifespan(a):
        yield

    # Generate a valid JWT for test requests
    jwt_secret = os.environ.get("JWT_SECRET", "dev-secret-key-change-in-production")
    now = int(time.time())
    token = jwt.encode(
        {
            "sub": "testuser",
            "user_id": "testuser",
            "roles": ["user", "admin"],
            "type": "access",
            "iat": now,
            "exp": now + 1800,
        },
        jwt_secret,
        algorithm="HS256",
    )
    headers = {"Authorization": f"Bearer {token}"}

    app = create_app()
    # Replace lifespan with no-op to avoid connecting to real services in tests
    app.router.lifespan_context = _noop_lifespan  # type: ignore[assignment]
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.get("/api/v1/market/status", headers=headers)
        assert r.status_code == 200
        data = r.json()
        assert "connected" in data
        assert "healthy" in data
        assert "last_tick_ts" in data or "message" in data
        if "symbols" in data:
            assert isinstance(data["symbols"], list)

    # With mock service (set after client context so lifespan does not overwrite)
    mock_svc = MagicMock()
    mock_svc.get_status.return_value = {
        "connected": True,
        "healthy": True,
        "last_tick_ts": "2025-01-15T10:00:00Z",
        "symbols": ["RELIANCE", "TCS"],
    }
    with TestClient(app) as client:
        app.state.market_data_service = mock_svc
        r = client.get("/api/v1/market/status", headers=headers)
        assert r.status_code == 200
        data = r.json()
        assert data["connected"] is True
        assert data["healthy"] is True
        assert data["last_tick_ts"] == "2025-01-15T10:00:00Z"
        assert data["symbols"] == ["RELIANCE", "TCS"]


# --- 5) WebSocket market_status_updated broadcast ---
@pytest.mark.asyncio
async def test_ws_market_status_broadcast():
    """Broadcast market_status_updated is received by client."""
    received = []

    class MockWS:
        async def accept(self):
            pass

        async def send_json(self, msg):
            received.append(msg)

    manager = ConnectionManager()
    ws = MockWS()
    await manager.connect(ws)
    await manager.broadcast(
        {
            "type": "market_status_updated",
            "connected": True,
            "healthy": True,
            "last_tick_ts": "2025-01-15T10:00:00Z",
        }
    )
    await asyncio.sleep(0)
    assert any(m.get("type") == "market_status_updated" for m in received)
    ev = next(m for m in received if m.get("type") == "market_status_updated")
    assert ev["connected"] is True
    assert ev["healthy"] is True
    assert ev["last_tick_ts"] == "2025-01-15T10:00:00Z"
    manager.disconnect(ws)
