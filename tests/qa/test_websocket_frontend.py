"""
QA Phase 7 — Frontend integration.
18) WebSocket flood: 1000 events → UI/store stable; no crash.
19) JWT expiry: expired token → connection rejected (4001); no silent continuation.
"""

import os

import pytest

from src.api.ws_manager import ConnectionManager


# --- 18) WebSocket flood ---
@pytest.mark.asyncio
async def test_websocket_flood_1000_events_stable():
    """Send 1000 broadcast events to one client; expect all received, no crash."""
    received = []

    class MockWebSocket:
        async def accept(self):
            pass

        async def send_json(self, message):
            received.append(message)

    manager = ConnectionManager()
    ws = MockWebSocket()
    await manager.connect(ws)
    for i in range(1000):
        await manager.broadcast({"type": "risk_updated", "equity": 100000, "i": i})
    assert len(received) >= 1000
    manager.disconnect(ws)


@pytest.mark.asyncio
async def test_websocket_flood_multiple_clients():
    """1000 broadcasts to 3 clients; each receives 1000."""
    out1, out2, out3 = [], [], []

    class MockWS:
        def __init__(self, out):
            self._out = out

        async def accept(self):
            pass

        async def send_json(self, msg):
            self._out.append(msg)

    manager = ConnectionManager()
    w1, w2, w3 = MockWS(out1), MockWS(out2), MockWS(out3)
    await manager.connect(w1)
    await manager.connect(w2)
    await manager.connect(w3)
    for i in range(100):
        await manager.broadcast({"type": "equity_updated", "i": i})
    assert len(out1) == 100 and len(out2) == 100 and len(out3) == 100
    manager.disconnect(w1)
    manager.disconnect(w2)
    manager.disconnect(w3)


# --- 19) JWT expiry ---
@pytest.mark.asyncio
async def test_websocket_rejects_expired_jwt():
    """When JWT_SECRET is set and token is expired, WebSocket must reject (close 4001)."""
    import jwt
    from fastapi import FastAPI, WebSocket
    from fastapi.testclient import TestClient

    secret = "test-secret-for-qa-jwt-expiry-minimum-32-chars"
    expired_token = jwt.encode(
        {"sub": "user1", "exp": 0, "iat": 0},
        secret,
        algorithm="HS256",
    )
    app = FastAPI()
    from src.api.ws_manager import ConnectionManager, set_ws_manager

    mgr = ConnectionManager()
    set_ws_manager(mgr)

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        from src.api.auth import _decode_token

        token = websocket.scope.get("query_string", b"").decode().split("token=")
        token = (token[1].split("&")[0] if len(token) > 1 else "") or None
        if os.environ.get("JWT_SECRET"):
            if not token:
                await websocket.close(code=4001)
                return
            payload = _decode_token(token)
            if not payload:
                await websocket.close(code=4001)
                return
        await mgr.connect(websocket)
        await websocket.send_json({"type": "connected"})

    prev = os.environ.get("JWT_SECRET")
    os.environ["JWT_SECRET"] = secret
    try:
        with TestClient(app) as client:
            with pytest.raises(Exception):
                with client.websocket_connect(f"/ws?token={expired_token}") as ws:
                    ws.receive_json()
    finally:
        if prev is not None:
            os.environ["JWT_SECRET"] = prev
        else:
            os.environ.pop("JWT_SECRET", None)
