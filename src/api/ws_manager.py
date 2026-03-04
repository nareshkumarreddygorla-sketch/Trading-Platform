"""
WebSocket connection manager: hold authenticated clients and broadcast events.
Events: order_created, order_filled, position_updated, equity_updated,
risk_updated, circuit_open, kill_switch_armed.
JWT validated on connect when JWT_SECRET is set.

Features:
  - Heartbeat ping/pong (30s interval, 90s timeout)
  - Per-client message queue (max 100 messages) for replay on reconnect
  - Graceful disconnect on pong timeout
"""
import asyncio
import logging
import time
from collections import deque
from typing import Any, Dict, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)

# Heartbeat configuration
HEARTBEAT_INTERVAL_SECONDS = 30
HEARTBEAT_TIMEOUT_SECONDS = 90
MAX_MESSAGE_QUEUE = 100


class _ClientState:
    """Per-client tracking state."""
    __slots__ = ("user_id", "last_pong", "message_queue")

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.last_pong: float = time.monotonic()
        self.message_queue: deque = deque(maxlen=MAX_MESSAGE_QUEUE)


class ConnectionManager:
    """Hold WebSocket connections with user identity tracking, heartbeat, and message queue."""

    def __init__(self):
        self._connections: Dict[WebSocket, _ClientState] = {}
        self._lock = asyncio.Lock()
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None, subprotocol: Optional[str] = None) -> None:
        await websocket.accept(subprotocol=subprotocol)
        state = _ClientState(user_id or "anonymous")
        async with self._lock:
            self._connections[websocket] = state
            # Start heartbeat if not running
            if self._heartbeat_task is None or self._heartbeat_task.done():
                self._heartbeat_task = asyncio.ensure_future(self._heartbeat_loop())
        logger.debug("WebSocket connected user=%s; total=%s", user_id, len(self._connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.pop(websocket, None)

    def record_pong(self, websocket: WebSocket) -> None:
        """Record pong receipt from client (call from WS receive handler)."""
        state = self._connections.get(websocket)
        if state:
            state.last_pong = time.monotonic()

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send message to all connected clients. Non-blocking; failures logged."""
        async with self._lock:
            conns = list(self._connections.items())
        dead = []
        for ws, state in conns:
            try:
                await ws.send_json(message)
                # Queue for replay on reconnect
                state.message_queue.append(message)
            except Exception as e:
                logger.debug("WebSocket send failed: %s", e)
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)

    async def send_to_user(self, user_id: str, message: dict[str, Any]) -> None:
        """Send message only to a specific user's connections."""
        async with self._lock:
            targets = [(ws, st) for ws, st in self._connections.items() if st.user_id == user_id]
        dead = []
        for ws, state in targets:
            try:
                await ws.send_json(message)
                state.message_queue.append(message)
            except Exception as e:
                logger.debug("WebSocket send_to_user failed: %s", e)
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)

    async def replay_recent_messages(self, websocket: WebSocket, user_id: str, max_messages: int = 50) -> int:
        """
        Replay recent messages to a reconnecting client.
        Call right after connect() for a client that reconnected.
        Returns number of messages replayed.
        """
        # Find any other connection for same user to get queue
        async with self._lock:
            source_queue = None
            for ws, state in self._connections.items():
                if state.user_id == user_id and ws != websocket:
                    source_queue = list(state.message_queue)
                    break

        if not source_queue:
            return 0

        count = 0
        for msg in source_queue[-max_messages:]:
            try:
                await websocket.send_json(msg)
                count += 1
            except Exception:
                break
        return count

    async def _heartbeat_loop(self) -> None:
        """Send ping to all clients every HEARTBEAT_INTERVAL_SECONDS; disconnect stale ones."""
        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)

                async with self._lock:
                    if not self._connections:
                        # No connections; stop heartbeat loop
                        return
                    conns = list(self._connections.items())

                now = time.monotonic()
                dead = []
                for ws, state in conns:
                    # Check pong timeout
                    if now - state.last_pong > HEARTBEAT_TIMEOUT_SECONDS:
                        logger.info(
                            "WebSocket client %s timed out (no pong for %.0fs)",
                            state.user_id,
                            now - state.last_pong,
                        )
                        dead.append(ws)
                        continue

                    # Send ping
                    try:
                        await ws.send_json({"type": "ping", "ts": int(now)})
                    except Exception:
                        dead.append(ws)

                for ws in dead:
                    try:
                        await ws.close(code=1000, reason="heartbeat_timeout")
                    except Exception:
                        pass
                    await self.disconnect(ws)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Heartbeat loop error: %s", e)

    @property
    def active_connections(self) -> int:
        return len(self._connections)


# Module-level singleton; app sets it on startup
_manager: Optional[ConnectionManager] = None


def get_ws_manager() -> Optional[ConnectionManager]:
    return _manager


def set_ws_manager(manager: ConnectionManager) -> None:
    global _manager
    _manager = manager
