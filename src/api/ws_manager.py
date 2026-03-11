"""
WebSocket connection manager: hold authenticated clients and broadcast events.

Real-time event types broadcasted to all connected clients:
  - snapshot           (periodic 5s: equity, PnL, positions, circuit/kill/feed state)
  - position_update    (when positions change)
  - pnl_update         (periodic PnL updates)
  - signal_generated   (when strategies produce signals)
  - order_submitted    (when an order is submitted)
  - order_filled       (when an order fills)
  - risk_alert         (when risk limits are breached)
  - circuit_state      (when circuit breaker state changes)
  - market_feed_status (market data health status)

Legacy types (still supported):
  order_created, position_updated, equity_updated, risk_updated,
  circuit_open, kill_switch_armed.

JWT validated on connect when JWT_SECRET is set.

Features:
  - Heartbeat ping/pong (30s interval, 90s timeout)
  - Per-client message queue (max 100 messages) for replay on reconnect
  - Graceful disconnect on pong timeout
  - Periodic snapshot broadcast every 5 seconds with full dashboard state
"""

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)

# Heartbeat configuration
HEARTBEAT_INTERVAL_SECONDS = 30
HEARTBEAT_TIMEOUT_SECONDS = 90
MAX_MESSAGE_QUEUE = 100

# Snapshot broadcast interval
SNAPSHOT_INTERVAL_SECONDS = 5


class _ClientState:
    """Per-client tracking state."""

    __slots__ = ("user_id", "last_pong", "message_queue")

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.last_pong: float = time.monotonic()
        self.message_queue: deque = deque(maxlen=MAX_MESSAGE_QUEUE)


class ConnectionManager:
    """Hold WebSocket connections with user identity tracking, heartbeat, message queue,
    and periodic snapshot broadcast."""

    def __init__(self):
        self._connections: dict[WebSocket, _ClientState] = {}
        self._lock = asyncio.Lock()
        self._heartbeat_task: asyncio.Task | None = None
        self._snapshot_task: asyncio.Task | None = None
        self._snapshot_provider: Callable[[], dict[str, Any]] | None = None

    # ── Connection lifecycle ──

    async def connect(self, websocket: WebSocket, user_id: str | None = None, subprotocol: str | None = None) -> None:
        try:
            await websocket.accept(subprotocol=subprotocol)
        except TypeError:
            # Fallback for Starlette versions / mocks that don't support subprotocol kwarg
            await websocket.accept()
        state = _ClientState(user_id or "anonymous")
        async with self._lock:
            self._connections[websocket] = state
            # Start heartbeat if not running
            if self._heartbeat_task is None or self._heartbeat_task.done():
                self._heartbeat_task = asyncio.ensure_future(self._heartbeat_loop())
            # Start snapshot broadcaster if provider is set and not running
            if self._snapshot_provider is not None:
                if self._snapshot_task is None or self._snapshot_task.done():
                    self._snapshot_task = asyncio.ensure_future(self._snapshot_loop())
        logger.debug("WebSocket connected user=%s; total=%s", user_id, len(self._connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.pop(websocket, None)

    def record_pong(self, websocket: WebSocket) -> None:
        """Record pong receipt from client (call from WS receive handler)."""
        state = self._connections.get(websocket)
        if state:
            state.last_pong = time.monotonic()

    # ── Broadcasting ──

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send message to all connected clients. Non-blocking; failures logged."""
        async with self._lock:
            conns = list(self._connections.items())
        if not conns:
            return
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

    async def broadcast_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Broadcast a structured event to all connected clients.

        Sends:
            {"type": event_type, "data": data, "timestamp": "<ISO 8601>"}

        Handles disconnected clients gracefully: dead connections are removed
        and send errors are logged without crashing.
        """
        message = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await self.broadcast(message)

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

    # ── Snapshot provider ──

    def set_snapshot_provider(self, provider: Callable[[], dict[str, Any]]) -> None:
        """Register a callable that returns the current dashboard snapshot dict.

        The provider is called every SNAPSHOT_INTERVAL_SECONDS and the result
        is broadcast as a ``snapshot`` event to all connected clients.

        Args:
            provider: Synchronous callable returning a dict with keys like
                equity, daily_pnl, open_positions_count, circuit_open,
                kill_switch_armed, market_feed_healthy.
        """
        self._snapshot_provider = provider
        # Start snapshot loop immediately if there are already connections
        if self._connections and (self._snapshot_task is None or self._snapshot_task.done()):
            self._snapshot_task = asyncio.ensure_future(self._snapshot_loop())

    async def _snapshot_loop(self) -> None:
        """Broadcast a dashboard snapshot to all clients every SNAPSHOT_INTERVAL_SECONDS."""
        while True:
            try:
                await asyncio.sleep(SNAPSHOT_INTERVAL_SECONDS)

                async with self._lock:
                    if not self._connections:
                        # No connections; stop snapshot loop (will restart on next connect)
                        return

                if self._snapshot_provider is None:
                    return

                try:
                    snapshot_data = self._snapshot_provider()
                except Exception as e:
                    logger.debug("Snapshot provider error: %s", e)
                    continue

                await self.broadcast_event("snapshot", snapshot_data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Snapshot loop error: %s", e)

    # ── Heartbeat ──

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
_manager: ConnectionManager | None = None


def get_ws_manager() -> ConnectionManager | None:
    return _manager


def set_ws_manager(manager: ConnectionManager) -> None:
    global _manager
    _manager = manager
