"""
Agent framework: BaseAgent, AgentMessage, AgentOrchestrator.
Manages lifecycle and inter-agent communication via asyncio queues.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Inter-agent communication message."""
    source: str
    target: str  # agent name or "broadcast"
    msg_type: str  # e.g. "opportunity", "risk_alert", "execute_signal"
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class BaseAgent(ABC):
    """Abstract base for all trading agents."""

    name: str = "base_agent"
    description: str = ""

    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._outbox: Optional[asyncio.Queue] = None  # Set by orchestrator
        self._tick_count = 0
        self._last_run: Optional[str] = None
        self._status: str = "idle"
        self._errors: List[str] = []

    def set_outbox(self, queue: asyncio.Queue) -> None:
        self._outbox = queue

    async def send_message(self, target: str, msg_type: str, payload: Dict[str, Any]) -> None:
        if self._outbox:
            msg = AgentMessage(source=self.name, target=target, msg_type=msg_type, payload=payload)
            await self._outbox.put(msg)

    async def receive_message(self) -> Optional[AgentMessage]:
        try:
            return self._inbox.get_nowait()
        except asyncio.QueueEmpty:
            return None

    @abstractmethod
    async def run_cycle(self) -> None:
        """Execute one cycle of the agent's logic."""
        ...

    @property
    @abstractmethod
    def interval_seconds(self) -> float:
        """How often this agent runs its cycle."""
        ...

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "running": self._running,
            "status": self._status,
            "tick_count": self._tick_count,
            "last_run": self._last_run,
            "inbox_size": self._inbox.qsize(),
            "recent_errors": self._errors[-3:],
        }

    async def _loop(self) -> None:
        # Yield control so the event loop can finish startup (e.g. uvicorn)
        await asyncio.sleep(5)
        while self._running:
            try:
                self._status = "running"
                await self.run_cycle()
                self._tick_count += 1
                self._last_run = datetime.now(timezone.utc).isoformat()
                self._status = "idle"
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._status = "error"
                err_msg = f"{type(e).__name__}: {e}"
                self._errors.append(err_msg)
                if len(self._errors) > 10:
                    self._errors = self._errors[-10:]
                logger.exception("Agent %s error: %s", self.name, e)
            await asyncio.sleep(self.interval_seconds)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Agent %s started (interval=%.1fs)", self.name, self.interval_seconds)

    async def stop(self) -> None:
        self._running = False
        self._status = "stopped"
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Agent %s stopped", self.name)


class AgentOrchestrator:
    """Manages agent lifecycle and routes inter-agent messages."""

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._router_task: Optional[asyncio.Task] = None
        self._running = False
        self._on_broadcast: Optional[Callable] = None

    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.name] = agent
        agent.set_outbox(self._message_queue)
        logger.info("Agent registered: %s", agent.name)

    def set_broadcast_callback(self, callback: Callable) -> None:
        """Set callback for broadcasting messages (e.g., to WebSocket)."""
        self._on_broadcast = callback

    def start_all(self) -> None:
        self._running = True
        for agent in self._agents.values():
            agent.start()
        self._router_task = asyncio.create_task(self._route_messages())
        logger.info("AgentOrchestrator started with %d agents", len(self._agents))

    async def stop_all(self) -> None:
        self._running = False
        for agent in self._agents.values():
            await agent.stop()
        if self._router_task:
            self._router_task.cancel()
            try:
                await self._router_task
            except asyncio.CancelledError:
                pass
        logger.info("AgentOrchestrator stopped")

    async def _route_messages(self) -> None:
        """Route messages between agents."""
        while self._running:
            try:
                msg = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                if msg.target == "broadcast":
                    for agent in self._agents.values():
                        if agent.name != msg.source:
                            await agent._inbox.put(msg)
                    if self._on_broadcast:
                        try:
                            await self._on_broadcast(msg)
                        except Exception as e:
                            logger.warning("Broadcast callback failed: %s", e)
                elif msg.target in self._agents:
                    await self._agents[msg.target]._inbox.put(msg)
                else:
                    logger.warning("Message target %s not found, dropping msg from %s (type=%s)",
                                   msg.target, msg.source, msg.msg_type)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Message routing error: %s", e)

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        return self._agents.get(name)

    def get_all_status(self) -> Dict[str, Dict]:
        return {name: agent.get_status() for name, agent in self._agents.items()}
