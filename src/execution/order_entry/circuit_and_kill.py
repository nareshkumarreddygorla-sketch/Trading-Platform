"""
Circuit breaker + global kill switch integration.
Auto-triggers: max daily loss, max drawdown, rejection spike, broker latency spike, fill mismatch, India VIX spike.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime

from .kill_switch import KillReason, KillSwitch

logger = logging.getLogger(__name__)


@dataclass
class CircuitKillConfig:
    max_daily_loss_pct: float = 2.0
    max_drawdown_pct: float = 5.0
    rejection_spike_window: int = 20
    rejection_spike_threshold: int = 20  # was 5; raised for multi-strategy paper trading
    broker_latency_spike_ms: float = 5000.0
    india_vix_spike_multiplier: float = 2.0
    vix_reference: float = 15.0


class CircuitAndKillController:
    """
    Wraps RiskManager (circuit) and KillSwitch. Provides:
    - Hard max daily loss: open circuit + arm kill switch.
    - Max drawdown: open circuit + arm kill switch.
    - Manual kill: admin calls arm(KillReason.MANUAL).
    - Auto kill on: rejection spike, broker latency spike, fill mismatch, India VIX spike.
    """

    def __init__(
        self,
        risk_manager,
        kill_switch: KillSwitch,
        config: CircuitKillConfig | None = None,
    ):
        self.risk_manager = risk_manager
        self.kill_switch = kill_switch
        self.config = config or CircuitKillConfig()
        self._recent_rejections: deque[datetime] = deque(maxlen=max(1, self.config.rejection_spike_window * 2))
        self._lock = asyncio.Lock()

    async def check_daily_loss_and_trip(self, current_equity: float) -> bool:
        """If daily loss exceeds limit: open circuit and arm kill switch. Returns True if tripped."""
        if current_equity <= 0:
            await self.kill_switch.arm(KillReason.MAX_DAILY_LOSS, "zero_equity")
            self.risk_manager.open_circuit(reason="zero_equity")
            return True
        loss_pct = -100.0 * self.risk_manager.daily_pnl / current_equity if self.risk_manager.daily_pnl < 0 else 0
        if loss_pct >= self.config.max_daily_loss_pct:
            await self.kill_switch.arm(KillReason.MAX_DAILY_LOSS, f"daily_loss_{loss_pct:.2f}pct")
            self.risk_manager.open_circuit(reason=f"daily_loss_{loss_pct:.2f}pct")
            return True
        return False

    async def check_drawdown_and_trip(self, peak_equity: float, current_equity: float) -> bool:
        if self.risk_manager.check_drawdown(peak_equity, current_equity):
            await self.kill_switch.arm(KillReason.MAX_DRAWDOWN, "drawdown_limit")
            self.risk_manager.open_circuit(reason="max_drawdown")
            return True
        return False

    async def record_rejection(self) -> None:
        """Call on each risk/gate rejection. Trip if spike within time window."""
        async with self._lock:
            now = datetime.now(UTC)
            self._recent_rejections.append(now)
            # Time-based decay: only count rejections within last 5 minutes
            from datetime import timedelta

            cutoff = now - timedelta(minutes=5)
            while self._recent_rejections and self._recent_rejections[0] < cutoff:
                self._recent_rejections.popleft()
            if len(self._recent_rejections) >= self.config.rejection_spike_threshold:
                await self.kill_switch.arm(
                    KillReason.REJECTION_SPIKE, f"rejections_{len(self._recent_rejections)}_in_5min"
                )
                logger.warning("Kill switch: rejection spike")

    async def check_broker_latency_and_trip(self, latency_ms: float) -> bool:
        if latency_ms >= self.config.broker_latency_spike_ms:
            await self.kill_switch.arm(KillReason.BROKER_LATENCY_SPIKE, f"latency_{latency_ms:.0f}ms")
            return True
        return False

    async def trip_fill_mismatch(self, detail: str = "") -> None:
        await self.kill_switch.arm(KillReason.FILL_MISMATCH, detail or "reconciliation_mismatch")
        self.risk_manager.open_circuit(reason=f"fill_mismatch: {detail or 'reconciliation_mismatch'}")

    async def check_india_vix_and_trip(self, india_vix: float) -> bool:
        if india_vix >= self.config.vix_reference * self.config.india_vix_spike_multiplier:
            await self.kill_switch.arm(KillReason.INDIA_VIX_SPIKE, f"vix_{india_vix:.1f}")
            return True
        return False

    async def check_market_feed_and_trip(self, feed_healthy: bool) -> bool:
        """If market feed is unhealthy, arm kill switch and open circuit. Returns True if tripped."""
        if not feed_healthy:
            await self.kill_switch.arm(KillReason.MARKET_FEED_FAILURE, "market_feed_unhealthy")
            self.risk_manager.open_circuit(reason="market_feed_unhealthy")
            return True
        return False
