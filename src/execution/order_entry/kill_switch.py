"""
Global kill switch: prevents new orders; allows position reduction only.
Atomic and immediate. Checked at step 4 of OrderEntryService pipeline.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class KillReason(str, Enum):
    MANUAL = "manual"
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_DRAWDOWN = "max_drawdown"
    DRIFT_SPIKE = "drift_spike"
    REJECTION_SPIKE = "rejection_spike"
    BROKER_LATENCY_SPIKE = "broker_latency_spike"
    FILL_MISMATCH = "fill_mismatch"
    INDIA_VIX_SPIKE = "india_vix_spike"
    MARKET_FEED_FAILURE = "market_feed_failure"


# Reasons that NEVER auto-disarm (require human intervention).
# Module-level constant so both KillSwitchState and KillSwitch can reference it.
_MANUAL_DISARM_ONLY = frozenset({
    KillReason.MANUAL,
    KillReason.MAX_DAILY_LOSS,
    KillReason.FILL_MISMATCH,
    KillReason.MAX_DRAWDOWN,
})


@dataclass
class KillSwitchState:
    armed: bool
    reason: Optional[KillReason] = None
    detail: str = ""
    ts: Optional[datetime] = None
    allow_reduce_only: bool = True  # if True, only orders that reduce position are allowed


class KillSwitch:
    """
    Single global kill switch. When armed:
    - New orders that increase exposure are blocked.
    - Orders that reduce position (e.g. SELL to close long) can be allowed if allow_reduce_only=True.
    Supports auto-disarm for recoverable conditions (broker latency, VIX spike, etc.).
    """

    def __init__(self, allow_reduce_only: bool = True, auto_disarm_after_minutes: int = 15):
        self._state = KillSwitchState(armed=False, allow_reduce_only=allow_reduce_only)
        self._lock = asyncio.Lock()
        self._auto_disarm_after_minutes = auto_disarm_after_minutes
        self._arm_timestamp: Optional[datetime] = None
        self._consecutive_healthy_checks: int = 0
        self._required_healthy_checks: int = 3  # 3 consecutive checks before auto-disarm

    async def is_armed(self) -> bool:
        async with self._lock:
            return self._state.armed

    async def arm(self, reason: KillReason, detail: str = "") -> None:
        async with self._lock:
            # Never downgrade from a manual-only reason to an auto-disarmable one.
            # E.g. MAX_DRAWDOWN must never be overwritten by MARKET_FEED_FAILURE.
            if self._state.armed and self._state.reason in _MANUAL_DISARM_ONLY:
                if reason not in _MANUAL_DISARM_ONLY:
                    logger.warning(
                        "Kill switch arm REJECTED: current reason=%s (manual-only) "
                        "cannot be downgraded to %s (auto-disarmable)",
                        self._state.reason.value, reason.value,
                    )
                    return
            self._state = KillSwitchState(
                armed=True,
                reason=reason,
                detail=detail,
                ts=datetime.now(timezone.utc),
                allow_reduce_only=self._state.allow_reduce_only,
            )
            self._arm_timestamp = datetime.now(timezone.utc)
            self._consecutive_healthy_checks = 0
            logger.warning("Kill switch ARMED: reason=%s detail=%s", reason.value, detail)

    async def disarm(self) -> None:
        async with self._lock:
            self._state = KillSwitchState(armed=False, allow_reduce_only=self._state.allow_reduce_only)
            logger.info("Kill switch DISARMED")

    async def get_state(self) -> KillSwitchState:
        async with self._lock:
            return KillSwitchState(
                armed=self._state.armed,
                reason=self._state.reason,
                detail=self._state.detail,
                ts=self._state.ts,
                allow_reduce_only=self._state.allow_reduce_only,
            )

    async def check_auto_disarm(self, broker_healthy: bool = True, vix_value: Optional[float] = None) -> bool:
        """
        Check if kill switch should auto-disarm based on recovery conditions.
        Call from periodic risk snapshot (every 5 minutes).

        Returns True if disarmed.
        Never auto-disarms for: MANUAL, MAX_DAILY_LOSS, FILL_MISMATCH, MAX_DRAWDOWN.
        """
        async with self._lock:
            if not self._state.armed:
                return False

            reason = self._state.reason

            # Never auto-disarm for human-required reasons
            if reason in _MANUAL_DISARM_ONLY:
                return False

            # Check broker-based auto-disarm
            if reason == KillReason.BROKER_LATENCY_SPIKE:
                if broker_healthy:
                    self._consecutive_healthy_checks += 1
                else:
                    self._consecutive_healthy_checks = 0

                if self._consecutive_healthy_checks >= self._required_healthy_checks:
                    self._state = KillSwitchState(armed=False, allow_reduce_only=self._state.allow_reduce_only)
                    self._arm_timestamp = None
                    self._consecutive_healthy_checks = 0
                    logger.info("Kill switch AUTO-DISARMED: broker recovered after %d healthy checks", self._required_healthy_checks)
                    return True

            # Check VIX-based auto-disarm
            if reason == KillReason.INDIA_VIX_SPIKE and vix_value is not None:
                if vix_value < 25.0:
                    self._consecutive_healthy_checks += 1
                else:
                    self._consecutive_healthy_checks = 0

                if self._consecutive_healthy_checks >= self._required_healthy_checks:
                    self._state = KillSwitchState(armed=False, allow_reduce_only=self._state.allow_reduce_only)
                    self._arm_timestamp = None
                    self._consecutive_healthy_checks = 0
                    logger.info("Kill switch AUTO-DISARMED: VIX normalized to %.1f", vix_value)
                    return True

            # Check market feed recovery
            if reason == KillReason.MARKET_FEED_FAILURE:
                if broker_healthy:  # proxy: broker connection implies market data likely back
                    self._consecutive_healthy_checks += 1
                else:
                    self._consecutive_healthy_checks = 0

                if self._consecutive_healthy_checks >= self._required_healthy_checks:
                    self._state = KillSwitchState(armed=False, allow_reduce_only=self._state.allow_reduce_only)
                    self._arm_timestamp = None
                    self._consecutive_healthy_checks = 0
                    logger.info("Kill switch AUTO-DISARMED: market feed recovered")
                    return True

            # Time-based auto-disarm for other recoverable reasons
            if reason in (KillReason.DRIFT_SPIKE, KillReason.REJECTION_SPIKE):
                if self._arm_timestamp:
                    elapsed_min = (datetime.now(timezone.utc) - self._arm_timestamp).total_seconds() / 60
                    if elapsed_min >= self._auto_disarm_after_minutes:
                        self._state = KillSwitchState(armed=False, allow_reduce_only=self._state.allow_reduce_only)
                        self._arm_timestamp = None
                        self._consecutive_healthy_checks = 0
                        logger.info("Kill switch AUTO-DISARMED: timed out after %.0f min (reason=%s)", elapsed_min, reason.value)
                        return True

        return False

    @staticmethod
    def allow_reduce_only_order(state: KillSwitchState, symbol: str, side: str, quantity: int, current_net_position: float) -> bool:
        """
        Given state, return True only if this order reduces net position (e.g. SELL when long).
        current_net_position: positive = long, negative = short.
        """
        if not state.armed:
            return True  # Kill switch not armed — all orders allowed
        if not state.allow_reduce_only:
            return False  # Armed and reduce-only not configured — block everything
        if side.upper() == "SELL" and current_net_position > 0 and quantity <= current_net_position:
            return True
        if side.upper() == "BUY" and current_net_position < 0 and quantity <= abs(current_net_position):
            return True
        return False
