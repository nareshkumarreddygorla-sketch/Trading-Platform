"""
Global kill switch: prevents new orders; allows position reduction only.
Atomic and immediate. Checked at step 4 of OrderEntryService pipeline.
"""
import asyncio
import logging
import os
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


_KILL_STATE_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..", "kill_switch_state.json"))


class KillSwitch:
    """
    Single global kill switch. When armed:
    - New orders that increase exposure are blocked.
    - Orders that reduce position (e.g. SELL to close long) can be allowed if allow_reduce_only=True.
    Supports auto-disarm for recoverable conditions (broker latency, VIX spike, etc.).
    """

    def __init__(self, allow_reduce_only: bool = True, auto_disarm_after_minutes: int = 15,
                 vix_disarm_threshold: float = 25.0):
        self._state = KillSwitchState(armed=False, allow_reduce_only=allow_reduce_only)
        # Note: asyncio.Lock() in __init__ is safe as long as KillSwitch is instantiated
        # within a running event loop (e.g., during lifespan startup). Creating it at
        # module level or in synchronous code requires Python 3.10+ lazy loop binding.
        self._lock = asyncio.Lock()
        self._auto_disarm_after_minutes = auto_disarm_after_minutes
        self._arm_timestamp: Optional[datetime] = None
        self._consecutive_healthy_checks: int = 0
        self._required_healthy_checks: int = 3  # 3 consecutive checks before auto-disarm
        self._vix_disarm_threshold = vix_disarm_threshold

        # Restore persisted state on startup
        saved = self._load_state()
        if saved and saved.get("armed"):
            restored_reduce_only = saved.get("allow_reduce_only", allow_reduce_only)
            self._state = KillSwitchState(
                armed=True,
                reason=KillReason(saved["reason"]) if saved.get("reason") else None,
                detail=saved.get("detail", ""),
                ts=datetime.fromisoformat(saved["armed_at"]) if saved.get("armed_at") else None,
                allow_reduce_only=restored_reduce_only,
            )
            self._arm_timestamp = datetime.fromisoformat(saved["armed_at"]) if saved.get("armed_at") else None
            logger.warning(
                "Kill switch restored from disk: reason=%s allow_reduce_only=%s",
                saved.get("reason"), restored_reduce_only,
            )

    async def _persist_state(self):
        """Persist kill switch state to disk for crash recovery.

        Uses atomic write (write to temp + os.replace) with fsync to ensure
        the state survives both process crashes and power failures.
        """
        import json, tempfile
        state = {
            "armed": self._state.armed,
            "reason": self._state.reason.value if self._state.reason else None,
            "detail": self._state.detail,
            "armed_at": self._arm_timestamp.isoformat() if self._arm_timestamp else None,
            "allow_reduce_only": self._state.allow_reduce_only,
            "persisted_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(_KILL_STATE_PATH), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(_KILL_STATE_PATH))
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(state, f)
                    f.flush()
                    os.fsync(f.fileno())  # Ensure data hits disk before rename
                os.replace(tmp, _KILL_STATE_PATH)
                logger.debug("Kill switch state persisted (armed=%s reason=%s)",
                           state["armed"], state["reason"])
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.error("Failed to persist kill switch state: %s", e)

    def _load_state(self):
        """Load persisted kill switch state on startup.

        Returns the saved state dict if the kill switch was armed, or None.
        This ensures the kill switch survives process restarts: if the system
        was shut down (or crashed) while armed, it stays armed on restart.
        """
        import json
        try:
            if os.path.exists(_KILL_STATE_PATH):
                with open(_KILL_STATE_PATH) as f:
                    raw = f.read().strip()
                if not raw:
                    logger.warning("Kill switch state file is empty, ignoring")
                    return None
                state = json.loads(raw)
                if state.get("armed"):
                    logger.critical(
                        "Kill switch was armed at shutdown (reason=%s, detail=%s, armed_at=%s) "
                        "— re-arming on startup",
                        state.get("reason"), state.get("detail", ""), state.get("armed_at"),
                    )
                    return state
                else:
                    logger.info("Kill switch state file loaded: disarmed (persisted_at=%s)",
                              state.get("persisted_at", "unknown"))
        except json.JSONDecodeError as e:
            logger.error("Kill switch state file is corrupt, ignoring: %s", e)
        except Exception as e:
            logger.warning("Failed to load kill switch state: %s", e)
        return None

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
        await self._persist_state()

    async def disarm(self) -> None:
        async with self._lock:
            self._state = KillSwitchState(armed=False, allow_reduce_only=self._state.allow_reduce_only)
            logger.info("Kill switch DISARMED")
        await self._persist_state()

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
        disarmed = False
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
                    disarmed = True

            # Check VIX-based auto-disarm
            if reason == KillReason.INDIA_VIX_SPIKE and vix_value is not None:
                if vix_value < self._vix_disarm_threshold:
                    self._consecutive_healthy_checks += 1
                else:
                    self._consecutive_healthy_checks = 0

                if self._consecutive_healthy_checks >= self._required_healthy_checks:
                    self._state = KillSwitchState(armed=False, allow_reduce_only=self._state.allow_reduce_only)
                    self._arm_timestamp = None
                    self._consecutive_healthy_checks = 0
                    logger.info("Kill switch AUTO-DISARMED: VIX normalized to %.1f", vix_value)
                    disarmed = True

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
                    disarmed = True

            # Time-based auto-disarm for other recoverable reasons
            if reason in (KillReason.DRIFT_SPIKE, KillReason.REJECTION_SPIKE):
                if self._arm_timestamp:
                    elapsed_min = (datetime.now(timezone.utc) - self._arm_timestamp).total_seconds() / 60
                    if elapsed_min >= self._auto_disarm_after_minutes:
                        self._state = KillSwitchState(armed=False, allow_reduce_only=self._state.allow_reduce_only)
                        self._arm_timestamp = None
                        self._consecutive_healthy_checks = 0
                        logger.info("Kill switch AUTO-DISARMED: timed out after %.0f min (reason=%s)", elapsed_min, reason.value)
                        disarmed = True

        if disarmed:
            await self._persist_state()
        return disarmed

    @staticmethod
    def allow_reduce_only_order(state, symbol, side, quantity, current_net_position, pending_reduce_qty: float = 0.0):
        """Check if an order is allowed under reduce-only mode.

        Args:
            state: KillSwitchState or dict with 'armed' key.
            symbol: Symbol being traded.
            side: "BUY" or "SELL".
            quantity: Order quantity.
            current_net_position: positive = long, negative = short.
            pending_reduce_qty: Total quantity of pending reduce orders for this symbol.
        """
        if not state.get("armed", False) if isinstance(state, dict) else not getattr(state, "armed", False):
            return True
        allow_reduce = state.get("allow_reduce_only", True) if isinstance(state, dict) else getattr(state, "allow_reduce_only", True)
        if not allow_reduce:
            return False
        # Adjust net position for pending reduce orders
        effective_net = current_net_position
        if side.upper() == "SELL" and current_net_position > 0:
            effective_net = current_net_position - pending_reduce_qty
        elif side.upper() == "BUY" and current_net_position < 0:
            effective_net = min(0, current_net_position + pending_reduce_qty)

        if side.upper() == "SELL" and current_net_position > 0 and quantity <= max(0, effective_net):
            return True
        if side.upper() == "BUY" and current_net_position < 0 and quantity <= max(0, abs(effective_net)):
            return True
        return False
