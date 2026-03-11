"""Circuit breaker: pause new orders and optionally flatten on risk breach.
Supports HALF_OPEN ramp-up: after reset, allows limited trading before full resumption."""
import logging
import threading
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Callable, Optional

from .manager import RiskManager

_FORCE_RESET_COOLDOWN = timedelta(minutes=5)

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """
    Listens to risk events; opens circuit on drawdown/loss breach.
    Optional: call flatten_all() to close positions.
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        on_open: Optional[Callable[[], None]] = None,
        on_close: Optional[Callable[[], None]] = None,
        flatten_on_open: bool = False,
        flatten_callback: Optional[Callable[[], None]] = None,
    ):
        self.risk_manager = risk_manager
        self.on_open = on_open
        self.on_close = on_close
        self.flatten_on_open = flatten_on_open
        self.flatten_callback = flatten_callback
        self._state = CircuitState.CLOSED
        self._peak_equity: float = risk_manager.equity
        self._lock = threading.Lock()
        # Half-open ramp-up state
        self._half_open_ts: Optional[datetime] = None
        self._half_open_trade_count: int = 0
        self._half_open_max_trades: int = 3
        self._half_open_observation_secs: float = 900.0  # 15 minutes
        # Force reset cooldown tracking
        self._last_force_reset_ts: Optional[datetime] = None

    def update_equity(self, current_equity: float) -> None:
        self._peak_equity = max(self._peak_equity, current_equity)
        if self.risk_manager.check_drawdown(self._peak_equity, current_equity):
            self.trip()

    def trip(self) -> None:
        """Open circuit and optionally flatten."""
        if self._state == CircuitState.OPEN:
            return
        self._state = CircuitState.OPEN
        self.risk_manager.open_circuit(reason="drawdown_circuit_breaker")
        logger.warning("Circuit breaker TRIPPED")
        if self.on_open:
            self.on_open()
        if self.flatten_on_open and self.flatten_callback:
            self.flatten_callback()

    def reset(self, current_equity: Optional[float] = None) -> None:
        """Reset to HALF_OPEN first (ramp-up). Full close after observation period.
        Prevents immediate re-entry at 100% after a loss event."""
        self._state = CircuitState.HALF_OPEN
        self._half_open_ts = datetime.now(timezone.utc)
        self._half_open_trade_count = 0
        # Set exposure multiplier to 50% during ramp-up
        self.risk_manager.set_exposure_multiplier(0.5)
        if current_equity is not None:
            self._peak_equity = float(current_equity)
        else:
            self._peak_equity = float(self.risk_manager.equity)
        logger.info(
            "Circuit breaker HALF-OPEN: ramp-up mode (50%% exposure, max %d trades, %ds observation)",
            self._half_open_max_trades, int(self._half_open_observation_secs),
        )

    def check_half_open_promotion(self) -> None:
        """Promote HALF_OPEN to CLOSED after successful observation period."""
        if self._state != CircuitState.HALF_OPEN:
            return
        if self._half_open_ts is None:
            return
        elapsed = (datetime.now(timezone.utc) - self._half_open_ts).total_seconds()
        if elapsed >= self._half_open_observation_secs and self._half_open_trade_count >= self._half_open_max_trades:
            self._state = CircuitState.CLOSED
            self.risk_manager.set_exposure_multiplier(1.0)
            self.risk_manager.close_circuit()
            logger.info(
                "Circuit breaker CLOSED: ramp-up complete (%.0fs elapsed, %d trades)",
                elapsed, self._half_open_trade_count,
            )
            if self.on_close:
                self.on_close()

    def force_reset(self, current_equity: Optional[float] = None) -> None:
        """Force immediate CLOSED (skip HALF_OPEN). Use only for admin override.

        Enforces a minimum cooldown of 5 minutes between force resets to prevent
        accidental rapid toggling. All invocations are audit-logged.
        """
        now = datetime.now(timezone.utc)

        # Enforce minimum cooldown between force resets
        if self._last_force_reset_ts is not None:
            elapsed = now - self._last_force_reset_ts
            if elapsed < _FORCE_RESET_COOLDOWN:
                remaining = (_FORCE_RESET_COOLDOWN - elapsed).total_seconds()
                logger.warning(
                    "AUDIT | force_reset REJECTED — cooldown active. "
                    "Last reset: %s, remaining: %.0fs. Previous state: %s",
                    self._last_force_reset_ts.isoformat(), remaining, self._state.value,
                )
                return

        previous_state = self._state
        self._state = CircuitState.CLOSED
        self.risk_manager.set_exposure_multiplier(1.0)
        self.risk_manager.close_circuit()
        if current_equity is not None:
            self._peak_equity = float(current_equity)
        self._last_force_reset_ts = now
        if self.on_close:
            self.on_close()
        logger.warning(
            "AUDIT | Circuit breaker FORCE RESET to CLOSED (admin override). "
            "Previous state: %s, equity: %s, timestamp: %s",
            previous_state.value,
            current_equity if current_equity is not None else "unchanged",
            now.isoformat(),
        )

    @property
    def state(self) -> CircuitState:
        return self._state

    def allow_order(self) -> bool:
        """Allow order if CLOSED, or limited orders if HALF_OPEN. Thread-safe."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.HALF_OPEN:
                self.check_half_open_promotion()
                if self._state == CircuitState.CLOSED:
                    return True
                if self._half_open_trade_count < self._half_open_max_trades:
                    self._half_open_trade_count += 1
                    logger.info(
                        "HALF_OPEN: allowing trade %d/%d",
                        self._half_open_trade_count, self._half_open_max_trades,
                    )
                    return True
                return False
            return False
