"""Circuit breaker: pause new orders and optionally flatten on risk breach."""
import logging
from enum import Enum
from typing import Callable, Optional

from .manager import RiskManager

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

    def update_equity(self, current_equity: float) -> None:
        self._peak_equity = max(self._peak_equity, current_equity)
        if self.risk_manager.check_drawdown(self._peak_equity, current_equity):
            self.trip()

    def trip(self) -> None:
        """Open circuit and optionally flatten."""
        if self._state == CircuitState.OPEN:
            return
        self._state = CircuitState.OPEN
        self.risk_manager.open_circuit()
        logger.warning("Circuit breaker TRIPPED")
        if self.on_open:
            self.on_open()
        if self.flatten_on_open and self.flatten_callback:
            self.flatten_callback()

    def reset(self, current_equity: Optional[float] = None) -> None:
        """Manual reset (e.g. after review). Optionally reset peak to current equity to avoid immediate re-trip."""
        self._state = CircuitState.CLOSED
        self.risk_manager.close_circuit()
        if current_equity is not None:
            self._peak_equity = float(current_equity)
        if self.on_close:
            self.on_close()

    @property
    def state(self) -> CircuitState:
        return self._state

    def allow_order(self) -> bool:
        return self._state == CircuitState.CLOSED
