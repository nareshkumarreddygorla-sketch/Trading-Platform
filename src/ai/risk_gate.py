"""
Risk-first control for AI: no AI decision bypasses the risk engine.
All ensemble signals, meta-allocator weights, and LLM suggestions are passed
through the same RiskManager checks before any order or parameter change.
"""
import logging
from typing import Any

from src.core.events import Signal
from src.risk_engine import RiskManager
from src.risk_engine.limits import LimitCheckResult

logger = logging.getLogger(__name__)


class AIRiskGate:
    """
    Ensures every AI-originated action (order from ensemble, weight change from
    meta-allocator, risk param from LLM) is validated by RiskManager.
    """

    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager

    def allow_signal(self, signal: Signal, quantity: int, price: float) -> LimitCheckResult:
        """AI-generated signal must pass same check as any strategy signal."""
        return self.risk_manager.can_place_order(signal, quantity, price)

    def allow_parameter_change(self, param: str, value: Any) -> bool:
        """
        LLM or meta-allocator may suggest e.g. max_position_pct. Only allow if
        within hard bounds and circuit not open.
        """
        if self.risk_manager.is_circuit_open():
            return False
        if param == "max_position_pct":
            return isinstance(value, (int, float)) and 0 < value <= 20.0
        if param == "max_daily_loss_pct":
            return isinstance(value, (int, float)) and 0 < value <= 5.0
        return False

    def max_quantity_for_ai_signal(self, price: float) -> int:
        """Same cap as for any strategy (uses effective equity from exposure multiplier)."""
        return self.risk_manager.max_quantity_for_signal(price)

    def set_exposure_multiplier(self, mult: float) -> None:
        """Set risk manager exposure multiplier from LLM advisory (0.5--1.5)."""
        self.risk_manager.set_exposure_multiplier(mult)
