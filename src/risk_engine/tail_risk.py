"""
Black swan / tail risk protection module.

Multi-layer defence:
  1. India VIX monitoring with graduated response (warning → reduce → kill switch)
  2. Rapid drawdown detection (2% in 30 min → emergency reduce)
  3. Post-circuit recovery ramp-up (25% → 50% → 100% over 3 sessions)
  4. Cascading circuit (no new positions for rest of session after trip)
"""
from __future__ import annotations

import logging
import time as _time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional, Tuple

logger = logging.getLogger(__name__)


class VIXLevel(str, Enum):
    NORMAL = "normal"         # VIX < 20
    ELEVATED = "elevated"     # 20 <= VIX < 25
    HIGH = "high"             # 25 <= VIX < 35
    EXTREME = "extreme"       # VIX >= 35


class RecoveryPhase(str, Enum):
    NORMAL = "normal"           # 100% exposure allowed
    PHASE_1 = "phase_1"        # 25% max exposure (first session after circuit)
    PHASE_2 = "phase_2"        # 50% max exposure (second session)
    PHASE_3 = "phase_3"        # 100% restored (third session)


@dataclass
class TailRiskState:
    """Current tail risk assessment."""
    vix_level: VIXLevel = VIXLevel.NORMAL
    vix_value: float = 0.0
    exposure_scale: float = 1.0      # multiplier for position sizes
    rapid_drawdown: bool = False
    recovery_phase: RecoveryPhase = RecoveryPhase.NORMAL
    recovery_max_exposure_pct: float = 100.0
    last_circuit_trip_ts: Optional[float] = None
    sessions_since_trip: int = 999

    def as_dict(self) -> dict:
        return {
            "vix_level": self.vix_level.value,
            "vix_value": round(self.vix_value, 2),
            "exposure_scale": round(self.exposure_scale, 2),
            "rapid_drawdown": self.rapid_drawdown,
            "recovery_phase": self.recovery_phase.value,
            "recovery_max_exposure_pct": self.recovery_max_exposure_pct,
            "sessions_since_trip": self.sessions_since_trip,
        }


class TailRiskProtector:
    """
    Multi-layer black swan protection system.
    """

    def __init__(
        self,
        vix_warning_level: float = 20.0,
        vix_reduce_level: float = 25.0,
        vix_extreme_level: float = 35.0,
        rapid_drawdown_pct: float = 2.0,
        rapid_drawdown_window_min: int = 30,
        recovery_sessions: int = 3,
    ):
        self.vix_warning_level = vix_warning_level
        self.vix_reduce_level = vix_reduce_level
        self.vix_extreme_level = vix_extreme_level
        self.rapid_drawdown_pct = rapid_drawdown_pct
        self.rapid_drawdown_window_min = rapid_drawdown_window_min
        self.recovery_sessions = recovery_sessions

        # Equity snapshots for rapid drawdown detection: (timestamp, equity)
        self._equity_snapshots: Deque[Tuple[float, float]] = deque(maxlen=120)
        self._state = TailRiskState()
        self._circuit_tripped_this_session = False

    @property
    def state(self) -> TailRiskState:
        return self._state

    def update_vix(self, india_vix: float) -> TailRiskState:
        """
        Update VIX level and adjust exposure scaling.

        Returns updated TailRiskState.
        """
        self._state.vix_value = india_vix

        if india_vix >= self.vix_extreme_level:
            self._state.vix_level = VIXLevel.EXTREME
            self._state.exposure_scale = 0.0  # no trading
        elif india_vix >= self.vix_reduce_level:
            self._state.vix_level = VIXLevel.HIGH
            self._state.exposure_scale = 0.5
        elif india_vix >= self.vix_warning_level:
            self._state.vix_level = VIXLevel.ELEVATED
            self._state.exposure_scale = 0.75
        else:
            self._state.vix_level = VIXLevel.NORMAL
            # Don't override recovery-phase scaling
            if self._state.recovery_phase == RecoveryPhase.NORMAL:
                self._state.exposure_scale = 1.0

        return self._state

    def record_equity(self, equity: float) -> None:
        """Record equity snapshot for rapid drawdown detection."""
        self._equity_snapshots.append((_time.time(), equity))

    def check_rapid_drawdown(self, current_equity: float) -> bool:
        """
        Check for rapid drawdown (equity drop > threshold in window).

        Returns True if rapid drawdown detected.
        """
        if not self._equity_snapshots:
            return False

        now = _time.time()
        window_start = now - (self.rapid_drawdown_window_min * 60)

        # Find peak equity in the window
        peak_in_window = current_equity
        for ts, eq in self._equity_snapshots:
            if ts >= window_start:
                peak_in_window = max(peak_in_window, eq)

        if peak_in_window <= 0:
            return False

        drawdown_pct = ((peak_in_window - current_equity) / peak_in_window) * 100

        if drawdown_pct >= self.rapid_drawdown_pct:
            self._state.rapid_drawdown = True
            self._state.exposure_scale = min(self._state.exposure_scale, 0.25)
            logger.warning(
                "Rapid drawdown detected: %.2f%% in %d min (peak=%.0f, current=%.0f)",
                drawdown_pct, self.rapid_drawdown_window_min, peak_in_window, current_equity,
            )
            return True

        self._state.rapid_drawdown = False
        return False

    def on_circuit_trip(self) -> None:
        """Called when circuit breaker trips. Start recovery sequence."""
        self._state.last_circuit_trip_ts = _time.time()
        self._state.sessions_since_trip = 0
        self._circuit_tripped_this_session = True
        self._update_recovery_phase()
        logger.warning("Circuit trip recorded. Recovery phase: %s", self._state.recovery_phase.value)

    def on_new_session(self) -> None:
        """Called at start of each trading session. Advance recovery if needed."""
        if self._circuit_tripped_this_session:
            self._circuit_tripped_this_session = False

        if self._state.last_circuit_trip_ts is not None:
            self._state.sessions_since_trip += 1
            self._update_recovery_phase()

        # Reset rapid drawdown flag for new session
        self._state.rapid_drawdown = False
        self._equity_snapshots.clear()

    def _update_recovery_phase(self) -> None:
        """Update recovery phase based on sessions since circuit trip."""
        s = self._state.sessions_since_trip
        if s >= self.recovery_sessions:
            self._state.recovery_phase = RecoveryPhase.NORMAL
            self._state.recovery_max_exposure_pct = 100.0
            if self._state.vix_level == VIXLevel.NORMAL:
                self._state.exposure_scale = 1.0
        elif s >= 2:
            self._state.recovery_phase = RecoveryPhase.PHASE_3
            self._state.recovery_max_exposure_pct = 100.0
            # Phase 3 = full recovery confirmed: override VIX scaling to 1.0
            self._state.exposure_scale = 1.0
        elif s >= 1:
            self._state.recovery_phase = RecoveryPhase.PHASE_2
            self._state.recovery_max_exposure_pct = 50.0
            self._state.exposure_scale = min(self._state.exposure_scale, 0.5)
        else:
            self._state.recovery_phase = RecoveryPhase.PHASE_1
            self._state.recovery_max_exposure_pct = 25.0
            self._state.exposure_scale = min(self._state.exposure_scale, 0.25)

    def get_exposure_scale(self) -> float:
        """Get current exposure scaling factor (0.0 to 1.0)."""
        return max(0.0, min(1.0, self._state.exposure_scale))

    def should_block_new_positions(self) -> Tuple[bool, str]:
        """Check if new positions should be blocked."""
        if self._circuit_tripped_this_session:
            return True, "Circuit breaker tripped this session — no new positions"
        if self._state.vix_level == VIXLevel.EXTREME:
            return True, f"India VIX extreme ({self._state.vix_value:.1f}) — no new positions"
        if self._state.rapid_drawdown:
            return True, "Rapid drawdown detected — no new positions"
        return False, ""
