"""
Phase 4: Regime specialists.
Registry: trend, mean_reversion, high_vol_breakout, low_liquidity_defensive.
Activate by regime_id; blend outputs with regime weights.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class SpecialistOutput:
    prob_up: float
    expected_return: float
    confidence: float
    specialist_id: str


@dataclass
class RegimeSpecialist:
    specialist_id: str
    regime_ids: List[int]  # regimes where this specialist is active
    predict_fn: Callable[[Dict[str, float], Any], SpecialistOutput]


class RegimeSpecialistRegistry:
    """
    Register specialists; get_active_models(regime_id) -> list of specialists;
    blend(predictions, regime_weights) -> combined prob_up, E[r], confidence.
    """

    def __init__(self):
        self._specialists: Dict[str, RegimeSpecialist] = {}
        self._regime_weights: Dict[int, Dict[str, float]] = {}  # regime_id -> specialist_id -> weight

    def register(self, specialist: RegimeSpecialist) -> None:
        self._specialists[specialist.specialist_id] = specialist

    def set_regime_weights(self, regime_id: int, weights: Dict[str, float]) -> None:
        """Weights for blending specialists in this regime (e.g. trend=0.7, mr=0.3)."""
        self._regime_weights[regime_id] = dict(weights)

    def get_active(self, regime_id: int) -> List[RegimeSpecialist]:
        """Return specialists active in this regime."""
        out = []
        for s in self._specialists.values():
            if regime_id in s.regime_ids:
                out.append(s)
        return out

    def predict_blend(
        self,
        regime_id: int,
        features: Dict[str, float],
        context: Optional[Any] = None,
    ) -> SpecialistOutput:
        """
        Get active specialists; run predict; blend with regime weights.
        Returns combined prob_up, expected_return, confidence.
        """
        active = self.get_active(regime_id)
        if not active:
            return SpecialistOutput(0.5, 0.0, 0.0, "blend_empty")
        weights = self._regime_weights.get(regime_id, {})
        if not weights:
            weights = {s.specialist_id: 1.0 / len(active) for s in active}
        total_w = sum(weights.get(s.specialist_id, 1.0 / len(active)) for s in active) or 1.0
        prob_up = 0.0
        exp_ret = 0.0
        conf = 0.0
        for s in active:
            w = weights.get(s.specialist_id, 1.0 / len(active)) / total_w
            out = s.predict_fn(features, context)
            prob_up += out.prob_up * w
            exp_ret += out.expected_return * w
            conf += out.confidence * w
        return SpecialistOutput(
            prob_up=float(np.clip(prob_up, 0.0, 1.0)),
            expected_return=exp_ret,
            confidence=float(np.clip(conf, 0.0, 1.0)),
            specialist_id="blend",
        )
