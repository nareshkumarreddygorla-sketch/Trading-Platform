"""
Meta-model: predicts P(primary wrong), P(confidence inflated), P(regime flip).
Inputs: primary prediction, confidence, recent errors, recent calibration error, regime state.
Trained on historical model errors, confidence vs realized hit rate, regime transitions.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MetaAlphaOutput:
    prob_primary_wrong: float
    prob_confidence_inflated: float
    prob_regime_flip: float
    recommendation: str  # "reduce_size" | "hold" | "filter_signal"


class MetaAlphaPredictor:
    """
    Lightweight meta-model. In production: train on rolling windows;
    features = [primary_pred, confidence, recent_error_rate, recent_calibration_error, regime_id, vol];
    targets = [primary_was_wrong, confidence_inflated, regime_flipped_next].
    """

    def __init__(self, model=None):
        self._model = model
        self._recent_errors: List[float] = []
        self._recent_conf: List[float] = []
        self._window = 50

    def update(self, primary_correct: bool, confidence: float, realized_hit: float) -> None:
        """Feed outcome for online tracking."""
        self._recent_errors.append(0.0 if primary_correct else 1.0)
        self._recent_conf.append(abs(confidence - realized_hit))
        if len(self._recent_errors) > self._window:
            self._recent_errors = self._recent_errors[-self._window :]
            self._recent_conf = self._recent_conf[-self._window :]

    def predict(
        self,
        primary_pred: int,
        confidence: float,
        regime_id: Optional[int] = None,
        vol: float = 0.01,
        features: Optional[Dict[str, float]] = None,
    ) -> MetaAlphaOutput:
        """
        Return P(primary wrong), P(confidence inflated), P(regime flip).
        If model not loaded, use heuristics: high recent error rate => prob_primary_wrong high.
        """
        if self._model is None:
            # Heuristic
            err_rate = np.mean(self._recent_errors) if self._recent_errors else 0.5
            cal_err = np.mean(self._recent_conf) if self._recent_conf else 0.0
            prob_wrong = min(1.0, err_rate + 0.2)
            prob_inflated = min(1.0, cal_err * 2)
            prob_flip = 0.1
            if prob_wrong > 0.6 or prob_inflated > 0.5:
                rec = "reduce_size"
            elif prob_wrong > 0.5:
                rec = "filter_signal"
            else:
                rec = "hold"
            return MetaAlphaOutput(
                prob_primary_wrong=prob_wrong,
                prob_confidence_inflated=prob_inflated,
                prob_regime_flip=prob_flip,
                recommendation=rec,
            )
        # TODO: call self._model.predict(feature_vector)
        return MetaAlphaOutput(0.5, 0.0, 0.1, "hold")
