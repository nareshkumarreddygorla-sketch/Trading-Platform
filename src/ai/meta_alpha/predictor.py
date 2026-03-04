"""
Meta-model: predicts P(primary wrong), P(confidence inflated), P(regime flip).
Inputs: primary prediction, confidence, recent errors, recent calibration error, regime state.
Trained on historical model errors, confidence vs realized hit rate, regime transitions.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


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
        # Model-based prediction
        try:
            feature_vector = np.array([
                float(primary_pred),
                float(confidence),
                float(np.mean(self._recent_errors)) if self._recent_errors else 0.5,
                float(np.mean(self._recent_conf)) if self._recent_conf else 0.0,
                float(regime_id) if regime_id is not None else -1.0,
                float(vol),
            ]).reshape(1, -1)

            if hasattr(self._model, "predict_proba"):
                preds = self._model.predict_proba(feature_vector)[0]
                prob_wrong = float(preds[0]) if len(preds) > 0 else 0.5
                prob_inflated = float(preds[1]) if len(preds) > 1 else 0.0
                prob_flip = float(preds[2]) if len(preds) > 2 else 0.1
            elif hasattr(self._model, "predict"):
                raw = self._model.predict(feature_vector)[0]
                if hasattr(raw, "__len__") and len(raw) >= 3:
                    prob_wrong, prob_inflated, prob_flip = float(raw[0]), float(raw[1]), float(raw[2])
                else:
                    prob_wrong = float(raw) if not hasattr(raw, "__len__") else float(raw[0])
                    prob_inflated = 0.0
                    prob_flip = 0.1
            else:
                logger.warning("Meta-alpha model has no predict/predict_proba method; using heuristic")
                self._model = None
                return self.predict(primary_pred, confidence, regime_id, vol, features)

            # Clamp to [0, 1]
            prob_wrong = max(0.0, min(1.0, prob_wrong))
            prob_inflated = max(0.0, min(1.0, prob_inflated))
            prob_flip = max(0.0, min(1.0, prob_flip))

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
        except Exception as e:
            logger.warning("Meta-alpha model prediction failed, falling back to heuristic: %s", e)
            model_backup = self._model
            self._model = None
            result = self.predict(primary_pred, confidence, regime_id, vol, features)
            self._model = model_backup
            return result
