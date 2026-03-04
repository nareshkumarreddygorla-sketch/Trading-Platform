"""XGBoost classifier for directional probability. Calibrated with walk-forward."""
from typing import Any, Dict, Optional

import numpy as np

from .base import BasePredictor, PredictionOutput


class XGBPredictor(BasePredictor):
    """XGBoost binary classifier: P(up). Load from artifact in production."""

    model_id = "xgb_direction"
    version = "v1"

    def __init__(self, model=None):
        self._model = model  # xgb.Booster or sklearn API
        self.path = ""

    def predict(self, features: Dict[str, float], context: Optional[Dict[str, Any]] = None) -> PredictionOutput:
        if self._model is None:
            return PredictionOutput(
                prob_up=0.5,
                expected_return=0.0,
                confidence=0.0,
                model_id=self.model_id,
                version=self.version,
                metadata={"reason": "not_loaded"},
            )
        try:
            import xgboost as xgb
        except ImportError:
            return PredictionOutput(0.5, 0.0, 0.0, self.model_id, self.version, {"reason": "xgboost_not_installed"})

        # Build feature vector in fixed order from config
        names = sorted(features.keys())
        X = np.array([[features.get(n, 0) for n in names]], dtype=np.float32)
        dmat = xgb.DMatrix(X, feature_names=names)
        prob = float(self._model.predict(dmat)[0])
        if prob < 0 or prob > 1:
            prob = 0.5
        # Expected return / confidence from model if available (e.g. custom objective)
        return PredictionOutput(
            prob_up=prob,
            expected_return=0.0,
            confidence=abs(prob - 0.5) * 2,
            model_id=self.model_id,
            version=self.version,
            metadata={"feature_count": len(names)},
        )
