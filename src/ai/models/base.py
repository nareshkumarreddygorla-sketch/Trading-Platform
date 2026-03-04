"""Base contract for ML predictors: directional prob, expected return, confidence."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PredictionOutput:
    """Standard output for every model in the ensemble."""
    prob_up: float  # P(price up)
    expected_return: float
    confidence: float  # 0-1
    model_id: str
    version: str
    metadata: Dict[str, Any]


class BasePredictor(ABC):
    """All ML models (XGBoost, LSTM, Transformer, RL, Vol) implement this."""

    model_id: str
    version: str

    @abstractmethod
    def predict(self, features: Dict[str, float], context: Optional[Dict[str, Any]] = None) -> PredictionOutput:
        """Return prob_up, expected_return, confidence."""
        ...

    def calibrate(self, X: np.ndarray, y: np.ndarray) -> None:
        """Optional: calibrate probabilities (e.g. isotonic)."""
        pass
