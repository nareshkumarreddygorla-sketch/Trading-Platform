"""Hidden Markov Model for regime detection. Optional dependency: hmmlearn."""
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class HMMRegimeDetector:
    """
    Fit HMM on returns (and optionally volatility); return state index per observation.
    States can be mapped to RegimeLabel (e.g. 0=low vol, 1=mid, 2=high vol).
    """

    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self._model = None
        self._state_map: List[str] = []  # state index -> label hint

    def fit(self, returns: np.ndarray, volatility: Optional[np.ndarray] = None) -> None:
        """Fit HMM on returns. Optionally include vol as second feature."""
        try:
            from hmmlearn import hmm
        except ImportError:
            logger.warning("hmmlearn not installed; HMMRegimeDetector no-op")
            return
        X = returns.reshape(-1, 1)
        if volatility is not None and len(volatility) == len(returns):
            X = np.column_stack([returns, volatility])
        self._model = hmm.GaussianHMM(n_components=self.n_states, covariance_type="full", n_iter=100)
        self._model.fit(X)

    def predict_state(self, returns: np.ndarray, volatility: Optional[np.ndarray] = None) -> Optional[int]:
        """Return most likely state for last observation."""
        if self._model is None:
            return None
        X = returns[-1:].reshape(1, -1)
        if volatility is not None and len(volatility) >= 1:
            X = np.column_stack([X, volatility[-1:]])
        try:
            states = self._model.predict(X)
            return int(states[0])
        except Exception:
            return None
