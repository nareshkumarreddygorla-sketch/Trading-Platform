"""Base contract for ML predictors: directional prob, expected return, confidence."""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

EXPECTED_RETURN_SCALE = 0.02  # Max expected return = +/- 1%

# Minimum confidence threshold for producing a tradable signal.
# Below this, predict() should return None to halt trading on this symbol.
MIN_CONFIDENCE_THRESHOLD = 0.55  # Production: require meaningful edge (was 0.15)


@dataclass
class PredictionOutput:
    """Standard output for every model in the ensemble."""

    prob_up: float  # P(price up)
    expected_return: float
    confidence: float  # 0-1
    model_id: str
    version: str
    metadata: dict[str, Any] = field(default_factory=dict)
    symbol: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        """Validate bounds on construction to prevent garbage propagation."""
        if not math.isfinite(self.prob_up) or self.prob_up < 0 or self.prob_up > 1:
            logger.warning(
                "PredictionOutput(%s): invalid prob_up=%.6f clamped to 0.5",
                self.model_id,
                self.prob_up
                if isinstance(self.prob_up, (int, float)) and math.isfinite(self.prob_up)
                else float("nan"),
            )
            self.prob_up = 0.5
        if not math.isfinite(self.expected_return):
            logger.warning("PredictionOutput(%s): non-finite expected_return clamped to 0.0", self.model_id)
            self.expected_return = 0.0
        if not math.isfinite(self.confidence) or self.confidence < 0:
            self.confidence = 0.0
        if self.confidence > 1:
            self.confidence = 1.0


def estimate_empirical_return(
    prob_up: float,
    historical_returns: np.ndarray | None = None,
    n_bins: int = 10,
) -> float | None:
    """
    Estimate expected return empirically by binning predictions vs actual
    realized returns, instead of using the fabricated formula
    ``(prob - 0.5) * EXPECTED_RETURN_SCALE``.

    If historical_returns is not provided (no calibration data available),
    falls back to the linear formula but applies a conservative discount.

    Args:
        prob_up: Model's predicted probability of upward move (0-1).
        historical_returns: Array of shape (N, 2) where column 0 is
            predicted prob_up and column 1 is realized return. Used to
            build empirical return bins.
        n_bins: Number of probability bins for empirical estimation.

    Returns:
        Estimated expected return, or None if the prediction is unreliable
        (e.g., prob too close to 0.5 indicating no edge).
    """
    # Halt if probability is too close to 0.5 (no directional edge)
    if abs(prob_up - 0.5) < 0.05:  # Require 5% directional edge (was 2%)
        return None

    if historical_returns is not None and len(historical_returns) >= 100:
        try:
            preds = historical_returns[:, 0]
            returns = historical_returns[:, 1]

            # Build bins from historical predictions
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
            bin_idx = np.digitize(prob_up, bin_edges) - 1
            bin_idx = max(0, min(bin_idx, n_bins - 1))

            # Find samples in the same bin
            bin_mask = (np.digitize(preds, bin_edges) - 1) == bin_idx
            bin_returns = returns[bin_mask]

            if len(bin_returns) >= 10:
                empirical_return = float(np.mean(bin_returns))
                # Sanity check: if direction disagrees with probability, return None
                if (prob_up > 0.5 and empirical_return < -0.005) or (prob_up < 0.5 and empirical_return > 0.005):
                    logger.warning(
                        "Empirical return (%.4f) contradicts prob_up (%.3f) — halting",
                        empirical_return,
                        prob_up,
                    )
                    return None
                return empirical_return
        except Exception as e:
            logger.debug("Empirical return estimation failed: %s, using fallback", e)

    # Fallback: conservative linear estimate with 50% discount factor
    # This is intentionally more conservative than (prob - 0.5) * EXPECTED_RETURN_SCALE
    discount = 0.5
    return (prob_up - 0.5) * EXPECTED_RETURN_SCALE * discount


class BasePredictor(ABC):
    """All ML models (XGBoost, LSTM, Transformer, RL, Vol) implement this."""

    model_id: str
    version: str

    # Historical prediction/return pairs for empirical return estimation.
    # Subclasses can populate this during calibration or from stored data.
    _empirical_returns: np.ndarray | None = None

    @abstractmethod
    def predict(self, features: dict[str, float], context: dict[str, Any] | None = None) -> PredictionOutput | None:
        """Return prob_up, expected_return, confidence, or None to halt trading.

        Returning None signals that the model does not have sufficient
        confidence to produce a tradable prediction. The ensemble and
        trading loop must treat None as 'no opinion / halt'.
        """
        ...

    def validate_prediction(self, prob_up: float, confidence: float) -> bool:
        """Check if a prediction meets minimum quality standards for trading.

        Returns False if the prediction should be suppressed (return None
        from predict()). This prevents silently trading on garbage signals.
        """
        # Reject predictions with near-zero directional edge
        if abs(prob_up - 0.5) < 0.05:  # Require 5% directional edge (was 2%)
            logger.debug(
                "%s: prediction prob_up=%.4f too close to 0.5, halting",
                self.model_id,
                prob_up,
            )
            return False

        # Reject predictions with confidence below threshold
        if confidence < MIN_CONFIDENCE_THRESHOLD:
            logger.debug(
                "%s: confidence=%.4f below threshold %.4f, halting",
                self.model_id,
                confidence,
                MIN_CONFIDENCE_THRESHOLD,
            )
            return False

        return True

    def batch_predict(
        self, feature_batch: list[dict[str, float]], contexts: list[dict[str, Any]] | None = None
    ) -> list[PredictionOutput | None]:
        """Batch prediction for multiple symbols. Default: sequential predict() calls.
        Override in subclasses for GPU-batched inference.
        Returns None for any symbol where the model declines to trade."""
        results = []
        for i, features in enumerate(feature_batch):
            ctx = contexts[i] if contexts and i < len(contexts) else None
            results.append(self.predict(features, ctx))
        return results

    def calibrate(self, X: np.ndarray, y: np.ndarray) -> None:
        """Optional: calibrate probabilities (e.g. isotonic)."""
        logger.warning(
            "%s.calibrate() called but not implemented — probabilities remain uncalibrated",
            self.__class__.__name__,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model_id={self.model_id!r} version={self.version!r} ready={self.is_ready}>"

    @property
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready for inference.
        Override in subclasses with model-specific readiness checks."""
        return True
