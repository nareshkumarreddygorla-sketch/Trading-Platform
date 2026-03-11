"""XGBoost classifier for directional probability. Calibrated with walk-forward."""
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BasePredictor, PredictionOutput, estimate_empirical_return

logger = logging.getLogger(__name__)

# Path to the meta file that stores the feature names used during training
_META_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "models",
    "alpha_xgb_meta.json",
)


class FeatureMetaMissingError(RuntimeError):
    """Raised when the XGBoost feature meta file is missing or corrupt.

    This is intentionally a hard error: silently falling back to
    sorted(features.keys()) causes feature mismatch and garbage predictions
    that are impossible to detect downstream.
    """


def _load_feature_names_from_meta(meta_path: str = _META_PATH) -> Optional[List[str]]:
    """Load saved feature_names from the XGB meta JSON file.

    Returns the list of feature names, or None if the meta file does not exist.
    Raises FeatureMetaMissingError if the file exists but is corrupt.
    """
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        names = meta.get("feature_names")
        if isinstance(names, list) and len(names) > 0:
            return list(names)
        raise FeatureMetaMissingError(
            f"XGB meta file {meta_path} exists but 'feature_names' is empty or invalid"
        )
    except json.JSONDecodeError as e:
        raise FeatureMetaMissingError(
            f"XGB meta file {meta_path} is corrupt (invalid JSON): {e}"
        ) from e


class XGBPredictor(BasePredictor):
    """XGBoost binary classifier: P(up). Load from artifact in production."""

    model_id = "xgb_direction"
    version = "v1"

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def __repr__(self) -> str:
        return f"<XGBPredictor model_id={self.model_id!r} version={self.version!r} loaded={self._model is not None}>"

    def __init__(self, model=None):
        self._model = model  # xgb.Booster or sklearn API
        self.path = ""
        # Load feature names from meta file at init time so predict() uses
        # the EXACT order the model was trained with.
        # If meta is missing AND a model is loaded, this is a hard error:
        # we cannot safely predict without knowing the training feature order.
        loaded_names = _load_feature_names_from_meta()
        if loaded_names is None and model is not None:
            raise FeatureMetaMissingError(
                f"XGBoost model provided but feature meta file not found at {_META_PATH}. "
                "Cannot predict without knowing training feature order. "
                "Re-train or restore the meta file."
            )
        self._feature_names: List[str] = loaded_names if loaded_names is not None else []
        if self._feature_names:
            logger.info("XGBPredictor: loaded %d feature names from meta file", len(self._feature_names))
        self._feature_importance: Dict[str, float] = {}
        self._prediction_count: int = 0

        # P1-7: Load empirical returns calibration data if available
        _calibration_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "models", "xgb_calibration.npz",
        )
        if os.path.exists(_calibration_path):
            try:
                cal_data = np.load(_calibration_path)
                if "predictions" in cal_data and "returns" in cal_data:
                    preds = cal_data["predictions"]
                    rets = cal_data["returns"]
                    if len(preds) >= 100 and len(preds) == len(rets):
                        self._empirical_returns = np.column_stack([preds, rets])
                        logger.info(
                            "XGBPredictor: loaded %d empirical return calibration samples",
                            len(preds),
                        )
            except Exception as e:
                logger.warning("XGBPredictor: calibration data load failed: %s", e)

    def predict(self, features: Dict[str, float], context: Optional[Dict[str, Any]] = None) -> Optional[PredictionOutput]:
        if self._model is None:
            return None
        try:
            import xgboost as xgb
        except ImportError:
            return None

        _predict_start = time.monotonic()

        # Strict feature ordering: MUST use training meta file order.
        # Never fall back to sorted(features.keys()) — that causes silent
        # feature mismatch and garbage predictions.
        if not self._feature_names:
            raise FeatureMetaMissingError(
                "XGBPredictor: feature meta not loaded. Cannot predict without "
                "knowing the exact training feature order. Re-train or restore "
                f"the meta file at {_META_PATH}"
            )

        names: List[str] = self._feature_names

        # Feature validation: check that incoming features contain expected features
        incoming_keys = set(features.keys())
        expected_keys = set(names)
        missing_features = expected_keys - incoming_keys
        unexpected_features = incoming_keys - expected_keys

        if missing_features:
            # Warn but allow: missing features become NaN (XGBoost handles NaN natively)
            if len(missing_features) > len(names) * 0.3:
                logger.error(
                    "XGB: >30%% features missing (%d/%d). Prediction unreliable. "
                    "Missing: %s",
                    len(missing_features), len(names),
                    list(missing_features)[:10],
                )
                return None
            else:
                logger.warning(
                    "XGB: %d/%d features missing (will use NaN): %s",
                    len(missing_features), len(names),
                    list(missing_features)[:5],
                )

        if unexpected_features and self._prediction_count == 0:
            logger.info(
                "XGB: %d unexpected features in input (ignored): %s",
                len(unexpected_features), list(unexpected_features)[:5],
            )

        X = np.array([[features.get(n, np.nan) for n in names]], dtype=np.float32)
        dmat = xgb.DMatrix(X, feature_names=names)
        prob = float(self._model.predict(dmat)[0])
        if not math.isfinite(prob) or prob < 0 or prob > 1:
            prob = 0.5
            logger.warning("XGB prediction returned invalid value, defaulting to 0.5")

        # Track feature importance from model
        self._prediction_count += 1
        if self._prediction_count % 100 == 1:  # Update importance every 100 predictions
            try:
                importance = self._model.get_score(importance_type="gain")
                if importance:
                    self._feature_importance = dict(importance)
            except Exception as e:
                logger.debug("Feature importance extraction failed: %s", e)

        # Calibrated confidence: use prediction interval from model if available
        raw_confidence = abs(prob - 0.5) * 2

        # Validate prediction meets minimum quality standards
        if not self.validate_prediction(prob, raw_confidence):
            return None

        # Expected return estimate using empirical calibration
        expected_return = estimate_empirical_return(prob, self._empirical_returns)
        if expected_return is None:
            return None

        predict_latency_ms = (time.monotonic() - _predict_start) * 1000

        return PredictionOutput(
            prob_up=prob,
            expected_return=expected_return,
            confidence=raw_confidence,
            model_id=self.model_id,
            version=self.version,
            metadata={
                "feature_count": len(names),
                "missing_features": len(missing_features),
                "feature_validation": "strict",
                "top_features": dict(sorted(self._feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]) if self._feature_importance else {},
                "predict_latency_ms": predict_latency_ms,
            },
        )
