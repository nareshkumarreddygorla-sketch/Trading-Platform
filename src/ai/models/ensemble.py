"""
Multi-model ensemble: XGBoost, LSTM, RL, Sentiment.
Outputs weighted prob_up, expected_return, confidence; calibrated via walk-forward.
Supports dynamic IC-weighted ensemble: per-model rolling Information Coefficient
drives automatic weight recalculation.

Calibration: supports Platt scaling or isotonic regression for prob_up calibration.
Calibrator is retrained weekly alongside model retrain.
"""
import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BasePredictor, PredictionOutput
from .registry import ModelRegistry

logger = logging.getLogger(__name__)

# Default model list matching actual model_id attributes in predictor classes
DEFAULT_MODEL_IDS: List[str] = ["xgboost_alpha", "lstm_ts", "rl_ppo", "sentiment_finbert"]

# IC calculation parameters
IC_ROLLING_WINDOW: int = 20          # Rolling window size for IC calculation
MIN_WEIGHT_FLOOR: float = 0.05       # 5% minimum weight for any non-zero IC model
NEGATIVE_IC_STREAK_LIMIT: int = 20   # Consecutive negative IC days to zero-out weight


class EnsembleEngine:
    """
    Aggregates predictions from multiple models with configurable weights.
    Weights can be fixed, come from meta-allocator (confidence-weighted),
    or be dynamically recalculated from rolling per-model IC scores.

    IC-weighted mode:
      - Track per-model Information Coefficient (rolling 20-day Pearson
        correlation of predicted signal vs actual return).
      - Daily recalculation: weight_i = max(0, IC_i) / sum(max(0, IC_j)).
      - Minimum weight floor of 5% for any model with positive IC.
      - Model with negative IC for 20 consecutive days gets weight = 0.
      - Full weight history is stored for audit.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        model_ids: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        calibrator: Optional[Any] = None,
        ic_weighted: bool = False,
    ):
        self.registry = registry
        self.model_ids = model_ids or list(DEFAULT_MODEL_IDS)
        self.weights = weights or {}  # model_id -> weight; if missing, equal weight
        self.calibrator = calibrator  # Platt or Isotonic; applied to prob_up after aggregation
        self.ic_weighted = ic_weighted

        # Try to load persisted IC weights from Redis (Sprint 8.6)
        if not self.weights or not any(v > 0 for v in self.weights.values()):
            try:
                import redis, json, os
                r = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"), socket_connect_timeout=2)
                saved = r.get("ensemble:ic_weights")
                if saved:
                    loaded = json.loads(saved)
                    if isinstance(loaded, dict) and loaded:
                        self.weights = loaded
                        logger.info("Loaded IC weights from Redis: %s", self.weights)
            except Exception:
                pass  # Redis not available — use defaults

        # --- IC tracking state ---
        # Per-model, per-symbol prediction history:
        #   _prediction_history[model_id][symbol] = list of (predicted_direction, actual_return)
        self._prediction_history: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Per-model current IC score (aggregated across symbols)
        self._ic_scores: Dict[str, float] = {}

        # Per-model count of consecutive days with negative IC
        self._negative_ic_streak: Dict[str, int] = defaultdict(int)

        # Audit trail: list of (timestamp, weights_snapshot) dicts
        self._weight_history: List[Dict[str, Any]] = []

    def set_weights(self, weights: Dict[str, float]) -> None:
        self.weights = weights

    # ------------------------------------------------------------------
    # IC-weighted ensemble methods
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        model_id: str,
        symbol: str,
        predicted_direction: float,
        actual_return: float,
    ) -> None:
        """
        Record a single prediction/actual pair for IC tracking.

        Args:
            model_id: Identifier of the model (e.g. "xgboost").
            symbol: Ticker symbol (e.g. "AAPL").
            predicted_direction: Model's predicted signal.  Positive means
                predicted up, negative means predicted down.  Magnitude may
                reflect conviction.
            actual_return: Realised return for the same period.
        """
        self._prediction_history[model_id][symbol].append(
            (float(predicted_direction), float(actual_return))
        )
        # Keep only the most recent IC_ROLLING_WINDOW entries per symbol
        # to bound memory usage.
        hist = self._prediction_history[model_id][symbol]
        if len(hist) > IC_ROLLING_WINDOW:
            self._prediction_history[model_id][symbol] = hist[-IC_ROLLING_WINDOW:]

    def _compute_model_ic(self, model_id: str) -> Optional[float]:
        """
        Compute the rolling IC for *model_id* as the average Pearson
        correlation (across all tracked symbols) of the most recent
        ``IC_ROLLING_WINDOW`` predicted-direction vs actual-return pairs.

        Returns ``None`` if insufficient data (fewer than 2 data points
        across all symbols).
        """
        symbol_histories = self._prediction_history.get(model_id, {})
        if not symbol_histories:
            return None

        per_symbol_ics: List[float] = []
        for _symbol, pairs in symbol_histories.items():
            recent = pairs[-IC_ROLLING_WINDOW:]
            if len(recent) < 2:
                continue
            preds = np.array([p for p, _a in recent], dtype=np.float64)
            actuals = np.array([a for _p, a in recent], dtype=np.float64)

            # Guard against zero-variance arrays (correlation undefined)
            if np.std(preds) < 1e-12 or np.std(actuals) < 1e-12:
                per_symbol_ics.append(0.0)
                continue

            corr = np.corrcoef(preds, actuals)[0, 1]
            if not math.isfinite(corr):
                corr = 0.0
            per_symbol_ics.append(float(corr))

        if not per_symbol_ics:
            return None
        return float(np.mean(per_symbol_ics))

    def update_weights_from_ic(self, recent_predictions=None, actual_returns=None, **kwargs) -> Dict[str, float]:
        """
        Recalculate ensemble weights based on rolling IC scores.

        Algorithm:
            1. Compute IC_i for each model.
            2. Models with negative IC for >= ``NEGATIVE_IC_STREAK_LIMIT``
               consecutive updates get weight 0.
            3. raw_weight_i = max(0, IC_i).
            4. Normalise so weights sum to 1.
            5. Apply minimum weight floor of ``MIN_WEIGHT_FLOOR`` for any
               model whose raw weight is > 0.
            6. Re-normalise after floor adjustment.

        Returns:
            The newly computed weight dict ``{model_id: weight}``.
        """
        # Step 1: compute IC per model
        for model_id in self.model_ids:
            ic = self._compute_model_ic(model_id)
            if ic is not None:
                self._ic_scores[model_id] = ic
            # If ic is None (not enough data), keep the previous score or
            # leave it absent so the model falls back to equal weight.

        # Step 2: update negative-IC streak counters
        for model_id in self.model_ids:
            ic = self._ic_scores.get(model_id)
            if ic is not None and ic < 0:
                self._negative_ic_streak[model_id] += 1
            else:
                self._negative_ic_streak[model_id] = 0

        # Step 3: compute raw weights
        raw: Dict[str, float] = {}
        for model_id in self.model_ids:
            # Zero weight if negative IC streak hit limit
            if self._negative_ic_streak.get(model_id, 0) >= NEGATIVE_IC_STREAK_LIMIT:
                raw[model_id] = 0.0
                logger.info(
                    "Model %s zeroed out: negative IC for %d consecutive updates",
                    model_id,
                    NEGATIVE_IC_STREAK_LIMIT,
                )
                continue

            ic = self._ic_scores.get(model_id)
            if ic is None:
                # No IC data yet; give equal share among unscored models
                raw[model_id] = 0.0
            else:
                raw[model_id] = max(0.0, ic)

        # If all raw weights are zero (all models negative or no data),
        # fall back to equal weights so the ensemble still produces output.
        total_raw = sum(raw.values())
        if total_raw < 1e-12:
            n = len(self.model_ids) or 1
            new_weights = {mid: 1.0 / n for mid in self.model_ids}
        else:
            # Step 4: normalise
            new_weights = {mid: raw[mid] / total_raw for mid in self.model_ids}

            # Step 5: apply minimum weight floor for positive-weight models
            positive_ids = [mid for mid in self.model_ids if new_weights[mid] > 0]
            if positive_ids:
                for mid in positive_ids:
                    if new_weights[mid] < MIN_WEIGHT_FLOOR:
                        new_weights[mid] = MIN_WEIGHT_FLOOR

                # Step 6: re-normalise after floor adjustment
                total_adj = sum(new_weights.values())
                if total_adj > 1e-12:
                    new_weights = {mid: w / total_adj for mid, w in new_weights.items()}

        self.weights = new_weights

        # Record to audit trail
        self._weight_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "weights": dict(new_weights),
            "ic_scores": dict(self._ic_scores),
            "negative_ic_streaks": dict(self._negative_ic_streak),
        })

        logger.info("IC-weighted ensemble weights updated: %s (IC scores: %s)", new_weights, dict(self._ic_scores))

        # Persist to Redis (Sprint 8.6)
        try:
            import redis, json, os
            r = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"), socket_connect_timeout=2)
            r.set("ensemble:ic_weights", json.dumps(new_weights), ex=86400 * 30)
            r.set("ensemble:ic_scores", json.dumps({k: v for k, v in self._ic_scores.items()}), ex=86400 * 30)
        except Exception:
            logger.debug("IC weight persistence to Redis failed (non-critical)")

        # Return IC scores dict (used by scheduler for IC degradation detection)
        return dict(self._ic_scores) if self._ic_scores else dict(new_weights)

    def get_weight_history(self) -> List[Dict[str, Any]]:
        """
        Return the full weight audit trail.

        Each entry is a dict with keys:
            - ``timestamp``: ISO-8601 UTC string of when weights were updated.
            - ``weights``: ``{model_id: weight}`` snapshot.
            - ``ic_scores``: ``{model_id: IC}`` at that point.
            - ``negative_ic_streaks``: ``{model_id: streak_count}``.
        """
        return list(self._weight_history)

    def get_model_ic_scores(self) -> Dict[str, Optional[float]]:
        """
        Return the current IC score for every tracked model.

        Returns:
            ``{model_id: ic_score}`` where *ic_score* is ``None`` if
            insufficient data has been recorded.
        """
        result: Dict[str, Optional[float]] = {}
        for model_id in self.model_ids:
            result[model_id] = self._ic_scores.get(model_id)
        return result

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def fit_calibrator(
        self,
        raw_probs: List[float],
        actual_outcomes: List[int],
        method: str = "isotonic",
    ) -> bool:
        """
        Fit a calibration model on historical raw ensemble prob_up vs actual outcomes.

        Args:
            raw_probs: List of raw ensemble prob_up values (0-1).
            actual_outcomes: List of binary outcomes (1=up, 0=down).
            method: "isotonic" (default) or "platt" (logistic regression).

        Returns:
            True if calibrator was successfully fitted.
        """
        if len(raw_probs) < 20:
            logger.warning("Calibrator fit: insufficient data (%d < 20)", len(raw_probs))
            return False
        try:
            from sklearn.isotonic import IsotonicRegression
            from sklearn.linear_model import LogisticRegression

            X = np.array(raw_probs).reshape(-1, 1)
            y = np.array(actual_outcomes)

            if method == "platt":
                lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
                lr.fit(X, y)
                # Wrap in a transform-compatible object
                class PlattCalibrator:
                    def __init__(self, model):
                        self._model = model
                    def transform(self, x):
                        return self._model.predict_proba(np.asarray(x).reshape(-1, 1))[:, 1]
                self.calibrator = PlattCalibrator(lr)
            else:  # isotonic
                ir = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
                ir.fit(np.array(raw_probs), y)
                self.calibrator = ir

            logger.info("Calibrator fitted: method=%s, n_samples=%d", method, len(raw_probs))
            self._weight_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "calibrator_fitted",
                "method": method,
                "n_samples": len(raw_probs),
            })
            return True
        except ImportError:
            logger.warning("sklearn not installed — calibrator not fitted")
            return False
        except Exception as e:
            logger.exception("Calibrator fit failed: %s", e)
            return False

    def update_ic_weights(
        self,
        recent_predictions: List[Dict[str, Any]],
        actual_returns: List[float],
    ) -> Dict[str, float]:
        """
        Convenience method: record prediction/actual pairs for all models,
        then recalculate IC-based weights.

        Args:
            recent_predictions: List of dicts with keys:
                {model_id, symbol, predicted_direction}
            actual_returns: Corresponding actual returns (same length).

        Returns:
            Updated weight dict.
        """
        for pred, actual_ret in zip(recent_predictions, actual_returns):
            model_id = pred.get("model_id", "")
            symbol = pred.get("symbol", "unknown")
            direction = pred.get("predicted_direction", 0.0)
            if model_id:
                self.record_prediction(model_id, symbol, direction, actual_ret)

        return self.update_weights_from_ic()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        features: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> PredictionOutput:
        """Aggregate predictions from all registered models."""
        predictions: List[PredictionOutput] = []
        ids = self.model_ids or list(self.weights.keys())
        if not ids:
            ids = list(self.registry._models)
        for model_id in ids:
            model = self.registry.get(model_id)
            if model is None:
                continue
            try:
                out = model.predict(features, context)
                predictions.append(out)
            except Exception as e:
                logger.exception("Model %s predict failed: %s", model_id, e)

        if not predictions:
            return PredictionOutput(
                prob_up=0.5,
                expected_return=0.0,
                confidence=0.0,
                model_id="ensemble",
                version="v1",
                metadata={"count": 0},
            )

        total_weight = 0.0
        prob_up = 0.0
        exp_ret = 0.0
        conf = 0.0
        for p in predictions:
            w = self.weights.get(p.model_id, 1.0)
            total_weight += w
            prob_up += p.prob_up * w
            exp_ret += p.expected_return * w
            conf += p.confidence * w
        if total_weight < 1e-12:
            total_weight = 1.0
        prob_up_f = prob_up / total_weight
        exp_ret_f = exp_ret / total_weight
        conf_f = min(1.0, conf / total_weight)
        # Sanitize NaN/Inf: avoid propagating to position sizing or signals
        if not math.isfinite(prob_up_f):
            prob_up_f = 0.5
        if not math.isfinite(exp_ret_f):
            exp_ret_f = 0.0
        if not math.isfinite(conf_f) or conf_f < 0:
            conf_f = 0.0
        # Calibration: apply Platt/Isotonic if fitted
        if self.calibrator is not None and hasattr(self.calibrator, "transform"):
            try:
                prob_up_f = float(np.asarray(self.calibrator.transform(np.array([prob_up_f]))).flat[0])
            except Exception:
                pass
        return PredictionOutput(
            prob_up=float(np.clip(prob_up_f, 0.0, 1.0)),
            expected_return=exp_ret_f,
            confidence=conf_f,
            model_id="ensemble",
            version="v1",
            metadata={"count": len(predictions), "models": [p.model_id for p in predictions]},
        )
