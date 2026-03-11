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
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import numpy as np

try:
    from scipy.stats import spearmanr as _spearmanr
except ImportError:
    _spearmanr = None

from .base import PredictionOutput
from .registry import ModelRegistry

logger = logging.getLogger(__name__)

# Default model list matching actual model_id attributes in predictor classes
DEFAULT_MODEL_IDS: list[str] = ["xgb_direction", "lstm_ts", "rl_ppo", "sentiment_finbert", "transformer_ts"]

# IC calculation parameters
IC_ROLLING_WINDOW: int = (
    63  # Rolling window size for IC calculation (63 ≈ 3 months trading days, statistically significant)
)
IC_THRESHOLD: float = 0.05  # Minimum IC for a model to receive any weight (raised from 0.02)
MIN_WEIGHT_FLOOR: float = 0.05  # 5% minimum weight for any non-zero IC model
NEGATIVE_IC_STREAK_LIMIT: int = 20  # Consecutive negative IC days to zero-out weight
MIN_AGREEING_MODELS: int = 3  # Minimum models that must agree on direction to produce signal (of 5)
MAX_INTER_MODEL_CORRELATION: float = 0.95  # Diversity: max pairwise correlation before warning


class EnsembleEngine:
    """
    Aggregates predictions from multiple models with configurable weights.
    Weights can be fixed, come from meta-allocator (confidence-weighted),
    or be dynamically recalculated from rolling per-model IC scores.

    IC-weighted mode:
      - Track per-model Information Coefficient (rolling 63-day
        Spearman rank correlation of predicted signal vs actual return).
      - Daily recalculation: weight_i = max(0, IC_i) / sum(max(0, IC_j)).
      - Minimum weight floor of 5% for any model with positive IC.
      - Model with negative IC for 20 consecutive days gets weight = 0.
      - Full weight history is stored for audit.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        model_ids: list[str] | None = None,
        weights: dict[str, float] | None = None,
        calibrator: Any | None = None,
        ic_weighted: bool = False,
    ):
        self.registry = registry
        self.model_ids = model_ids or list(DEFAULT_MODEL_IDS)
        self.weights = weights or {}  # model_id -> weight; if missing, equal weight
        self.calibrator = calibrator  # Platt or Isotonic; applied to prob_up after aggregation
        self.ic_weighted = ic_weighted
        self._lock = threading.Lock()
        self._redis_client = None

        # Try to load persisted IC weights from Redis (Sprint 8.6)
        if not self.weights or not any(v > 0 for v in self.weights.values()):
            try:
                import json
                import os

                import redis

                self._redis_client = redis.Redis.from_url(
                    os.environ.get("REDIS_URL", "redis://localhost:6379"), socket_connect_timeout=2
                )
                saved = self._redis_client.get("ensemble:ic_weights")
                if saved:
                    loaded = json.loads(saved)
                    if isinstance(loaded, dict) and loaded:
                        self.weights = loaded
                        logger.info("Loaded IC weights from Redis: %s", self.weights)
            except Exception as e:
                logger.warning("Failed to load IC weights from Redis: %s", e)

        # --- IC tracking state ---
        # Per-model, per-symbol prediction history:
        #   _prediction_history[model_id][symbol] = list of (predicted_direction, actual_return)
        self._prediction_history: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Per-model current IC score (aggregated across symbols)
        self._ic_scores: dict[str, float] = {}

        # Per-model count of consecutive days with negative IC
        self._negative_ic_streak: dict[str, int] = defaultdict(int)

        # Halt flag: True when all model ICs are below threshold
        self._all_models_halted: bool = False

        # Diversity tracking
        self._last_diversity_score: float = 1.0
        self._diversity_check_interval: int = 50  # Check diversity every N predictions
        self._prediction_counter: int = 0

        # Audit trail: list of (timestamp, weights_snapshot) dicts
        self._weight_history: list[dict[str, Any]] = []

    def set_weights(self, weights: dict[str, float]) -> None:
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
        with self._lock:
            self._prediction_history[model_id][symbol].append((float(predicted_direction), float(actual_return)))
            # Keep only the most recent IC_ROLLING_WINDOW entries per symbol
            # to bound memory usage.
            hist = self._prediction_history[model_id][symbol]
            if len(hist) > IC_ROLLING_WINDOW:
                self._prediction_history[model_id][symbol] = hist[-IC_ROLLING_WINDOW:]

    def _compute_model_ic(self, model_id: str) -> float | None:
        """
        Compute the rolling IC for *model_id* as the average Spearman
        rank correlation (across all tracked symbols) of the most recent
        ``IC_ROLLING_WINDOW`` predicted-direction vs actual-return pairs.

        Returns ``None`` if insufficient data (fewer than 2 data points
        across all symbols).
        """
        symbol_histories = self._prediction_history.get(model_id, {})
        if not symbol_histories:
            return None

        per_symbol_ics: list[float] = []
        for _symbol, pairs in symbol_histories.items():
            recent = pairs[-IC_ROLLING_WINDOW:]
            if len(recent) < 20:
                continue
            preds = np.array([p for p, _a in recent], dtype=np.float64)
            actuals = np.array([a for _p, a in recent], dtype=np.float64)

            # Guard against zero-variance arrays (correlation undefined)
            if np.std(preds) < 1e-12 or np.std(actuals) < 1e-12:
                per_symbol_ics.append(0.0)
                continue

            # Use Spearman rank correlation (not Pearson) for proper IC calculation
            # Rank correlation is more robust to outliers and non-linear relationships
            if _spearmanr is None:
                per_symbol_ics.append(0.0)
                continue
            corr, _pval = _spearmanr(preds, actuals)
            if not math.isfinite(corr):
                corr = 0.0
            per_symbol_ics.append(float(corr))

        if not per_symbol_ics:
            return None
        return float(np.mean(per_symbol_ics))

    def update_weights_from_ic(self, recent_predictions=None, actual_returns=None, **kwargs) -> dict[str, float]:
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
        raw: dict[str, float] = {}
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
                # No IC data yet — do not give any weight until IC is established
                raw[model_id] = 0.0
            elif ic < IC_THRESHOLD:
                # IC below institutional-grade threshold (0.05) — zero weight
                raw[model_id] = 0.0
            else:
                raw[model_id] = max(0.0, ic)

        # If all raw weights are zero (all models below IC threshold or negative),
        # return zero weights to signal halt — do NOT fall back to equal weights.
        total_raw = sum(raw.values())
        if total_raw < 1e-12:
            logger.warning(
                "ALL model ICs below threshold (%.2f) or negative — ensemble halted. IC scores: %s",
                IC_THRESHOLD,
                dict(self._ic_scores),
            )
            new_weights = {mid: 0.0 for mid in self.model_ids}
            self._all_models_halted = True
        else:
            self._all_models_halted = False
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
        self._weight_history.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "weights": dict(new_weights),
                "ic_scores": dict(self._ic_scores),
                "negative_ic_streaks": dict(self._negative_ic_streak),
            }
        )
        if len(self._weight_history) > 500:
            self._weight_history = self._weight_history[-500:]

        logger.info("IC-weighted ensemble weights updated: %s (IC scores: %s)", new_weights, dict(self._ic_scores))

        # Persist to Redis (Sprint 8.6)
        try:
            import json
            import os

            if self._redis_client is None:
                import redis

                self._redis_client = redis.Redis.from_url(
                    os.environ.get("REDIS_URL", "redis://localhost:6379"), socket_connect_timeout=2
                )
            self._redis_client.set("ensemble:ic_weights", json.dumps(new_weights), ex=86400 * 30)
            self._redis_client.set(
                "ensemble:ic_scores", json.dumps({k: v for k, v in self._ic_scores.items()}), ex=86400 * 30
            )
        except Exception as e:
            logger.warning("IC weight persistence to Redis failed (non-critical): %s", e)

        # Return IC scores dict (used by scheduler for IC degradation detection)
        return dict(new_weights)

    def get_weight_history(self) -> list[dict[str, Any]]:
        """
        Return the full weight audit trail.

        Each entry is a dict with keys:
            - ``timestamp``: ISO-8601 UTC string of when weights were updated.
            - ``weights``: ``{model_id: weight}`` snapshot.
            - ``ic_scores``: ``{model_id: IC}`` at that point.
            - ``negative_ic_streaks``: ``{model_id: streak_count}``.
        """
        return list(self._weight_history)

    def get_model_ic_scores(self) -> dict[str, float | None]:
        """
        Return the current IC score for every tracked model.

        Returns:
            ``{model_id: ic_score}`` where *ic_score* is ``None`` if
            insufficient data has been recorded.
        """
        result: dict[str, float | None] = {}
        for model_id in self.model_ids:
            result[model_id] = self._ic_scores.get(model_id)
        return result

    def check_ensemble_diversity(self) -> dict[str, Any]:
        """
        Check inter-model prediction correlation to ensure ensemble diversity.

        Returns a dict with pairwise correlations, a diversity_ok flag,
        a diversity_score (0-1 where 1 = perfect diversity), and
        weight_adjustments if correlated models need penalisation.

        Models that are too correlated (> MAX_INTER_MODEL_CORRELATION) provide
        redundant information and reduce ensemble value. When detected, the
        correlated model with the LOWER IC score gets its weight halved.
        """
        pairwise_corrs: dict[str, float] = {}
        model_ids_with_data = []

        for model_id in self.model_ids:
            histories = self._prediction_history.get(model_id, {})
            if histories:
                model_ids_with_data.append(model_id)

        if len(model_ids_with_data) < 2:
            return {
                "diversity_ok": True,
                "diversity_score": 1.0,
                "pairwise_correlations": {},
                "reason": "insufficient_models",
                "weight_adjustments": {},
            }

        # Collect all predictions per model (flattened across symbols)
        model_preds: dict[str, list[float]] = {}
        for model_id in model_ids_with_data:
            preds = []
            for _sym, pairs in self._prediction_history[model_id].items():
                preds.extend([p for p, _a in pairs[-IC_ROLLING_WINDOW:]])
            model_preds[model_id] = preds

        # Compute pairwise correlations
        high_corr_pairs = []
        all_corrs: list[float] = []
        for i, m1 in enumerate(model_ids_with_data):
            for m2 in model_ids_with_data[i + 1 :]:
                min_len = min(len(model_preds[m1]), len(model_preds[m2]))
                if min_len < 20:
                    continue
                p1 = np.array(model_preds[m1][:min_len])
                p2 = np.array(model_preds[m2][:min_len])
                if np.std(p1) < 1e-12 or np.std(p2) < 1e-12:
                    continue
                corr = float(np.corrcoef(p1, p2)[0, 1])
                if not math.isfinite(corr):
                    continue
                pair_key = f"{m1}_vs_{m2}"
                pairwise_corrs[pair_key] = round(corr, 4)
                all_corrs.append(abs(corr))
                if abs(corr) > MAX_INTER_MODEL_CORRELATION:
                    high_corr_pairs.append((m1, m2, corr))

        # Diversity score: 1 - mean(abs(pairwise correlations))
        # 1.0 = perfectly uncorrelated (ideal), 0.0 = perfectly correlated (bad)
        diversity_score = 1.0 - (float(np.mean(all_corrs)) if all_corrs else 0.0)
        diversity_score = float(np.clip(diversity_score, 0.0, 1.0))

        diversity_ok = len(high_corr_pairs) == 0
        weight_adjustments: dict[str, float] = {}

        if not diversity_ok:
            logger.warning(
                "Ensemble diversity warning: %d model pairs exceed correlation threshold %.2f: %s",
                len(high_corr_pairs),
                MAX_INTER_MODEL_CORRELATION,
                [(p[0], p[1], f"{p[2]:.3f}") for p in high_corr_pairs],
            )
            # Penalise the weaker model in each correlated pair:
            # The model with lower IC gets its weight halved
            for m1, m2, corr in high_corr_pairs:
                ic1 = self._ic_scores.get(m1, 0.0)
                ic2 = self._ic_scores.get(m2, 0.0)
                weaker = m2 if ic1 >= ic2 else m1
                penalty = 0.5  # Halve the weight of the weaker correlated model
                current_weight = self.weights.get(weaker, 0.0)
                adjusted = current_weight * penalty
                weight_adjustments[weaker] = adjusted
                logger.info(
                    "Diversity enforcement: reducing %s weight %.4f -> %.4f (correlated %.3f with %s)",
                    weaker,
                    current_weight,
                    adjusted,
                    corr,
                    m1 if weaker == m2 else m2,
                )

            # Apply weight adjustments
            if weight_adjustments:
                for model_id, new_weight in weight_adjustments.items():
                    self.weights[model_id] = new_weight
                # Re-normalise weights
                total = sum(self.weights.get(mid, 0.0) for mid in self.model_ids)
                if total > 1e-12:
                    self.weights = {mid: self.weights.get(mid, 0.0) / total for mid in self.model_ids}

        self._last_diversity_score = diversity_score
        return {
            "diversity_ok": diversity_ok,
            "diversity_score": diversity_score,
            "pairwise_correlations": pairwise_corrs,
            "high_correlation_pairs": [(p[0], p[1], p[2]) for p in high_corr_pairs],
            "weight_adjustments": weight_adjustments,
        }

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def fit_calibrator(
        self,
        raw_probs: list[float],
        actual_outcomes: list[int],
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
        if len(raw_probs) < 100:
            logger.warning("Calibrator fit: insufficient data (%d < 100)", len(raw_probs))
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
            self._weight_history.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "event": "calibrator_fitted",
                    "method": method,
                    "n_samples": len(raw_probs),
                }
            )
            if len(self._weight_history) > 500:
                self._weight_history = self._weight_history[-500:]
            return True
        except ImportError:
            logger.warning("sklearn not installed — calibrator not fitted")
            return False
        except Exception as e:
            logger.exception("Calibrator fit failed: %s", e)
            return False

    def update_ic_weights(
        self,
        recent_predictions: list[dict[str, Any]],
        actual_returns: list[float],
    ) -> dict[str, float]:
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

    def __repr__(self) -> str:
        return f"<EnsembleEngine models={self.model_ids} ic_weighted={self.ic_weighted} weights={self.weights}>"

    def predict(
        self,
        features: dict[str, float],
        context: dict[str, Any] | None = None,
    ) -> PredictionOutput | None:
        """Aggregate predictions from all registered models.

        Returns None (halt trading) when:
          - All model ICs are below threshold (halted state)
          - Fewer than MIN_AGREEING_MODELS agree on direction
          - No models produce valid predictions
        """
        _predict_start = time.monotonic()

        # Check if all models are halted due to IC degradation
        if self._all_models_halted:
            logger.warning("Ensemble halted: all model ICs below threshold %.2f", IC_THRESHOLD)
            return None

        # Periodic diversity check: enforce decorrelation before prediction
        self._prediction_counter += 1
        if self._prediction_counter % self._diversity_check_interval == 0:
            diversity_result = self.check_ensemble_diversity()
            self._last_diversity_score = diversity_result.get("diversity_score", 1.0)
            if not diversity_result.get("diversity_ok", True):
                logger.warning(
                    "Ensemble diversity score: %.3f (weights adjusted for %d correlated pairs)",
                    self._last_diversity_score,
                    len(diversity_result.get("high_correlation_pairs", [])),
                )

        predictions: list[PredictionOutput] = []
        ids = self.model_ids or list(self.weights.keys())
        if not ids:
            ids = self.registry.list_models()
        for model_id in ids:
            model = self.registry.get(model_id)
            if model is None:
                continue
            try:
                out = model.predict(features, context)
                # Models can now return None to signal halt
                if out is not None:
                    predictions.append(out)
            except Exception as e:
                logger.exception("Model %s predict failed: %s", model_id, e)

        if not predictions:
            return None

        # Check model agreement: at least MIN_AGREEING_MODELS must agree on direction
        n_bullish = sum(1 for p in predictions if p.prob_up > 0.5)
        n_bearish = sum(1 for p in predictions if p.prob_up < 0.5)
        max_agreeing = max(n_bullish, n_bearish)

        if len(predictions) >= MIN_AGREEING_MODELS and max_agreeing < MIN_AGREEING_MODELS:
            logger.info(
                "Ensemble halt: insufficient model agreement (bullish=%d, bearish=%d, need %d agreeing). Models: %s",
                n_bullish,
                n_bearish,
                MIN_AGREEING_MODELS,
                [(p.model_id, f"{p.prob_up:.3f}") for p in predictions],
            )
            return None

        total_weight = 0.0
        prob_up = 0.0
        exp_ret = 0.0
        conf = 0.0
        for p in predictions:
            # Default to 0.0 for unregistered models (not 1.0) to prevent
            # unknown models from dominating the ensemble
            w = self.weights.get(p.model_id, 0.0)
            total_weight += w
            prob_up += p.prob_up * w
            exp_ret += p.expected_return * w
            conf += p.confidence * w
        if total_weight < 1e-12:
            # No weighted models — return None to halt trading
            logger.warning(
                "Ensemble: total_weight < 1e-12 with %d predictions but no IC-weighted models. Halting.",
                len(predictions),
            )
            return None
        prob_up_f = prob_up / total_weight
        exp_ret_f = exp_ret / total_weight
        conf_f = min(1.0, conf / total_weight)

        # Disagreement penalty: scale confidence by agreement ratio
        n_models = len(predictions)
        if n_models >= 2:
            agreement_ratio = max_agreeing / n_models  # 3/5=0.6, 4/5=0.8, 5/5=1.0
            conf_f *= agreement_ratio
            logger.debug(
                "Ensemble agreement: %d/%d agree, penalty=%.2f, adjusted_conf=%.4f",
                max_agreeing,
                n_models,
                agreement_ratio,
                conf_f,
            )

        # Halt when probability is too close to coin flip (noise, not signal)
        if abs(prob_up_f - 0.5) < 0.05:
            logger.info(
                "Ensemble halt: prob_up=%.4f too close to 0.5 (noise). Models: %s",
                prob_up_f,
                [(p.model_id, f"{p.prob_up:.3f}") for p in predictions],
            )
            return None

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
            except Exception as e:
                logger.warning("Calibrator transform failed, using uncalibrated probability: %s", e)
        latency_ms = (time.monotonic() - _predict_start) * 1000
        return PredictionOutput(
            prob_up=float(np.clip(prob_up_f, 0.0, 1.0)),
            expected_return=exp_ret_f,
            confidence=conf_f,
            model_id="ensemble",
            version="v1",
            metadata={
                "count": len(predictions),
                "models": [p.model_id for p in predictions],
                "latency_ms": latency_ms,
                "n_bullish": n_bullish,
                "n_bearish": n_bearish,
                "model_agreement": max_agreeing >= MIN_AGREEING_MODELS,
                "diversity_score": self._last_diversity_score,
            },
        )
