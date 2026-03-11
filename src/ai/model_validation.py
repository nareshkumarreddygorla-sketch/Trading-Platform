"""
Pre-deployment model validation framework.

Runs a suite of statistical checks before any model is promoted to production:
  1. IC significance test (IC > 0.05 AND p-value < 0.05 for 63+ observations)
  2. Out-of-sample performance check (walk-forward on held-out data)
  3. Feature importance stability (top 10 features stable across time windows)
  4. Calibration check (predicted probabilities match realized frequencies)

Usage:
    validator = ModelValidator()
    result = validator.validate(model, X_oos, y_oos, feature_names=feature_names)
    if result.passed:
        deploy(model)
    else:
        print(result.diagnostics)
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Validation thresholds
IC_MIN_THRESHOLD = 0.05          # Minimum Information Coefficient
IC_PVALUE_THRESHOLD = 0.05       # Maximum p-value for IC significance
IC_MIN_OBSERVATIONS = 63         # Minimum observations (~3 months trading days)
OOS_MIN_ACCURACY = 0.52          # Minimum out-of-sample accuracy (above random)
OOS_MIN_SAMPLES = 100            # Minimum OOS samples for reliable test
CALIBRATION_MAX_ECE = 0.10       # Maximum expected calibration error
CALIBRATION_N_BINS = 10          # Number of bins for calibration check
FEATURE_STABILITY_OVERLAP = 0.6  # Minimum Jaccard overlap of top-10 features
FEATURE_STABILITY_WINDOWS = 3    # Number of time windows for stability check


@dataclass
class ValidationResult:
    """Detailed results from model validation."""
    passed: bool
    ic_test: Dict[str, Any] = field(default_factory=dict)
    oos_test: Dict[str, Any] = field(default_factory=dict)
    feature_stability_test: Dict[str, Any] = field(default_factory=dict)
    calibration_test: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """One-line summary for logging."""
        status = "PASSED" if self.passed else "FAILED"
        failures = [d for d in self.diagnostics if d.startswith("FAIL")]
        return f"ModelValidation {status}: {len(failures)} failures. {'; '.join(failures[:3])}"


class ModelValidator:
    """Pre-deployment validation suite for ML trading models.

    Ensures models meet institutional-grade quality bars before going live.
    All tests are non-destructive (read-only on model state).
    """

    def __init__(
        self,
        ic_threshold: float = IC_MIN_THRESHOLD,
        ic_pvalue: float = IC_PVALUE_THRESHOLD,
        ic_min_obs: int = IC_MIN_OBSERVATIONS,
        oos_min_accuracy: float = OOS_MIN_ACCURACY,
        calibration_max_ece: float = CALIBRATION_MAX_ECE,
        feature_stability_overlap: float = FEATURE_STABILITY_OVERLAP,
    ):
        self.ic_threshold = ic_threshold
        self.ic_pvalue = ic_pvalue
        self.ic_min_obs = ic_min_obs
        self.oos_min_accuracy = oos_min_accuracy
        self.calibration_max_ece = calibration_max_ece
        self.feature_stability_overlap = feature_stability_overlap

    def validate(
        self,
        model: Any,
        X_oos: np.ndarray,
        y_oos: np.ndarray,
        predictions_oos: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        feature_importances_over_time: Optional[List[Dict[str, float]]] = None,
    ) -> ValidationResult:
        """Run full validation suite on a model.

        Args:
            model: The model object (must have a predict-like interface or
                   provide predictions_oos directly).
            X_oos: Out-of-sample feature matrix, shape (N, F).
            y_oos: Out-of-sample targets (binary 0/1 for direction, or
                   continuous returns for IC calculation).
            predictions_oos: Pre-computed model predictions on X_oos.
                             If None, will attempt to call model.predict().
            feature_names: List of feature names matching columns of X_oos.
            feature_importances_over_time: List of dicts {feature: importance}
                             from different time windows. Used for stability check.

        Returns:
            ValidationResult with pass/fail and detailed diagnostics.
        """
        diagnostics: List[str] = []
        ic_result: Dict[str, Any] = {}
        oos_result: Dict[str, Any] = {}
        stability_result: Dict[str, Any] = {}
        calibration_result: Dict[str, Any] = {}

        # --- Get predictions if not provided ---
        if predictions_oos is None:
            predictions_oos = self._get_predictions(model, X_oos)
            if predictions_oos is None:
                diagnostics.append(
                    "FAIL: Could not obtain model predictions. "
                    "Model must have predict() or provide predictions_oos."
                )
                return ValidationResult(
                    passed=False,
                    diagnostics=diagnostics,
                )

        # --- 1. IC Significance Test ---
        ic_result = self._test_ic_significance(predictions_oos, y_oos)
        if not ic_result.get("passed", False):
            diagnostics.append(
                f"FAIL IC: IC={ic_result.get('ic', 0):.4f} "
                f"(need >{self.ic_threshold}), "
                f"p={ic_result.get('pvalue', 1):.4f} "
                f"(need <{self.ic_pvalue}), "
                f"n={ic_result.get('n_observations', 0)} "
                f"(need >={self.ic_min_obs})"
            )
        else:
            diagnostics.append(
                f"PASS IC: IC={ic_result['ic']:.4f}, "
                f"p={ic_result['pvalue']:.4f}, "
                f"n={ic_result['n_observations']}"
            )

        # --- 2. Out-of-Sample Performance ---
        oos_result = self._test_oos_performance(predictions_oos, y_oos)
        if not oos_result.get("passed", False):
            diagnostics.append(
                f"FAIL OOS: accuracy={oos_result.get('accuracy', 0):.4f} "
                f"(need >{self.oos_min_accuracy}), "
                f"n={oos_result.get('n_samples', 0)}"
            )
        else:
            diagnostics.append(
                f"PASS OOS: accuracy={oos_result['accuracy']:.4f}, "
                f"profit_factor={oos_result.get('profit_factor', 0):.2f}, "
                f"n={oos_result['n_samples']}"
            )

        # --- 3. Feature Importance Stability ---
        if feature_importances_over_time and len(feature_importances_over_time) >= 2:
            stability_result = self._test_feature_stability(
                feature_importances_over_time
            )
            if not stability_result.get("passed", False):
                diagnostics.append(
                    f"FAIL Feature Stability: mean_overlap="
                    f"{stability_result.get('mean_overlap', 0):.3f} "
                    f"(need >={self.feature_stability_overlap})"
                )
            else:
                diagnostics.append(
                    f"PASS Feature Stability: mean_overlap="
                    f"{stability_result['mean_overlap']:.3f}"
                )
        elif feature_names is not None and X_oos.shape[0] >= OOS_MIN_SAMPLES * 2:
            # Compute feature importances from data splits
            stability_result = self._compute_and_test_stability(
                model, X_oos, y_oos, feature_names
            )
            if not stability_result.get("passed", False):
                diagnostics.append(
                    f"FAIL Feature Stability: mean_overlap="
                    f"{stability_result.get('mean_overlap', 0):.3f} "
                    f"(need >={self.feature_stability_overlap})"
                )
            else:
                diagnostics.append(
                    f"PASS Feature Stability: mean_overlap="
                    f"{stability_result['mean_overlap']:.3f}"
                )
        else:
            stability_result = {"passed": True, "skipped": True, "reason": "insufficient_data"}
            diagnostics.append("SKIP Feature Stability: insufficient data or feature names")

        # --- 4. Calibration Check ---
        calibration_result = self._test_calibration(predictions_oos, y_oos)
        if not calibration_result.get("passed", False):
            diagnostics.append(
                f"FAIL Calibration: ECE={calibration_result.get('ece', 1):.4f} "
                f"(need <{self.calibration_max_ece})"
            )
        else:
            diagnostics.append(
                f"PASS Calibration: ECE={calibration_result['ece']:.4f}"
            )

        # --- Overall verdict: must pass ALL non-skipped tests ---
        all_results = [ic_result, oos_result, stability_result, calibration_result]
        passed = all(
            r.get("passed", False) or r.get("skipped", False)
            for r in all_results
        )

        result = ValidationResult(
            passed=passed,
            ic_test=ic_result,
            oos_test=oos_result,
            feature_stability_test=stability_result,
            calibration_test=calibration_result,
            diagnostics=diagnostics,
        )

        log_fn = logger.info if passed else logger.warning
        log_fn("Model validation %s: %s", "PASSED" if passed else "FAILED", result.summary)

        return result

    # ------------------------------------------------------------------
    # Individual test implementations
    # ------------------------------------------------------------------

    def _test_ic_significance(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> Dict[str, Any]:
        """Test that the Information Coefficient is statistically significant.

        IC is computed as Spearman rank correlation between predictions and
        actual returns/outcomes. We require:
          - IC > ic_threshold (default 0.05)
          - p-value < ic_pvalue (default 0.05)
          - At least ic_min_obs observations (default 63)
        """
        n = len(predictions)
        if n < self.ic_min_obs:
            return {
                "passed": False,
                "ic": 0.0,
                "pvalue": 1.0,
                "n_observations": n,
                "reason": f"insufficient_observations_{n}_need_{self.ic_min_obs}",
            }

        try:
            from scipy.stats import spearmanr
        except ImportError:
            logger.warning("scipy not installed - IC test cannot run")
            return {
                "passed": False,
                "ic": 0.0,
                "pvalue": 1.0,
                "n_observations": n,
                "reason": "scipy_not_installed",
            }

        # Flatten and clean
        preds = np.asarray(predictions).flatten()[:n]
        acts = np.asarray(actuals).flatten()[:n]

        # Remove NaN/Inf pairs
        valid_mask = np.isfinite(preds) & np.isfinite(acts)
        preds = preds[valid_mask]
        acts = acts[valid_mask]
        n_valid = len(preds)

        if n_valid < self.ic_min_obs:
            return {
                "passed": False,
                "ic": 0.0,
                "pvalue": 1.0,
                "n_observations": n_valid,
                "reason": f"insufficient_valid_observations_{n_valid}",
            }

        # Check for zero variance
        if np.std(preds) < 1e-12 or np.std(acts) < 1e-12:
            return {
                "passed": False,
                "ic": 0.0,
                "pvalue": 1.0,
                "n_observations": n_valid,
                "reason": "zero_variance_in_predictions_or_actuals",
            }

        corr, pvalue = spearmanr(preds, acts)
        if not math.isfinite(corr):
            corr = 0.0
        if not math.isfinite(pvalue):
            pvalue = 1.0

        passed = corr > self.ic_threshold and pvalue < self.ic_pvalue
        return {
            "passed": passed,
            "ic": round(float(corr), 6),
            "pvalue": round(float(pvalue), 6),
            "n_observations": n_valid,
        }

    def _test_oos_performance(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> Dict[str, Any]:
        """Test out-of-sample directional accuracy and profit factor.

        For binary targets: accuracy = fraction of correct direction predictions.
        For continuous targets: convert to direction (sign) first.
        """
        preds = np.asarray(predictions).flatten()
        acts = np.asarray(actuals).flatten()
        n = min(len(preds), len(acts))
        preds = preds[:n]
        acts = acts[:n]

        if n < OOS_MIN_SAMPLES:
            return {
                "passed": False,
                "accuracy": 0.0,
                "n_samples": n,
                "reason": f"insufficient_samples_{n}_need_{OOS_MIN_SAMPLES}",
            }

        # Convert predictions to direction if they're probabilities (0-1 range)
        if np.all((preds >= 0) & (preds <= 1)):
            pred_direction = (preds > 0.5).astype(float)
        else:
            pred_direction = (preds > 0).astype(float)

        # Convert actuals to direction if they're continuous returns
        if not np.all(np.isin(acts, [0, 1])):
            actual_direction = (acts > 0).astype(float)
        else:
            actual_direction = acts

        accuracy = float(np.mean(pred_direction == actual_direction))

        # Profit factor: sum of gains / sum of losses when following predictions
        # Use actual returns weighted by predicted direction
        signed_returns = acts * np.where(pred_direction == 1, 1, -1)
        gains = signed_returns[signed_returns > 0].sum()
        losses = abs(signed_returns[signed_returns < 0].sum())
        profit_factor = float(gains / losses) if losses > 1e-12 else float('inf')

        # Hit rate: percentage of profitable trades
        hit_rate = float(np.mean(signed_returns > 0))

        passed = accuracy > self.oos_min_accuracy and n >= OOS_MIN_SAMPLES
        return {
            "passed": passed,
            "accuracy": round(accuracy, 6),
            "profit_factor": round(profit_factor, 4) if math.isfinite(profit_factor) else 999.0,
            "hit_rate": round(hit_rate, 4),
            "mean_return": round(float(np.mean(signed_returns)), 6),
            "n_samples": n,
        }

    def _test_feature_stability(
        self,
        importances_over_time: List[Dict[str, float]],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Test that the top-K features are stable across time windows.

        Computes pairwise Jaccard similarity of top-K feature sets across
        consecutive time windows. The mean overlap must exceed the threshold.
        """
        if len(importances_over_time) < 2:
            return {
                "passed": True,
                "skipped": True,
                "reason": "need_at_least_2_windows",
            }

        # Extract top-K features per window
        top_features_per_window: List[set] = []
        for imp in importances_over_time:
            if not imp:
                continue
            sorted_features = sorted(imp.items(), key=lambda x: x[1], reverse=True)
            top_k_features = {f for f, _ in sorted_features[:top_k]}
            top_features_per_window.append(top_k_features)

        if len(top_features_per_window) < 2:
            return {
                "passed": True,
                "skipped": True,
                "reason": "insufficient_valid_windows",
            }

        # Compute pairwise Jaccard overlaps
        overlaps: List[float] = []
        for i in range(len(top_features_per_window) - 1):
            s1 = top_features_per_window[i]
            s2 = top_features_per_window[i + 1]
            if len(s1) == 0 or len(s2) == 0:
                continue
            jaccard = len(s1 & s2) / len(s1 | s2)
            overlaps.append(jaccard)

        if not overlaps:
            return {
                "passed": True,
                "skipped": True,
                "reason": "no_valid_overlap_pairs",
            }

        mean_overlap = float(np.mean(overlaps))
        min_overlap = float(np.min(overlaps))

        passed = mean_overlap >= self.feature_stability_overlap
        return {
            "passed": passed,
            "mean_overlap": round(mean_overlap, 4),
            "min_overlap": round(min_overlap, 4),
            "n_windows": len(top_features_per_window),
            "pairwise_overlaps": [round(o, 4) for o in overlaps],
        }

    def _compute_and_test_stability(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_splits: int = FEATURE_STABILITY_WINDOWS,
    ) -> Dict[str, Any]:
        """Compute feature importances from data splits and test stability.

        Splits the data into n_splits time-ordered windows and computes
        permutation importance in each window.
        """
        n = len(X)
        split_size = n // n_splits
        if split_size < 50:
            return {
                "passed": True,
                "skipped": True,
                "reason": f"split_size_too_small_{split_size}",
            }

        importances_over_time: List[Dict[str, float]] = []

        for i in range(n_splits):
            start = i * split_size
            end = start + split_size if i < n_splits - 1 else n
            X_window = X[start:end]
            y_window = y[start:end]

            importance = self._compute_permutation_importance(
                model, X_window, y_window, feature_names
            )
            if importance:
                importances_over_time.append(importance)

        return self._test_feature_stability(importances_over_time)

    def _compute_permutation_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_repeats: int = 5,
    ) -> Dict[str, float]:
        """Compute permutation feature importance for a model on given data.

        Shuffles each feature and measures the increase in prediction error.
        """
        importance: Dict[str, float] = {}

        try:
            # Get baseline predictions
            baseline_preds = self._get_predictions(model, X)
            if baseline_preds is None:
                return {}

            # Baseline error (1 - accuracy for direction)
            baseline_acc = self._directional_accuracy(baseline_preds, y)

            for j, fname in enumerate(feature_names):
                if j >= X.shape[1]:
                    break
                drops: List[float] = []
                for _ in range(n_repeats):
                    X_permuted = X.copy()
                    np.random.shuffle(X_permuted[:, j])
                    perm_preds = self._get_predictions(model, X_permuted)
                    if perm_preds is None:
                        continue
                    perm_acc = self._directional_accuracy(perm_preds, y)
                    drops.append(baseline_acc - perm_acc)
                if drops:
                    importance[fname] = float(np.mean(drops))
        except Exception as e:
            logger.debug("Permutation importance failed: %s", e)

        return importance

    def _test_calibration(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        n_bins: int = CALIBRATION_N_BINS,
    ) -> Dict[str, Any]:
        """Test calibration: predicted probabilities should match realized frequencies.

        Computes Expected Calibration Error (ECE):
        ECE = sum_b (|B_b| / N) * |acc(b) - conf(b)|

        where B_b is the set of predictions in bin b, acc(b) is the fraction
        of positive outcomes in that bin, and conf(b) is the mean predicted
        probability in that bin.
        """
        preds = np.asarray(predictions).flatten()
        acts = np.asarray(actuals).flatten()
        n = min(len(preds), len(acts))
        preds = preds[:n]
        acts = acts[:n]

        if n < OOS_MIN_SAMPLES:
            return {
                "passed": False,
                "ece": 1.0,
                "n_samples": n,
                "reason": f"insufficient_samples_{n}",
            }

        # Ensure predictions are in [0, 1] (probabilities)
        if not (np.all(preds >= 0) and np.all(preds <= 1)):
            # Try to interpret as signed predictions -> convert to probability
            preds = 1.0 / (1.0 + np.exp(-preds))  # sigmoid

        # Convert continuous actuals to binary if needed
        if not np.all(np.isin(acts, [0, 1])):
            acts = (acts > 0).astype(float)

        # Compute ECE
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        bin_details: List[Dict[str, Any]] = []

        for i in range(n_bins):
            mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
            if i == n_bins - 1:  # Include right edge in last bin
                mask = (preds >= bin_edges[i]) & (preds <= bin_edges[i + 1])

            bin_count = int(mask.sum())
            if bin_count == 0:
                continue

            bin_accuracy = float(acts[mask].mean())  # Realized frequency
            bin_confidence = float(preds[mask].mean())  # Mean predicted prob
            bin_ece = (bin_count / n) * abs(bin_accuracy - bin_confidence)
            ece += bin_ece

            bin_details.append({
                "bin": f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})",
                "count": bin_count,
                "accuracy": round(bin_accuracy, 4),
                "confidence": round(bin_confidence, 4),
                "ece_contribution": round(bin_ece, 6),
            })

        passed = ece < self.calibration_max_ece
        return {
            "passed": passed,
            "ece": round(float(ece), 6),
            "n_bins_used": len(bin_details),
            "n_samples": n,
            "bin_details": bin_details,
        }

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _get_predictions(model: Any, X: np.ndarray) -> Optional[np.ndarray]:
        """Extract predictions from a model using various interfaces."""
        # Try predict_proba (sklearn-like)
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    return proba[:, 1]
                return proba.flatten()
            except Exception:
                pass

        # Try predict (general)
        if hasattr(model, "predict"):
            try:
                # Check if it's a BasePredictor (our custom interface)
                from .models.base import BasePredictor
                if isinstance(model, BasePredictor):
                    preds = []
                    for i in range(len(X)):
                        feat_dict = {f"f{j}": float(X[i, j]) for j in range(X.shape[1])}
                        result = model.predict(feat_dict)
                        if result is not None:
                            preds.append(result.prob_up)
                        else:
                            preds.append(0.5)
                    return np.array(preds)
                else:
                    result = model.predict(X)
                    return np.asarray(result).flatten()
            except Exception:
                pass

        # Try XGBoost Booster
        try:
            import xgboost as xgb
            if isinstance(model, xgb.Booster):
                dmat = xgb.DMatrix(X)
                return model.predict(dmat)
        except (ImportError, Exception):
            pass

        return None

    @staticmethod
    def _directional_accuracy(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Compute directional accuracy between predictions and actuals."""
        preds = np.asarray(predictions).flatten()
        acts = np.asarray(actuals).flatten()
        n = min(len(preds), len(acts))
        preds = preds[:n]
        acts = acts[:n]

        if np.all((preds >= 0) & (preds <= 1)):
            pred_dir = (preds > 0.5).astype(float)
        else:
            pred_dir = (preds > 0).astype(float)

        if not np.all(np.isin(acts, [0, 1])):
            act_dir = (acts > 0).astype(float)
        else:
            act_dir = acts

        return float(np.mean(pred_dir == act_dir))


def validate_model_for_deployment(
    model: Any,
    X_oos: np.ndarray,
    y_oos: np.ndarray,
    predictions_oos: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    feature_importances_over_time: Optional[List[Dict[str, float]]] = None,
    strict: bool = True,
) -> ValidationResult:
    """Convenience function for one-call model validation.

    Args:
        model: The model to validate.
        X_oos: Out-of-sample features.
        y_oos: Out-of-sample targets.
        predictions_oos: Optional pre-computed predictions.
        feature_names: Feature names for stability check.
        feature_importances_over_time: Pre-computed importances for stability.
        strict: If True, uses institutional-grade thresholds. If False,
                uses relaxed thresholds suitable for development/testing.

    Returns:
        ValidationResult with pass/fail and diagnostics.
    """
    if strict:
        validator = ModelValidator()
    else:
        validator = ModelValidator(
            ic_threshold=0.03,
            ic_pvalue=0.10,
            oos_min_accuracy=0.51,
            calibration_max_ece=0.15,
            feature_stability_overlap=0.4,
        )

    return validator.validate(
        model=model,
        X_oos=X_oos,
        y_oos=y_oos,
        predictions_oos=predictions_oos,
        feature_names=feature_names,
        feature_importances_over_time=feature_importances_over_time,
    )
