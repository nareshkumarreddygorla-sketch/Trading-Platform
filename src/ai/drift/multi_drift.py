"""
Multi-layer drift: prediction distribution, calibration, rolling Sharpe,
feature importance, regime frequency, correlation structure.
Threshold design; avoid false retrains; shadow model evaluation before replace.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np


class DriftType(str, Enum):
    PREDICTION_DIST = "prediction_dist"
    CALIBRATION = "calibration"
    SHARPE = "sharpe"
    FEATURE_IMPORTANCE = "feature_importance"
    REGIME_FREQ = "regime_freq"
    CORRELATION = "correlation"


@dataclass
class DriftSignal:
    drifted: bool
    drift_type: DriftType
    value: float
    threshold: float
    detail: str = ""


class MultiLayerDriftDetector:
    """
    Multiple drift checks; each returns DriftSignal.
    Thresholds: use historical percentiles (e.g. alert if PSI > 95th pct of historical).
    """

    def __init__(
        self,
        psi_threshold: float = 0.2,
        sharpe_drop_threshold: float = 0.5,
        calibration_mse_threshold: float = 0.01,
        importance_cosine_min: float = 0.9,
        regime_chi2_threshold: float = 10.0,
        corr_frobenius_threshold: float = 0.5,
    ):
        self.psi_threshold = psi_threshold
        self.sharpe_drop_threshold = sharpe_drop_threshold
        self.calibration_mse_threshold = calibration_mse_threshold
        self.importance_cosine_min = importance_cosine_min
        self.regime_chi2_threshold = regime_chi2_threshold
        self.corr_frobenius_threshold = corr_frobenius_threshold
        self._reference_pred: Optional[np.ndarray] = None
        self._reference_importance: Optional[np.ndarray] = None
        self._reference_regime_dist: Optional[np.ndarray] = None
        self._reference_corr: Optional[np.ndarray] = None
        self._peak_sharpe: float = 0.0

    def set_reference(
        self,
        pred: Optional[np.ndarray] = None,
        importance: Optional[np.ndarray] = None,
        regime_dist: Optional[np.ndarray] = None,
        corr: Optional[np.ndarray] = None,
    ) -> None:
        self._reference_pred = pred
        self._reference_importance = importance
        self._reference_regime_dist = regime_dist
        self._reference_corr = corr

    def check_prediction_dist(self, current_pred: np.ndarray, n_bins: int = 10) -> DriftSignal:
        """PSI on binned predictions."""
        if self._reference_pred is None or len(self._reference_pred) < 10:
            return DriftSignal(False, DriftType.PREDICTION_DIST, 0.0, self.psi_threshold)
        ref_hist, _ = np.histogram(self._reference_pred, bins=n_bins, range=(0, 1))
        cur_hist, _ = np.histogram(np.clip(current_pred, 0, 1), bins=n_bins, range=(0, 1))
        ref_p = (ref_hist + 1e-6) / (ref_hist.sum() + 1e-6 * n_bins)
        cur_p = (cur_hist + 1e-6) / (cur_hist.sum() + 1e-6 * n_bins)
        psi = np.sum((cur_p - ref_p) * np.log((cur_p + 1e-6) / (ref_p + 1e-6)))
        return DriftSignal(
            drifted=psi > self.psi_threshold,
            drift_type=DriftType.PREDICTION_DIST,
            value=float(psi),
            threshold=self.psi_threshold,
            detail=f"PSI={psi:.4f}",
        )

    def check_calibration(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> DriftSignal:
        """MSE of binned predicted prob vs realized frequency."""
        from ..calibration.calibrate import reliability_curve
        _, mean_pred, mean_real = reliability_curve(y_true, y_prob, n_bins)
        valid = ~np.isnan(mean_real)
        if not np.any(valid):
            return DriftSignal(False, DriftType.CALIBRATION, 0.0, self.calibration_mse_threshold)
        mse = float(np.nanmean((mean_pred[valid] - mean_real[valid]) ** 2))
        return DriftSignal(
            drifted=mse > self.calibration_mse_threshold,
            drift_type=DriftType.CALIBRATION,
            value=mse,
            threshold=self.calibration_mse_threshold,
        )

    def check_sharpe_drop(self, current_sharpe: float) -> DriftSignal:
        """Alert if current Sharpe dropped from peak by more than threshold."""
        self._peak_sharpe = max(self._peak_sharpe, current_sharpe)
        drop = self._peak_sharpe - current_sharpe
        return DriftSignal(
            drifted=drop >= self.sharpe_drop_threshold,
            drift_type=DriftType.SHARPE,
            value=current_sharpe,
            threshold=self._peak_sharpe - self.sharpe_drop_threshold,
            detail=f"Sharpe drop {drop:.3f}",
        )

    def check_importance_divergence(self, current_importance: np.ndarray) -> DriftSignal:
        """Cosine similarity of current vs reference importance."""
        if self._reference_importance is None or len(current_importance) != len(self._reference_importance):
            return DriftSignal(False, DriftType.FEATURE_IMPORTANCE, 1.0, self.importance_cosine_min)
        a = np.asarray(current_importance, dtype=float)
        b = np.asarray(self._reference_importance, dtype=float)
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        cos = float(np.dot(a, b))
        return DriftSignal(
            drifted=cos < self.importance_cosine_min,
            drift_type=DriftType.FEATURE_IMPORTANCE,
            value=cos,
            threshold=self.importance_cosine_min,
        )

    def run_all(
        self,
        current_pred: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None,
        y_prob: Optional[np.ndarray] = None,
        current_sharpe: Optional[float] = None,
        current_importance: Optional[np.ndarray] = None,
    ) -> List[DriftSignal]:
        """Run all applicable checks; return list of DriftSignal."""
        out: List[DriftSignal] = []
        if current_pred is not None:
            out.append(self.check_prediction_dist(current_pred))
        if y_true is not None and y_prob is not None:
            out.append(self.check_calibration(y_true, y_prob))
        if current_sharpe is not None:
            out.append(self.check_sharpe_drop(current_sharpe))
        if current_importance is not None:
            out.append(self.check_importance_divergence(current_importance))
        return out
