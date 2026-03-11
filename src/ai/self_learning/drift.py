"""
Concept drift detection and data distribution monitoring.
Triggers retrain when feature or target distribution shifts.
"""
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ConceptDriftDetector:
    """
    Detect drift between reference (training) distribution and current.
    Simple approach: compare mean/std of key features; or use KS test / PSI.
    """

    def __init__(self, threshold: float = 0.3, reference_stats: Optional[Dict[str, Dict[str, float]]] = None):
        self.threshold = threshold
        self.reference_stats = reference_stats or {}  # feature_name -> {mean, std}

    def set_reference(self, feature_matrix: np.ndarray, feature_names: List[str]) -> None:
        """Set reference stats from training data."""
        for i, name in enumerate(feature_names):
            if i < feature_matrix.shape[1]:
                col = feature_matrix[:, i]
                self.reference_stats[name] = {"mean": float(np.mean(col)), "std": float(np.std(col)) or 1e-12}

    def detect(self, features: Dict[str, float]) -> tuple[bool, str]:
        """
        Return (drift_detected, reason). Drift if current feature stats
        deviate from reference beyond threshold.
        """
        for name, ref in self.reference_stats.items():
            val = features.get(name)
            if val is None:
                continue
            mean_ref = ref["mean"]
            std_ref = ref["std"]
            if std_ref < 1e-12:
                continue
            z = abs(val - mean_ref) / std_ref
            if z > self.threshold * 10:  # scale
                return True, f"feature_{name}_drift_z={z:.2f}"
        return False, ""

    def detect_batch(self, feature_matrix: np.ndarray, feature_names: List[str]) -> tuple[bool, str]:
        """Drift on batch: compare current batch stats to reference."""
        if not self.reference_stats:
            return False, ""
        for i, name in enumerate(feature_names):
            if name not in self.reference_stats or i >= feature_matrix.shape[1]:
                continue
            col = feature_matrix[:, i]
            curr_mean = np.mean(col)
            curr_std = np.std(col) or 1e-12
            ref = self.reference_stats[name]
            mean_diff = abs(curr_mean - ref["mean"]) / (ref["std"] + 1e-12)
            std_ratio = curr_std / (ref["std"] + 1e-12)
            if mean_diff > self.threshold * 5 or std_ratio < 0.5 or std_ratio > 2.0:
                return True, f"batch_{name}_drift"
        return False, ""


class DataDistributionMonitor:
    """Track feature distribution over time; alert on significant shift."""

    def __init__(self, window: int = 100):
        self.window = window
        self._samples: Dict[str, List[float]] = {}

    def add(self, features: Dict[str, float]) -> None:
        for k, v in features.items():
            if isinstance(v, (int, float)):
                if k not in self._samples:
                    self._samples[k] = []
                self._samples[k].append(float(v))
                if len(self._samples[k]) > self.window:
                    self._samples[k] = self._samples[k][-self.window :]

    def psi(self, feature_name: str, bins: int = 10) -> float:
        """Population Stability Index: current vs previous half. >0.2 suggests shift."""
        if feature_name not in self._samples or len(self._samples[feature_name]) < 20:
            return 0.0
        arr = np.array(self._samples[feature_name])
        n = len(arr)
        first = arr[: n // 2]
        second = arr[n // 2 :]
        min_val = min(first.min(), second.min())
        max_val = max(first.max(), second.max())
        if max_val - min_val < 1e-12:
            return 0.0
        edges = np.linspace(min_val, max_val, bins + 1)
        p1, _ = np.histogram(first, bins=edges)
        p2, _ = np.histogram(second, bins=edges)
        p1 = (p1 + 1e-6) / (p1.sum() + 1e-6 * bins)
        p2 = (p2 + 1e-6) / (p2.sum() + 1e-6 * bins)
        psi = np.sum((p2 - p1) * np.log((p2 + 1e-6) / (p1 + 1e-6)))
        return float(psi)
