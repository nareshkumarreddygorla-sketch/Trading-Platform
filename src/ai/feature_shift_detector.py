"""
Feature distribution shift detector for live inference.

Monitors live feature distributions against training-time statistics and alerts
when significant drift occurs, preventing stale/miscalibrated model predictions.

Uses Population Stability Index (PSI) and Kolmogorov-Smirnov test for detection.

Check interval: configurable, defaults to every 30 minutes during market hours
(09:15 - 15:30 IST). The trading loop should call ``should_check_now()`` to
determine if a check is due, and ``check_shift()`` to run it.
"""
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# PSI thresholds (industry standard)
PSI_INSIGNIFICANT = 0.10    # < 0.10: no significant shift
PSI_MODERATE = 0.20         # 0.10-0.20: moderate shift (monitor)
PSI_SIGNIFICANT = 0.25      # > 0.25: significant shift (alert + potential retrain)

# Recommendation thresholds
RETRAIN_SIGNIFICANT_COUNT = 3
MONITOR_MODERATE_FRACTION = 0.25


@dataclass
class ShiftResult:
    """Result of a distribution shift check for a single feature."""
    feature_name: str
    psi: float                      # Population Stability Index
    ks_stat: float                  # Kolmogorov-Smirnov statistic
    ks_pvalue: float                # KS test p-value
    training_mean: float
    training_std: float
    live_mean: float
    live_std: float
    shift_level: str                # "none", "moderate", "significant"
    mean_shift_pct: float           # How much the mean has shifted (%)

    def as_dict(self) -> dict:
        return {
            "feature": self.feature_name,
            "psi": round(self.psi, 4),
            "ks_stat": round(self.ks_stat, 4),
            "ks_pvalue": round(self.ks_pvalue, 4),
            "training_mean": round(self.training_mean, 6),
            "live_mean": round(self.live_mean, 6),
            "shift_level": self.shift_level,
            "mean_shift_pct": round(self.mean_shift_pct, 2),
        }


@dataclass
class HaltDecision:
    """Structured halt decision that the trading loop can act on directly."""
    should_halt: bool               # True = stop trading immediately
    should_reduce_size: bool        # True = reduce position sizes (moderate drift)
    should_retrain: bool            # True = trigger model retraining
    reason: str                     # Human-readable reason
    severity: str                   # "none", "moderate", "significant", "critical"
    max_psi: float                  # Worst PSI score for logging
    shifted_features_count: int     # Number of features with drift
    check_timestamp: str            # When this decision was made

    def as_dict(self) -> dict:
        return {
            "should_halt": self.should_halt,
            "should_reduce_size": self.should_reduce_size,
            "should_retrain": self.should_retrain,
            "reason": self.reason,
            "severity": self.severity,
            "max_psi": round(self.max_psi, 4),
            "shifted_features_count": self.shifted_features_count,
            "check_timestamp": self.check_timestamp,
        }


@dataclass
class OverallShiftReport:
    """Aggregate shift report across all features."""
    timestamp: str
    total_features: int
    features_shifted: int
    features_moderate: int
    features_significant: int
    avg_psi: float
    max_psi: float
    max_psi_feature: str
    recommendation: str             # "ok", "monitor", "retrain", "halt"
    halt_decision: Optional[HaltDecision] = None
    details: List[ShiftResult] = field(default_factory=list)

    def as_dict(self) -> dict:
        result = {
            "timestamp": self.timestamp,
            "total_features": self.total_features,
            "features_shifted": self.features_shifted,
            "features_moderate": self.features_moderate,
            "features_significant": self.features_significant,
            "avg_psi": round(self.avg_psi, 4),
            "max_psi": round(self.max_psi, 4),
            "max_psi_feature": self.max_psi_feature,
            "recommendation": self.recommendation,
            "shifted_features": [d.as_dict() for d in self.details if d.shift_level != "none"],
        }
        if self.halt_decision is not None:
            result["halt_decision"] = self.halt_decision.as_dict()
        return result


class FeatureShiftDetector:
    """
    Monitors live feature distributions against training-time baselines.

    Usage:
        detector = FeatureShiftDetector()
        detector.set_training_stats({"rsi_14": {"mean": 50.0, "std": 15.0, "histogram": [...]}, ...})
        # During live inference:
        detector.record_live_features({"rsi_14": 45.2, "macd": 0.003, ...})
        # Periodically check:
        report = detector.check_shift()
        if report.recommendation == "retrain":
            trigger_retrain()
    """

    def __init__(
        self,
        psi_bins: int = 10,
        min_samples: int = 100,
        alert_callback=None,
        halt_threshold: float = 0.30,
        check_interval_minutes: int = 30,
        market_open: dtime = dtime(9, 15),   # IST market open
        market_close: dtime = dtime(15, 30),  # IST market close
    ):
        self.psi_bins = psi_bins
        self.min_samples = min_samples
        self._alert_callback = alert_callback
        self._halt_threshold = halt_threshold
        self._check_interval_minutes = check_interval_minutes
        self._market_open = market_open
        self._market_close = market_close

        # Training-time statistics: feature_name -> {mean, std, min, max, percentiles, histogram}
        self._training_stats: Dict[str, Dict[str, Any]] = {}

        # Live feature buffer: feature_name -> list of recent values
        self._live_buffer: Dict[str, List[float]] = defaultdict(list)
        self._max_buffer_size: int = 5000

        # History of shift reports for trend monitoring
        self._shift_history: List[OverallShiftReport] = []

        # Last check timestamp for interval scheduling
        self._last_check_time: Optional[datetime] = None

        # Last halt decision (structured response for trading loop)
        self._last_halt_decision: Optional[HaltDecision] = None

    def __repr__(self) -> str:
        return (
            f"<FeatureShiftDetector features={len(self._training_stats)} "
            f"live_buffer_keys={len(self._live_buffer)} "
            f"halt_threshold={self._halt_threshold} "
            f"check_interval={self._check_interval_minutes}min>"
        )

    def set_training_stats(self, stats: Dict[str, Dict[str, Any]]) -> None:
        """Set training-time feature statistics as baseline for comparison."""
        self._training_stats = dict(stats)
        logger.info("Feature shift detector: loaded training stats for %d features", len(stats))

    def load_training_stats_from_arrays(self, feature_names: List[str], feature_matrix: np.ndarray) -> None:
        """Compute and store training statistics from a feature matrix (n_samples x n_features)."""
        if feature_matrix.shape[1] != len(feature_names):
            raise ValueError(f"Feature matrix has {feature_matrix.shape[1]} columns but {len(feature_names)} names")

        for i, name in enumerate(feature_names):
            col = feature_matrix[:, i]
            valid = col[np.isfinite(col)]
            if len(valid) < 10:
                continue
            percentiles = np.percentile(valid, [5, 25, 50, 75, 95]).tolist()
            hist, bin_edges = np.histogram(valid, bins=self.psi_bins, density=False)
            # Store as proportions (relative frequencies) not density
            hist = hist.astype(np.float64)
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist = hist / hist_sum
            self._training_stats[name] = {
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid)),
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
                "percentiles": percentiles,
                "histogram": hist.tolist(),
                "bin_edges": bin_edges.tolist(),
                "n_samples": len(valid),
            }
        logger.info("Feature shift detector: computed training stats for %d features from %d samples",
                    len(self._training_stats), feature_matrix.shape[0])

    def record_live_features(self, features: Dict[str, float]) -> None:
        """Record a single observation of live features."""
        for name, value in features.items():
            if not math.isfinite(value):
                continue
            buf = self._live_buffer[name]
            buf.append(value)
            if len(buf) > self._max_buffer_size:
                self._live_buffer[name] = buf[-self._max_buffer_size:]

    def _compute_psi(self, training_hist: List[float], live_values: np.ndarray, bin_edges: List[float]) -> float:
        """Compute Population Stability Index between training histogram and live values."""
        live_hist, _ = np.histogram(live_values, bins=bin_edges, density=False)
        live_hist = live_hist.astype(np.float64)

        # Normalize to proportions
        train_props = np.array(training_hist, dtype=np.float64)
        live_props = np.array(live_hist, dtype=np.float64)

        # Normalize so they sum to 1
        train_sum = train_props.sum()
        live_sum = live_props.sum()
        if train_sum > 0:
            train_props /= train_sum
        if live_sum > 0:
            live_props /= live_sum

        # Replace zeros with small epsilon to avoid log(0)
        eps = 1e-8
        train_props = np.maximum(train_props, eps)
        live_props = np.maximum(live_props, eps)

        # PSI = sum((live_i - train_i) * ln(live_i / train_i))
        psi = float(np.sum((live_props - train_props) * np.log(live_props / train_props)))
        return max(0.0, psi)

    def _compute_ks(self, training_stats: Dict, live_values: np.ndarray) -> Tuple[float, float]:
        """Compute Kolmogorov-Smirnov test statistic using normal approximation."""
        training_mean = training_stats.get("mean", 0.0)
        training_std = training_stats.get("std", 1.0)

        if training_std < 1e-10:
            return 0.0, 1.0

        # Standardize live values using training statistics
        standardized = (live_values - training_mean) / training_std

        try:
            from scipy.stats import kstest
            stat, pvalue = kstest(standardized, 'norm')
            return float(stat), float(pvalue)
        except ImportError:
            # Fallback: use simple z-test
            live_mean = np.mean(live_values)
            n = len(live_values)
            z = abs(live_mean - training_mean) / (training_std / math.sqrt(max(n, 1)))
            # Approximate p-value from z-score
            pvalue = max(0.0, 2 * (1 - min(1.0, 0.5 * (1 + math.erf(z / math.sqrt(2))))))
            return min(1.0, z / 10.0), pvalue  # Crude KS approximation

    def check_shift(self) -> OverallShiftReport:
        """Run distribution shift analysis on all features with sufficient data."""
        results: List[ShiftResult] = []

        for feature_name, stats in self._training_stats.items():
            live_values = self._live_buffer.get(feature_name, [])
            if len(live_values) < self.min_samples:
                continue

            live_arr = np.array(live_values, dtype=np.float64)
            training_mean = stats.get("mean", 0.0)
            training_std = stats.get("std", 1.0)
            live_mean = float(np.mean(live_arr))
            live_std = float(np.std(live_arr))

            # Mean shift percentage
            if abs(training_mean) > 1e-10:
                mean_shift_pct = abs((live_mean - training_mean) / training_mean) * 100
            else:
                mean_shift_pct = abs(live_mean - training_mean) * 100

            # PSI
            psi = 0.0
            if "histogram" in stats and "bin_edges" in stats:
                psi = self._compute_psi(stats["histogram"], live_arr, stats["bin_edges"])

            # KS test
            ks_stat, ks_pvalue = self._compute_ks(stats, live_arr)

            # Determine shift level
            if psi >= PSI_SIGNIFICANT or ks_pvalue < 0.01:
                shift_level = "significant"
            elif psi >= PSI_MODERATE or ks_pvalue < 0.05:
                shift_level = "moderate"
            else:
                shift_level = "none"

            results.append(ShiftResult(
                feature_name=feature_name,
                psi=psi,
                ks_stat=ks_stat,
                ks_pvalue=ks_pvalue,
                training_mean=training_mean,
                training_std=training_std,
                live_mean=live_mean,
                live_std=live_std,
                shift_level=shift_level,
                mean_shift_pct=mean_shift_pct,
            ))

        # Aggregate
        n_significant = sum(1 for r in results if r.shift_level == "significant")
        n_moderate = sum(1 for r in results if r.shift_level == "moderate")
        n_shifted = n_significant + n_moderate
        avg_psi = float(np.mean([r.psi for r in results])) if results else 0.0
        max_psi_result = max(results, key=lambda r: r.psi) if results else None

        # Recommendation
        if not results:
            recommendation = "insufficient_data"
        elif n_significant > len(results) * self._halt_threshold:
            recommendation = "halt"  # >30% features significantly shifted
        elif n_significant >= RETRAIN_SIGNIFICANT_COUNT:
            recommendation = "retrain"
        elif n_moderate > len(results) * MONITOR_MODERATE_FRACTION:
            recommendation = "monitor"
        else:
            recommendation = "ok"

        # Build structured halt decision for the trading loop
        now_str = datetime.now(timezone.utc).isoformat()
        if recommendation == "halt":
            halt_decision = HaltDecision(
                should_halt=True,
                should_reduce_size=True,
                should_retrain=True,
                reason=f"Critical feature drift: {n_significant}/{len(results)} features "
                       f"significantly shifted (>{self._halt_threshold*100:.0f}% threshold). "
                       f"Max PSI={max_psi_result.psi:.4f} on {max_psi_result.feature_name}."
                       if max_psi_result else "Critical feature drift detected.",
                severity="critical",
                max_psi=max_psi_result.psi if max_psi_result else 0.0,
                shifted_features_count=n_shifted,
                check_timestamp=now_str,
            )
        elif recommendation == "retrain":
            halt_decision = HaltDecision(
                should_halt=False,
                should_reduce_size=True,
                should_retrain=True,
                reason=f"Significant feature drift: {n_significant} features significantly "
                       f"shifted (>={RETRAIN_SIGNIFICANT_COUNT} threshold). Retrain recommended.",
                severity="significant",
                max_psi=max_psi_result.psi if max_psi_result else 0.0,
                shifted_features_count=n_shifted,
                check_timestamp=now_str,
            )
        elif recommendation == "monitor":
            halt_decision = HaltDecision(
                should_halt=False,
                should_reduce_size=True,
                should_retrain=False,
                reason=f"Moderate feature drift: {n_moderate} features moderately shifted. "
                       f"Consider reducing position sizes.",
                severity="moderate",
                max_psi=max_psi_result.psi if max_psi_result else 0.0,
                shifted_features_count=n_shifted,
                check_timestamp=now_str,
            )
        else:
            halt_decision = HaltDecision(
                should_halt=False,
                should_reduce_size=False,
                should_retrain=False,
                reason="No significant feature drift detected." if results else "Insufficient data for shift check.",
                severity="none",
                max_psi=max_psi_result.psi if max_psi_result else 0.0,
                shifted_features_count=n_shifted,
                check_timestamp=now_str,
            )

        self._last_halt_decision = halt_decision
        self._last_check_time = datetime.now(timezone.utc)

        report = OverallShiftReport(
            timestamp=now_str,
            total_features=len(results),
            features_shifted=n_shifted,
            features_moderate=n_moderate,
            features_significant=n_significant,
            avg_psi=avg_psi,
            max_psi=max_psi_result.psi if max_psi_result else 0.0,
            max_psi_feature=max_psi_result.feature_name if max_psi_result else "",
            recommendation=recommendation,
            halt_decision=halt_decision,
            details=results,
        )

        self._shift_history.append(report)
        # Trim history
        if len(self._shift_history) > 100:
            self._shift_history = self._shift_history[-100:]

        # Alert if needed
        if recommendation in ("retrain", "halt") and self._alert_callback:
            try:
                self._alert_callback(report)
            except Exception as e:
                logger.error("Feature shift alert callback failed: %s", e)

        if recommendation != "ok":
            logger.warning(
                "Feature shift detected: %d/%d features shifted (recommendation=%s, avg_psi=%.4f, max_psi=%.4f on %s)",
                n_shifted, len(results), recommendation, avg_psi,
                max_psi_result.psi if max_psi_result else 0.0,
                max_psi_result.feature_name if max_psi_result else "N/A",
            )

        return report

    def get_shift_history(self) -> List[dict]:
        """Return recent shift check history."""
        return [r.as_dict() for r in self._shift_history[-20:]]

    def should_check_now(self, now: Optional[datetime] = None) -> bool:
        """
        Determine if a drift check is due based on the configurable interval.

        Checks run every ``check_interval_minutes`` (default 30) during market
        hours. This replaces the hardcoded 15:45 check time.

        Args:
            now: Current datetime (UTC). If None, uses current time.

        Returns:
            True if a check should be performed now.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # Convert to IST (UTC+5:30) for market hours comparison
        ist_offset = timedelta(hours=5, minutes=30)
        now_ist = now + ist_offset
        current_time = now_ist.time()

        # Only check during market hours
        if current_time < self._market_open or current_time > self._market_close:
            return False

        # Check if enough time has elapsed since last check
        if self._last_check_time is not None:
            elapsed = (now - self._last_check_time).total_seconds()
            if elapsed < self._check_interval_minutes * 60:
                return False

        return True

    def get_last_halt_decision(self) -> Optional[HaltDecision]:
        """
        Return the most recent halt decision for the trading loop to act on.

        The trading loop should call this method to get a structured response:
          - halt_decision.should_halt -> stop all trading
          - halt_decision.should_reduce_size -> reduce position sizes
          - halt_decision.should_retrain -> trigger model retraining pipeline
        """
        return self._last_halt_decision

    def get_check_schedule(self) -> Dict[str, Any]:
        """Return the current check schedule configuration."""
        return {
            "check_interval_minutes": self._check_interval_minutes,
            "market_open": self._market_open.isoformat(),
            "market_close": self._market_close.isoformat(),
            "last_check_time": self._last_check_time.isoformat() if self._last_check_time else None,
            "last_recommendation": self._last_halt_decision.severity if self._last_halt_decision else None,
        }

    def clear_live_buffer(self) -> None:
        """Clear the live feature buffer (e.g., after model retrain)."""
        self._live_buffer.clear()
        self._last_halt_decision = None
        logger.info("Feature shift detector: live buffer cleared")
