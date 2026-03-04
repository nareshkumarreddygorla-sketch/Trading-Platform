"""
Phase J: Research automation pipeline.
Data → Hypothesis → Statistical Filter → Backtest → OOS → FDR → Clustering →
Capacity Sim → Shadow → Promote → Monitor → Decay handling.
Zero manual bias; orchestrate existing modules.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import logging

from ..hypothesis import AlphaHypothesisGenerator, HypothesisSpec
from ..validation import StatisticalValidator
from ..validation.validator import ValidationResult
from ..scoring import AlphaQualityScorer
from ..clustering import SignalClustering
from ..capacity import CapacityModel
from ..decay import DecayMonitor
from ..rules import EdgePreservationRules

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    max_candidates_per_run: int = 100
    min_sample_size: int = 500
    fdr_alpha: float = 0.05
    top_decile: bool = True
    run_backtest_fn: Optional[Callable[[str, Dict], Dict]] = None  # (signal_id, params) -> {sharpe, turnover, ...}
    run_shadow_fn: Optional[Callable[[str], None]] = None  # deploy to shadow
    existing_returns: Optional[Dict[str, Any]] = None  # signal_id -> return array


class ResearchPipeline:
    """
    Orchestrate: generate hypotheses → pre-filter → validate (IC, FDR, OOS, WF) →
    score (AlphaQualityScore) → rank → cluster → capacity check →
    (optional) shadow deploy → promotion rule external;
    decay monitor and edge preservation rules applied throughout.
    """

    def __init__(
        self,
        hypothesis_generator: AlphaHypothesisGenerator,
        validator: StatisticalValidator,
        scorer: AlphaQualityScorer,
        clustering: SignalClustering,
        capacity_model: CapacityModel,
        decay_monitor: DecayMonitor,
        preservation_rules: EdgePreservationRules,
        config: Optional[PipelineConfig] = None,
    ):
        self.hypothesis_generator = hypothesis_generator
        self.validator = validator
        self.scorer = scorer
        self.clustering = clustering
        self.capacity_model = capacity_model
        self.decay_monitor = decay_monitor
        self.preservation_rules = preservation_rules
        self.config = config or PipelineConfig()
        self._validated: List[ValidationResult] = []
        self._selected_ids: List[str] = []
        self._quality_scores: Dict[str, float] = {}

    def run_generation(self) -> List[HypothesisSpec]:
        """Phase A: generate candidate hypotheses (no backtest yet)."""
        candidates = self.hypothesis_generator.generate_candidates()
        return candidates[: self.config.max_candidates_per_run]

    def run_validation(
        self,
        candidates: List[HypothesisSpec],
        signal_matrix: Optional[Dict[str, Any]] = None,
        forward_returns: Optional[Any] = None,
        regime_labels: Optional[Any] = None,
        backtest_results: Optional[Dict[str, Dict]] = None,
    ) -> List[ValidationResult]:
        """
        Phase B: validate each candidate.
        signal_matrix: hypothesis_id -> signal array (if pre_filter used);
        forward_returns: array; regime_labels: array;
        backtest_results: signal_id -> {sharpe_oos, turnover, mean_return_gross, n_wf_positive}.
        """
        results: List[ValidationResult] = []
        backtest_results = backtest_results or {}
        import numpy as np
        for c in candidates:
            sig = signal_matrix.get(c.hypothesis_id) if signal_matrix else None
            fwd = forward_returns
            bt = backtest_results.get(c.hypothesis_id, {})
            if sig is None or fwd is None or len(sig) < self.config.min_sample_size:
                vr = ValidationResult(
                    signal_id=c.hypothesis_id,
                    passed=False,
                    reason="no_signal_or_insufficient_data",
                    n_samples=len(sig) if sig is not None else 0,
                    min_wf_cycles_required=self.validator.min_wf_positive_cycles,
                )
            else:
                vr = self.validator.validate_one(
                    signal_id=c.hypothesis_id,
                    signal=np.asarray(sig),
                    forward_return=np.asarray(fwd),
                    turnover=bt.get("turnover", 0.0),
                    regime_labels=regime_labels,
                    mean_return_gross=bt.get("mean_return_gross", 0.0),
                    sharpe_oos=bt.get("sharpe_oos", 0.0),
                    n_wf_positive=bt.get("n_wf_positive", 0),
                )
            # Edge preservation
            if vr.passed and self.config.existing_returns:
                ok, reason = self.preservation_rules.check_all(
                    turnover=vr.turnover,
                    n_wf_positive=vr.n_wf_positive,
                    e_return_after_cost=vr.e_return_after_cost,
                    ic_oos=vr.ic_result.ic_mean if vr.ic_result else 0.0,
                    signal_returns=sig,
                    existing_returns=self.config.existing_returns,
                )
                if not ok:
                    vr.passed = False
                    vr.reason = reason
            results.append(vr)
        self._validated = self.validator.validate_batch_with_fdr(results)
        return self._validated

    def run_scoring(
        self,
        capacity_scores: Optional[Dict[str, float]] = None,
        slippage_sensitivities: Optional[Dict[str, float]] = None,
    ) -> List[tuple[str, float]]:
        """Phase C: score validated signals; return top decile (signal_id, score)."""
        ranked = self.scorer.rank_and_select(
            self._validated,
            capacity_scores=capacity_scores,
            slippage_sensitivities=slippage_sensitivities,
        )
        self._quality_scores = dict(ranked)
        return ranked

    def run_clustering(
        self,
        signal_returns_matrix: Any,
        signal_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Phase D: cluster by correlation; select one per cluster by quality score."""
        if signal_ids is None:
            signal_ids = list(self._quality_scores.keys())
        if not signal_ids or signal_returns_matrix is None:
            return signal_ids
        selected = self.clustering.cluster(
            signal_ids,
            signal_returns_matrix,
            quality_scores=self._quality_scores,
        )
        self._selected_ids = selected
        return selected

    def run_capacity_check(
        self,
        adv_by_signal: Optional[Dict[str, float]] = None,
        e_return_gross_by_signal: Optional[Dict[str, float]] = None,
    ) -> Dict[str, bool]:
        """Phase E: estimate capacity per signal; return signal_id -> passed."""
        passed: Dict[str, bool] = {}
        for sid in self._selected_ids:
            adv = (adv_by_signal or {}).get(sid, 1e9)
            e_ret = (e_return_gross_by_signal or {}).get(sid, 0.0)
            cr = self.capacity_model.estimate_capacity(adv, e_ret)
            passed[sid] = cr.passed
        return passed

    def get_decay_weight_multipliers(self, signal_ids: List[str]) -> Dict[str, float]:
        """Phase G: recommended weight multipliers from decay monitor."""
        return {sid: self.decay_monitor.recommended_weight_multiplier(sid) for sid in signal_ids}
