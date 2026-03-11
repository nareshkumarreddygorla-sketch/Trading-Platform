"""
Phase C: Signal quality scoring.
AlphaQualityScore = w1*IC_mean + w2*IC_stability + w3*Sharpe_OOS + w4*Regime_Robustness
  + w5*Capacity_Score - w6*Turnover_penalty - w7*Slippage_sensitivity
Rank; keep top decile; archive rest.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..validation.validator import ValidationResult


@dataclass
class AlphaQualityScoreConfig:
    w_ic_mean: float = 0.25
    w_ic_stability: float = 0.20
    w_sharpe_oos: float = 0.20
    w_regime_robustness: float = 0.15
    w_capacity_score: float = 0.10
    w_turnover_penalty: float = 0.05
    w_slippage_sensitivity: float = 0.05
    top_decile: bool = True  # keep top 10%
    turnover_ref: float = 0.5  # normalize turnover penalty
    slippage_ref: float = 1.0  # normalize slippage sensitivity


class AlphaQualityScorer:
    """
    Compute AlphaQualityScore for each validated signal; rank descending;
    keep only top decile (or top N); archive others.
    """

    def __init__(self, config: Optional[AlphaQualityScoreConfig] = None):
        self.config = config or AlphaQualityScoreConfig()

    def score_one(
        self,
        validation_result: ValidationResult,
        capacity_score: float = 0.5,
        slippage_sensitivity: float = 0.0,
    ) -> float:
        """Compute AlphaQualityScore for one signal."""
        cfg = self.config
        if validation_result.ic_result is None:
            return -1.0
        ic = validation_result.ic_result
        ic_mean_norm = np.clip(ic.ic_mean * 10.0, -1.0, 1.0)  # scale IC to ~[-1,1]
        ic_stab = ic.ic_stability
        sharpe_norm = np.clip(validation_result.sharpe_oos * 2.0, -1.0, 1.0)
        regime_rob = np.clip(ic.regime_robustness, 0.0, 1.0)
        cap_score = np.clip(capacity_score, 0.0, 1.0)
        turnover_pen = min(1.0, (validation_result.turnover or 0.0) / (cfg.turnover_ref + 1e-8))
        slip_pen = min(1.0, (slippage_sensitivity or 0.0) / (cfg.slippage_ref + 1e-8))
        score = (
            cfg.w_ic_mean * ic_mean_norm
            + cfg.w_ic_stability * ic_stab
            + cfg.w_sharpe_oos * sharpe_norm
            + cfg.w_regime_robustness * regime_rob
            + cfg.w_capacity_score * cap_score
            - cfg.w_turnover_penalty * turnover_pen
            - cfg.w_slippage_sensitivity * slip_pen
        )
        return float(np.clip(score, -1.0, 1.0))

    def rank_and_select(
        self,
        validation_results: List[ValidationResult],
        capacity_scores: Optional[Dict[str, float]] = None,
        slippage_sensitivities: Optional[Dict[str, float]] = None,
    ) -> List[tuple[str, float]]:
        """
        Score each; sort by score descending; return list of (signal_id, score).
        If top_decile: return only top 10% of list; else return all sorted.
        """
        capacity_scores = capacity_scores or {}
        slippage_sensitivities = slippage_sensitivities or {}
        scored: List[tuple[str, float]] = []
        for vr in validation_results:
            if not vr.passed:
                continue
            cap = capacity_scores.get(vr.signal_id, 0.5)
            slip = slippage_sensitivities.get(vr.signal_id, 0.0)
            s = self.score_one(vr, capacity_score=cap, slippage_sensitivity=slip)
            scored.append((vr.signal_id, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        if self.config.top_decile and scored:
            k = max(1, len(scored) // 10)
            return scored[:k]
        return scored
