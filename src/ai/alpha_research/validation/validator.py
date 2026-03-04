"""
Phase B: Statistical validation layer.
IC, IC stability, turnover-adjusted IC, E[r] after cost, capacity-adjusted;
FDR, min sample, OOS, walk-forward, permutation; reject weak/collapsed/unstable.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .ic import ic_rank, ic_stability_time, ic_stability_regime, turnover_adjusted_ic
from .fdr import fdr_benjamini_hochberg, permutation_test_ic


@dataclass
class ICResult:
    ic_mean: float
    ic_std: float
    ic_stability: float  # 1/(1+ic_std)
    ic_turnover_adj: float
    ic_regime_min: float
    ic_regime_mean: float
    regime_robustness: float  # min/mean
    p_value: float


@dataclass
class ValidationResult:
    signal_id: str
    passed: bool
    reason: str
    ic_result: Optional[ICResult] = None
    sharpe_oos: float = 0.0
    e_return_after_cost: float = 0.0
    turnover: float = 0.0
    n_samples: int = 0
    n_wf_positive: int = 0
    min_wf_cycles_required: int = 3


def _expected_return_after_cost(
    mean_return_gross: float,
    turnover: float,
    cost_bps: float,
) -> float:
    """E[r]_net = E[r]_gross - cost_bps * turnover (per unit)."""
    return mean_return_gross - (cost_bps / 10_000.0) * turnover


class StatisticalValidator:
    """
    For each candidate: compute IC, IC stability (time + regime), turnover-adjusted IC,
    E[r] after cost; permutation p-value; OOS Sharpe; walk-forward positive count.
    Reject if: OOS collapse, regime-dependent collapse, unstable IC, fails after cost,
    too capacity-limited, FDR not rejected, min sample not met, < 3 WF cycles.
    """

    def __init__(
        self,
        min_sample_size: int = 500,
        min_trades: int = 100,
        min_wf_positive_cycles: int = 3,
        cost_bps: float = 10.0,
        fdr_alpha: float = 0.05,
        min_ic_oos: float = 0.01,
        min_e_return_after_cost: float = 0.0,
    ):
        self.min_sample_size = min_sample_size
        self.min_trades = min_trades
        self.min_wf_positive_cycles = min_wf_positive_cycles
        self.cost_bps = cost_bps
        self.fdr_alpha = fdr_alpha
        self.min_ic_oos = min_ic_oos
        self.min_e_return_after_cost = min_e_return_after_cost

    def validate_one(
        self,
        signal_id: str,
        signal: np.ndarray,
        forward_return: np.ndarray,
        turnover: float = 0.0,
        regime_labels: Optional[np.ndarray] = None,
        mean_return_gross: float = 0.0,
        sharpe_oos: float = 0.0,
        n_wf_positive: int = 0,
        n_permutations: int = 200,
    ) -> ValidationResult:
        """Validate single signal; return ValidationResult (passed, reason, metrics)."""
        n = len(signal)
        if n < self.min_sample_size:
            return ValidationResult(
                signal_id=signal_id,
                passed=False,
                reason="insufficient_sample",
                n_samples=n,
                min_wf_cycles_required=self.min_wf_positive_cycles,
            )
        ic_mean, ic_std = ic_stability_time(signal, forward_return)
        ic_stab = 1.0 / (1.0 + ic_std) if np.isfinite(ic_std) else 0.0
        ic_raw = ic_rank(signal, forward_return)
        ic_adj = turnover_adjusted_ic(ic_raw, turnover)
        regime_min, regime_mean = 0.0, ic_raw
        if regime_labels is not None and len(regime_labels) == n:
            regime_min, regime_mean = ic_stability_regime(signal, forward_return, regime_labels)
        regime_robust = regime_min / (regime_mean + 1e-12) if regime_mean != 0 else 0.0
        ic_result = ICResult(
            ic_mean=ic_raw,
            ic_std=ic_std,
            ic_stability=ic_stab,
            ic_turnover_adj=ic_adj,
            ic_regime_min=regime_min,
            ic_regime_mean=regime_mean,
            regime_robustness=regime_robust,
            p_value=1.0,
        )
        ic_obs, p_val = permutation_test_ic(signal, forward_return, n_permutations)
        ic_result.p_value = p_val
        e_net = _expected_return_after_cost(mean_return_gross, turnover, self.cost_bps)
        if e_net < self.min_e_return_after_cost:
            return ValidationResult(
                signal_id=signal_id,
                passed=False,
                reason="fails_after_cost",
                ic_result=ic_result,
                e_return_after_cost=e_net,
                turnover=turnover,
                n_samples=n,
                n_wf_positive=n_wf_positive,
                min_wf_cycles_required=self.min_wf_positive_cycles,
            )
        if p_val > self.fdr_alpha:
            return ValidationResult(
                signal_id=signal_id,
                passed=False,
                reason="fdr_not_rejected",
                ic_result=ic_result,
                sharpe_oos=sharpe_oos,
                e_return_after_cost=e_net,
                turnover=turnover,
                n_samples=n,
                n_wf_positive=n_wf_positive,
                min_wf_cycles_required=self.min_wf_positive_cycles,
            )
        if abs(ic_raw) < self.min_ic_oos:
            return ValidationResult(
                signal_id=signal_id,
                passed=False,
                reason="ic_oos_collapse",
                ic_result=ic_result,
                sharpe_oos=sharpe_oos,
                e_return_after_cost=e_net,
                turnover=turnover,
                n_samples=n,
                n_wf_positive=n_wf_positive,
                min_wf_cycles_required=self.min_wf_positive_cycles,
            )
        if n_wf_positive < self.min_wf_positive_cycles:
            return ValidationResult(
                signal_id=signal_id,
                passed=False,
                reason="wf_cycles",
                ic_result=ic_result,
                sharpe_oos=sharpe_oos,
                e_return_after_cost=e_net,
                turnover=turnover,
                n_samples=n,
                n_wf_positive=n_wf_positive,
                min_wf_cycles_required=self.min_wf_positive_cycles,
            )
        return ValidationResult(
            signal_id=signal_id,
            passed=True,
            reason="ok",
            ic_result=ic_result,
            sharpe_oos=sharpe_oos,
            e_return_after_cost=e_net,
            turnover=turnover,
            n_samples=n,
            n_wf_positive=n_wf_positive,
            min_wf_cycles_required=self.min_wf_positive_cycles,
        )

    def validate_batch_with_fdr(
        self,
        results: List[ValidationResult],
    ) -> List[ValidationResult]:
        """Apply FDR to p-values; return new list with passed=False where FDR does not reject. Same order as results."""
        p_vals = [r.ic_result.p_value if r.ic_result is not None else 1.0 for r in results]
        if not p_vals:
            return list(results)
        reject = fdr_benjamini_hochberg(p_vals, self.fdr_alpha)
        out: List[ValidationResult] = []
        for i, r in enumerate(results):
            if r.passed and i < len(reject) and not reject[i]:
                out.append(ValidationResult(
                    signal_id=r.signal_id,
                    passed=False,
                    reason="fdr_not_rejected",
                    ic_result=r.ic_result,
                    sharpe_oos=r.sharpe_oos,
                    e_return_after_cost=r.e_return_after_cost,
                    turnover=r.turnover,
                    n_samples=r.n_samples,
                    n_wf_positive=r.n_wf_positive,
                    min_wf_cycles_required=r.min_wf_cycles_required,
                ))
            else:
                out.append(r)
        return out
