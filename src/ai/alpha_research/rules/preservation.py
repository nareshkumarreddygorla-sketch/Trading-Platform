"""
Phase I: Edge preservation rules (hard constraints).
1) No alpha optimized on Sharpe alone.
2) All signals pass cost-aware validation.
3) No extreme turnover unless justified.
4) All signals survive >= 3 walk-forward cycles.
5) Correlation with existing alpha below threshold.
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class EdgePreservationRules:
    """
    Hard constraints applied at validation and promotion.
    No alpha may be optimized directly on Sharpe alone; cost-aware; turnover cap;
    min 3 WF cycles; correlation with existing < max_correlation.
    """

    max_turnover: float = 1.0  # daily turnover cap unless justified
    min_wf_cycles: int = 3
    max_correlation_with_existing: float = 0.8
    require_cost_positive: bool = True
    require_oos_positive_ic: bool = True

    def check_turnover(self, turnover: float, justified: bool = False) -> bool:
        """True if turnover <= max or justified."""
        if justified:
            return True
        return turnover <= self.max_turnover

    def check_wf_cycles(self, n_positive_cycles: int) -> bool:
        """True if >= min_wf_cycles."""
        return n_positive_cycles >= self.min_wf_cycles

    def check_correlation_with_existing(
        self,
        signal_returns: np.ndarray,
        existing_returns: Dict[str, np.ndarray],
    ) -> bool:
        """True if correlation with every existing alpha < max_correlation_with_existing."""
        for sid, ret in existing_returns.items():
            if len(ret) != len(signal_returns):
                continue
            corr = np.corrcoef(signal_returns, ret)[0, 1]
            if np.isfinite(corr) and abs(corr) >= self.max_correlation_with_existing:
                return False
        return True

    def check_all(
        self,
        turnover: float,
        n_wf_positive: int,
        e_return_after_cost: float,
        ic_oos: float,
        signal_returns: Optional[np.ndarray] = None,
        existing_returns: Optional[Dict[str, np.ndarray]] = None,
        turnover_justified: bool = False,
    ) -> tuple[bool, str]:
        """
        Return (passed, reason). Passed only if all checks pass.
        """
        if self.require_cost_positive and e_return_after_cost <= 0:
            return False, "cost_positive"
        if self.require_oos_positive_ic and ic_oos <= 0:
            return False, "oos_positive_ic"
        if not self.check_turnover(turnover, turnover_justified):
            return False, "turnover"
        if not self.check_wf_cycles(n_wf_positive):
            return False, "wf_cycles"
        if signal_returns is not None and existing_returns:
            if not self.check_correlation_with_existing(signal_returns, existing_returns):
                return False, "correlation_existing"
        return True, "ok"
