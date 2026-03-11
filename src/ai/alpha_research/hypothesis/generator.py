"""
Phase A: Alpha hypothesis generation engine.
Systematic variations from templates; avoid combinatorial explosion;
apply statistical pre-filter (univariate IC) before expensive backtest.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .transforms import TransformTemplate, transform_templates


@dataclass
class HypothesisSpec:
    hypothesis_id: str
    name: str
    family: str
    description: str
    params: dict[str, Any]
    feature_refs: list[str]  # base feature names needed to compute
    univariate_ic: float | None = None  # set after pre-filter
    passed_pre_filter: bool = False


def _expand_template(t: TransformTemplate) -> list[dict[str, Any]]:
    """Expand param_ranges into list of param dicts; cap at max_combinations."""
    if not t.param_ranges:
        return [{}]
    keys = list(t.param_ranges.keys())
    from itertools import product

    combos = list(product(*[t.param_ranges[k] for k in keys]))
    out = [dict(zip(keys, c)) for c in combos[: t.max_combinations]]
    return out


def _univariate_ic_simple(signal: np.ndarray, forward_return: np.ndarray) -> float:
    """Rank IC if arrays same length; else 0."""
    if len(signal) != len(forward_return) or len(signal) < 30:
        return 0.0
    signal = np.asarray(signal)
    forward_return = np.asarray(forward_return)
    mask = np.isfinite(signal) & np.isfinite(forward_return)
    if np.sum(mask) < 30:
        return 0.0
    from scipy.stats import spearmanr

    r, _ = spearmanr(signal[mask], forward_return[mask])
    return float(r) if np.isfinite(r) else 0.0


class AlphaHypothesisGenerator:
    """
    Generate candidate hypotheses from templates; expand within max_combinations;
    optional pre-filter: compute univariate IC on provided (signal, forward_return)
    and keep only those with |IC| >= min_univariate_ic.
    """

    def __init__(self, min_univariate_ic: float = 0.01, max_total_candidates: int = 200):
        self.min_univariate_ic = min_univariate_ic
        self.max_total_candidates = max_total_candidates
        self._templates = transform_templates()

    def generate_candidates(self) -> list[HypothesisSpec]:
        """Generate all candidate specs from templates (no pre-filter)."""
        candidates: list[HypothesisSpec] = []
        for t in self._templates:
            param_list = _expand_template(t)
            for i, params in enumerate(param_list):
                if len(candidates) >= self.max_total_candidates:
                    break
                hid = f"{t.name}_{t.family.value}_{i}"
                candidates.append(
                    HypothesisSpec(
                        hypothesis_id=hid,
                        name=t.name,
                        family=t.family.value,
                        description=t.description,
                        params=params,
                        feature_refs=_feature_refs_for(t),
                    )
                )
        return candidates[: self.max_total_candidates]

    def pre_filter(
        self,
        candidates: list[HypothesisSpec],
        signal_fn_by_id: dict[str, Any] | None = None,
        forward_returns: np.ndarray | None = None,
    ) -> list[HypothesisSpec]:
        """
        If signal_fn_by_id maps hypothesis_id -> array (signal), and forward_returns given,
        compute univariate IC; set passed_pre_filter = True only if |IC| >= min_univariate_ic.
        If signal_fn_by_id is None, skip IC and return all (caller does backtest later).
        """
        if signal_fn_by_id is None or forward_returns is None:
            return candidates
        out = []
        for c in candidates:
            sig = signal_fn_by_id.get(c.hypothesis_id)
            if sig is None:
                c.univariate_ic = None
                c.passed_pre_filter = True
                out.append(c)
                continue
            ic = _univariate_ic_simple(np.asarray(sig), forward_returns)
            c.univariate_ic = ic
            c.passed_pre_filter = abs(ic) >= self.min_univariate_ic
            if c.passed_pre_filter:
                out.append(c)
        return out


def _feature_refs_for(t: TransformTemplate) -> list[str]:
    """Infer base feature refs from template name/family."""
    refs = []
    if "return" in t.description or t.family.value == "price":
        refs.extend(["returns_1m", "returns_5m", "rolling_vol_20"])
    if "ofi" in t.name or "volume" in t.name or t.family.value == "microstructure":
        refs.extend(["order_flow_imbalance", "volume_delta"])
    if "sector" in t.name or "dispersion" in t.name or t.family.value == "cross_sectional":
        refs.extend(["returns_1m", "sector_momentum_rank"])
    if t.family.value == "regime_conditioned":
        refs.extend(["vol_cluster_20", "trend_strength_index"])
    if not refs:
        refs = ["returns_1m"]
    return refs
