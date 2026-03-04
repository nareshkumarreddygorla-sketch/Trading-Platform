"""
Phase B: False Discovery Rate (Benjamini-Hochberg), permutation test for IC.
"""
from typing import List, Optional, Tuple

import numpy as np


def _ic_rank(signal: np.ndarray, forward_return: np.ndarray) -> float:
    """Spearman correlation (avoid circular import)."""
    from .ic import ic_rank
    return ic_rank(signal, forward_return)


def fdr_benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Benjamini-Hochberg: order p-values; find largest k with p_(k) <= (k/m)*alpha;
    reject hypotheses 1..k (all indices with p <= threshold). Returns list of bool (True = reject H0).
    """
    if not p_values:
        return []
    m = len(p_values)
    order = np.argsort(p_values)
    p_sorted = np.array(p_values)[order]
    reject = np.zeros(m, dtype=bool)
    for k in range(m - 1, -1, -1):
        if p_sorted[k] <= (k + 1) / m * alpha:
            for j in range(k + 1):
                reject[order[j]] = True
            break
    return list(reject)


def permutation_test_ic(
    signal: np.ndarray,
    forward_return: np.ndarray,
    n_permutations: int = 500,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Observed IC (rank); permute forward_return; compute IC under null.
    p-value = fraction of permutations with |IC_perm| >= |IC_obs|.
    Returns (ic_observed, p_value). Optional seed for reproducibility.
    """
    ic_obs = _ic_rank(signal, forward_return)
    if not np.isfinite(ic_obs):
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    count = 0
    ret = np.asarray(forward_return)
    for _ in range(n_permutations):
        perm = rng.permutation(ret)
        ic_p = _ic_rank(signal, perm)
        if abs(ic_p) >= abs(ic_obs):
            count += 1
    p_val = (count + 1) / (n_permutations + 1)
    return float(ic_obs), float(p_val)
