"""
Phase D: Signal clustering & diversification.
Correlation matrix / distance → hierarchical clustering → one strongest per cluster.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ClusterConfig:
    max_correlation: float = 0.7  # merge if corr > threshold (or use distance)
    min_cluster_size: int = 1
    method: str = "average"  # linkage: average, single, complete


def _hierarchical_clusters(corr_matrix: np.ndarray, threshold: float) -> list[list[int]]:
    """Cluster indices by correlation; merge if correlation > threshold (1 - distance)."""
    n = corr_matrix.shape[0]
    if n <= 1:
        return [[i] for i in range(n)]
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        # distance = 1 - |corr|; so high corr -> low distance
        dist = 1.0 - np.abs(np.clip(corr_matrix, -1, 1))
        np.fill_diagonal(dist, 0)
        cond = squareform(dist, checks=False)
        Z = linkage(cond, method="average")
        # threshold for cluster merge: ~1 - max_correlation
        t = 1.0 - threshold
        labels = fcluster(Z, t, criterion="distance")
        clusters: dict[int, list[int]] = {}
        for i, l in enumerate(labels):
            clusters.setdefault(int(l), []).append(i)
        return list(clusters.values())
    except Exception:
        return [[i] for i in range(n)]


class SignalClustering:
    """
    Build correlation matrix of signal returns (or signal values);
    hierarchical cluster; from each cluster select one strongest alpha (by quality score).
    """

    def __init__(self, config: ClusterConfig | None = None):
        self.config = config or ClusterConfig()

    def cluster(
        self,
        signal_ids: list[str],
        signal_returns: np.ndarray,
        quality_scores: dict[str, float] | None = None,
    ) -> list[str]:
        """
        signal_returns: shape (n_signals, T) — each row is a signal's return series (or signal series).
        Compute correlation matrix; cluster; from each cluster pick signal_id with highest quality_score.
        Returns list of selected signal_ids (one per cluster).
        """
        if signal_returns.shape[0] != len(signal_ids) or signal_returns.shape[0] == 0:
            return list(signal_ids)
        if signal_returns.shape[1] < 2:
            return list(signal_ids)
        quality_scores = quality_scores or {sid: 0.0 for sid in signal_ids}
        corr = np.corrcoef(signal_returns)
        if corr is None or not np.isfinite(corr).all():
            return list(signal_ids)
        np.fill_diagonal(corr, 1.0)
        clusters = _hierarchical_clusters(corr, self.config.max_correlation)
        selected: list[str] = []
        for c in clusters:
            if not c:
                continue
            best_idx = max(c, key=lambda i: quality_scores.get(signal_ids[i], -1e9))
            selected.append(signal_ids[best_idx])
        return selected if selected else list(signal_ids)
