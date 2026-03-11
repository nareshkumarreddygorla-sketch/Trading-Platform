"""
Phase 1: Cross-sectional opportunity ranking.
OpportunityScore = calibrated_prob + E[r]_adj + meta_conf - vol_pen - spread_pen - liq_pen - noise_pen.
Rank descending; select top N with score threshold, liquidity, spread, min risk-adj return;
sector cap, correlation cluster cap, max concurrent signals.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class OpportunityScoreConfig:
    w_cal_prob: float = 0.25
    w_expected_return_adj: float = 0.25
    w_meta_confidence: float = 0.15
    w_regime: float = 0.10
    w_vol_penalty: float = 0.10
    w_spread_penalty: float = 0.08
    w_liquidity_penalty: float = 0.05
    w_noise_penalty: float = 0.02
    score_threshold: float = 0.2
    min_adv_ratio: float = 0.01  # min symbol_volume / universe_avg_volume
    max_spread_bps: float = 50.0
    min_risk_adj_return: float = 0.1  # min E[r]/sigma
    max_sector_pct: float = 25.0
    max_correlated_cluster_pct: float = 40.0
    max_concurrent_signals: int = 20
    ref_vol: float = 0.02
    ref_spread_bps: float = 10.0


@dataclass
class RankedSymbol:
    symbol: str
    score: float
    calibrated_prob: float
    expected_return: float
    volatility: float
    sector: str = ""
    cluster_id: int = 0
    passed_filters: bool = True
    reason: str = ""


def _calibrated_prob_direction(prob_up: float) -> float:
    """Strength of direction 0..1."""
    return abs(prob_up - 0.5) * 2.0


def _expected_return_risk_adj(expected_return: float, vol: float, eps: float = 1e-8) -> float:
    """E[r] / (sigma + eps), normalized to ~0..1 for scoring."""
    if vol <= 0:
        return 0.0
    x = expected_return / (vol + eps)
    return float(np.clip(x * 10.0, 0.0, 1.0))  # scale for typical E[r]/sigma


def _vol_penalty(vol: float, ref_vol: float) -> float:
    return min(1.0, vol / (ref_vol + 1e-8))


def _spread_penalty(spread_bps: float, ref_bps: float) -> float:
    return min(1.0, spread_bps / (ref_bps + 1e-8))


def _liquidity_penalty(adv_ratio: float) -> float:
    """1 - min(1, adv_ratio); low liquidity -> high penalty."""
    return 1.0 - min(1.0, adv_ratio)


class OpportunityRanker:
    """
    Evaluate all tradable symbols; compute OpportunityScore; rank; select top N
    with filters and sector/correlation caps.
    """

    def __init__(self, config: OpportunityScoreConfig | None = None):
        self.config = config or OpportunityScoreConfig()

    def score_one(
        self,
        symbol: str,
        calibrated_prob_up: float,
        expected_return: float,
        volatility: float,
        meta_confidence: float,
        regime_weight: float,
        spread_bps: float = 10.0,
        adv_ratio: float = 1.0,
        microstructure_noise: float = 0.0,
        sector: str = "",
        cluster_id: int = 0,
    ) -> float:
        """Compute OpportunityScore for one symbol."""
        cfg = self.config
        cal_dir = _calibrated_prob_direction(calibrated_prob_up)
        ret_adj = _expected_return_risk_adj(expected_return, volatility)
        vol_pen = _vol_penalty(volatility, cfg.ref_vol)
        spread_pen = _spread_penalty(spread_bps, cfg.ref_spread_bps)
        liq_pen = _liquidity_penalty(adv_ratio)
        noise_pen = min(1.0, microstructure_noise * 100.0)  # proxy scale
        score = (
            cfg.w_cal_prob * cal_dir
            + cfg.w_expected_return_adj * ret_adj
            + cfg.w_meta_confidence * meta_confidence
            + cfg.w_regime * regime_weight
            - cfg.w_vol_penalty * vol_pen
            - cfg.w_spread_penalty * spread_pen
            - cfg.w_liquidity_penalty * liq_pen
            - cfg.w_noise_penalty * noise_pen
        )
        return float(np.clip(score, -1.0, 1.0))

    def rank(
        self,
        symbol_data: list[dict],
        current_sector_exposure: dict[str, float] | None = None,
        current_cluster_exposure: dict[int, float] | None = None,
    ) -> list[RankedSymbol]:
        """
        symbol_data: list of dicts with keys symbol, calibrated_prob_up, expected_return,
          volatility, meta_confidence, regime_weight, spread_bps, adv_ratio, microstructure_noise,
          sector, cluster_id.
        Returns list of RankedSymbol sorted by score descending; applies filters and caps.
        """
        cfg = self.config
        current_sector_exposure = current_sector_exposure or {}
        current_cluster_exposure = current_cluster_exposure or {}
        scored: list[RankedSymbol] = []
        for d in symbol_data:
            score = self.score_one(
                symbol=d.get("symbol", ""),
                calibrated_prob_up=d.get("calibrated_prob_up", 0.5),
                expected_return=d.get("expected_return", 0.0),
                volatility=d.get("volatility", 0.02),
                meta_confidence=d.get("meta_confidence", 0.5),
                regime_weight=d.get("regime_weight", 0.5),
                spread_bps=d.get("spread_bps", 10.0),
                adv_ratio=d.get("adv_ratio", 1.0),
                microstructure_noise=d.get("microstructure_noise", 0.0),
                sector=d.get("sector", ""),
                cluster_id=d.get("cluster_id", 0),
            )
            rs = RankedSymbol(
                symbol=d.get("symbol", ""),
                score=score,
                calibrated_prob=d.get("calibrated_prob_up", 0.5),
                expected_return=d.get("expected_return", 0.0),
                volatility=d.get("volatility", 0.02),
                sector=d.get("sector", ""),
                cluster_id=d.get("cluster_id", 0),
                passed_filters=True,
                reason="",
            )
            if score < cfg.score_threshold:
                rs.passed_filters = False
                rs.reason = "score_below_threshold"
            if d.get("adv_ratio", 0) < cfg.min_adv_ratio:
                rs.passed_filters = False
                rs.reason = rs.reason or "liquidity"
            if d.get("spread_bps", 999) > cfg.max_spread_bps:
                rs.passed_filters = False
                rs.reason = rs.reason or "spread"
            vol = d.get("volatility", 1e-8)
            ret_adj = d.get("expected_return", 0.0) / (vol + 1e-8)
            if ret_adj < cfg.min_risk_adj_return:
                rs.passed_filters = False
                rs.reason = rs.reason or "min_risk_adj_return"
            scored.append(rs)
        scored.sort(key=lambda x: x.score, reverse=True)
        # Apply caps: max concurrent, sector, cluster
        out: list[RankedSymbol] = []
        sector_used: dict[str, float] = dict(current_sector_exposure)
        cluster_used: dict[int, float] = dict(current_cluster_exposure)
        for rs in scored:
            if not rs.passed_filters:
                continue
            if len(out) >= cfg.max_concurrent_signals:
                break
            sector = rs.sector or "_"
            cluster = rs.cluster_id
            sector_used[sector] = sector_used.get(sector, 0) + 1.0  # simple count; could be weight
            cluster_used[cluster] = cluster_used.get(cluster, 0) + 1.0
            if sector_used[sector] > cfg.max_sector_pct / 5.0:  # rough: 5% per name -> 25% for 5 names
                continue
            if cluster_used[cluster] > cfg.max_correlated_cluster_pct / 5.0:
                continue
            out.append(rs)
        return out
