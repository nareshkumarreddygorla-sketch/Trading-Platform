#!/usr/bin/env python3
"""
Run the Alpha Research & Edge Discovery pipeline end-to-end.
Uses synthetic data if no feature store / backtest; otherwise wire run_backtest_fn.
Usage: PYTHONPATH=. python scripts/run_alpha_research.py
"""
import logging
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def make_synthetic_data(n_bars: int = 600, n_signals: int = 20, seed: int = 42) -> tuple:
    """Synthetic signal and forward return for pipeline demo. Autocorrelated returns so IC > 0."""
    rng = np.random.default_rng(seed)
    # Autocorrelated returns: r[t] = 0.2*r[t-1] + eps so signal[t]=r[t-1] predicts r[t]
    eps = rng.standard_normal(n_bars) * 0.01
    forward_returns = np.zeros(n_bars)
    forward_returns[0] = eps[0]
    for t in range(1, n_bars):
        forward_returns[t] = 0.2 * forward_returns[t - 1] + eps[t]
    forward_returns = np.asarray(forward_returns, dtype=float)
    signals = {}
    for i in range(n_signals):
        # Signal[t] = forward_return[t-1] + noise -> predicts forward_return[t] (positive IC)
        lag_return = np.roll(forward_returns, 1)
        lag_return[0] = 0
        noise = rng.standard_normal(n_bars) * 0.003
        s = 0.4 * lag_return + noise
        signals[f"syn_signal_{i}"] = np.asarray(s, dtype=float)
    return signals, forward_returns


def main() -> None:
    from src.ai.alpha_research import (
        AlphaHypothesisGenerator,
        StatisticalValidator,
        AlphaQualityScorer,
        AlphaQualityScoreConfig,
        SignalClustering,
        ClusterConfig,
        CapacityModel,
        DecayMonitor,
        DecayConfig,
        EdgePreservationRules,
        ResearchPipeline,
        PipelineConfig,
    )

    # 1) Generate candidates
    generator = AlphaHypothesisGenerator(min_univariate_ic=0.005, max_total_candidates=50)
    candidates = generator.generate_candidates()
    logger.info("Generated %d hypothesis candidates", len(candidates))

    # 2) Synthetic data (replace with real feature store + forward returns in production)
    n_sig = min(25, len(candidates))
    signal_matrix, forward_returns = make_synthetic_data(n_bars=600, n_signals=n_sig)
    # Map each of first N candidates to a synthetic signal (same order as candidates)
    keys = list(signal_matrix.keys())
    for i, c in enumerate(candidates[: len(keys)]):
        signal_matrix[c.hypothesis_id] = np.asarray(signal_matrix[keys[i]], dtype=float)
    # Sanity: IC of first synthetic signal vs forward_returns
    from src.ai.alpha_research.validation.ic import ic_rank
    first_sig = signal_matrix.get(keys[0])
    if first_sig is not None and len(first_sig) == len(forward_returns):
        ic0 = ic_rank(np.asarray(first_sig), forward_returns)
        logger.info("Sanity IC(syn_signal_0, forward_returns)=%.4f", ic0)

    # 3) Validator, scorer, clustering, capacity, decay, rules
    validator = StatisticalValidator(
        min_sample_size=100,
        min_wf_positive_cycles=1,
        fdr_alpha=0.15,
        min_ic_oos=0.008,
    )
    scorer = AlphaQualityScorer(AlphaQualityScoreConfig(top_decile=True))
    clustering = SignalClustering(ClusterConfig(max_correlation=0.85))
    capacity_model = CapacityModel(target_capital=1e6, max_participation_pct=10.0)
    decay_monitor = DecayMonitor(DecayConfig(rolling_window=20))
    preservation_rules = EdgePreservationRules(min_wf_cycles=1, max_turnover=1.0)

    pipeline = ResearchPipeline(
        hypothesis_generator=generator,
        validator=validator,
        scorer=scorer,
        clustering=clustering,
        capacity_model=capacity_model,
        decay_monitor=decay_monitor,
        preservation_rules=preservation_rules,
        config=PipelineConfig(
            max_candidates_per_run=30,
            min_sample_size=100,
            fdr_alpha=0.10,
            top_decile=True,
        ),
    )

    # 4) Run pipeline — use same candidates for validation (so signal_matrix keys match)
    candidates = pipeline.run_generation()
    # Ensure every candidate in this run has a signal (reuse keys by index)
    for i, c in enumerate(candidates):
        if c.hypothesis_id not in signal_matrix and i < len(keys):
            signal_matrix[c.hypothesis_id] = np.asarray(signal_matrix[keys[i % len(keys)]], dtype=float)
    # Stub backtest: fake sharpe, turnover, n_wf_positive for candidates that have signals
    backtest_results = {}
    rng = np.random.default_rng(43)
    for c in candidates:
        if c.hypothesis_id in signal_matrix:
            turnover = 0.15 + float(rng.random()) * 0.25
            mean_gross = 0.0003 + float(rng.random()) * 0.0003
            backtest_results[c.hypothesis_id] = {
                "sharpe_oos": 0.4 + float(rng.random()) * 0.5,
                "turnover": turnover,
                "mean_return_gross": mean_gross,
                "n_wf_positive": 3,
            }

    validated = pipeline.run_validation(
        candidates,
        signal_matrix=signal_matrix,
        forward_returns=forward_returns,
        backtest_results=backtest_results,
    )
    passed = sum(1 for v in validated if v.passed)
    logger.info("Validation: %d passed, %d total", passed, len(validated))
    if passed == 0 and validated:
        with_ic = [v for v in validated if v.ic_result is not None]
        logger.info("Validated with ic_result: %d", len(with_ic))
        if with_ic:
            v0 = with_ic[0]
            ic = v0.ic_result
            logger.info("Sample: %s reason=%s ic_mean=%.4f p_value=%.4f e_net=%.6f n_wf=%d",
                v0.signal_id, v0.reason, ic.ic_mean, ic.p_value, v0.e_return_after_cost, v0.n_wf_positive)

    ranked = pipeline.run_scoring(
        capacity_scores={v.signal_id: 0.6 for v in validated if v.passed},
    )
    logger.info("Top decile: %d signals", len(ranked))
    for sid, score in ranked[:10]:
        logger.info("  %s -> %.4f", sid, score)

    if len(ranked) >= 2:
        # Build signal returns matrix (n_signals x T) for clustering
        ids = [r[0] for r in ranked]
        mat = np.array([signal_matrix.get(sid, np.zeros(len(forward_returns))) for sid in ids])
        if mat.shape[0] == len(ids) and mat.shape[1] >= 100:
            selected = pipeline.run_clustering(mat, signal_ids=ids)
            logger.info("After clustering: %d signals -> %s", len(selected), selected[:5])
        capacity_passed = pipeline.run_capacity_check(
            adv_by_signal={sid: 1e8 for sid in ids},
            e_return_gross_by_signal={sid: 0.0002 for sid in ids},
        )
        logger.info("Capacity check: %s", capacity_passed)

    logger.info("Alpha research pipeline run complete.")


if __name__ == "__main__":
    main()
