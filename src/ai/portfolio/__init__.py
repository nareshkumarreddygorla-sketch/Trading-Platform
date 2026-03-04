"""
Correlation-aware portfolio optimizer (Phase 2).
Rolling correlation, MCR, heat, concentration; risk parity, vol target, correlation penalty;
max gross/net/sector/cluster exposure; integrate with effective_equity.
"""
from .optimizer import CorrelationOptimizer, PortfolioWeights, OptimizerConfig

__all__ = ["CorrelationOptimizer", "PortfolioWeights", "OptimizerConfig"]
