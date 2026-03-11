"""
Cross-sectional opportunity ranking engine (Phase 1).
OpportunityScore per symbol; rank; filters and caps (liquidity, spread, sector, correlation).
"""

from .ranker import OpportunityRanker, OpportunityScoreConfig, RankedSymbol

__all__ = ["OpportunityRanker", "OpportunityScoreConfig", "RankedSymbol"]
