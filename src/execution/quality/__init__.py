"""
Phase 8: Execution quality feedback loop.
Realized vs expected slippage, rejection rate, partial fill rate, latency;
feed into position sizing, strategy disable, broker failover.
"""
from .tracker import ExecutionQualityTracker, QualityMetrics

__all__ = ["ExecutionQualityTracker", "QualityMetrics"]
