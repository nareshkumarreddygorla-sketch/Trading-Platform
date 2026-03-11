"""
Execution quality feedback loop (enhanced).
Realized vs expected slippage, implementation shortfall, benchmark comparison,
per-algo scoring, daily reports; feed into position sizing, strategy disable, broker failover.
"""

from .tracker import (
    AlgoQualityScore,
    BenchmarkComparison,
    DailyExecutionReport,
    ExecutionQualityTracker,
    ImplementationShortfallResult,
    QualityMetrics,
    SlippageMeasurement,
)

__all__ = [
    "AlgoQualityScore",
    "BenchmarkComparison",
    "DailyExecutionReport",
    "ExecutionQualityTracker",
    "ImplementationShortfallResult",
    "QualityMetrics",
    "SlippageMeasurement",
]
