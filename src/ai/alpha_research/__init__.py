"""
Alpha Research & Edge Discovery Engine.
Automated hypothesis generation, statistical validation, quality scoring,
clustering, capacity modeling, decay monitoring, edge preservation, pipeline.
"""

from .capacity import CapacityModel, CapacityResult
from .clustering import ClusterConfig, SignalClustering
from .decay import DecayConfig, DecayMonitor
from .hypothesis import AlphaHypothesisGenerator, HypothesisSpec
from .pipeline import PipelineConfig, ResearchPipeline
from .rules import EdgePreservationRules
from .scoring import AlphaQualityScoreConfig, AlphaQualityScorer
from .validation import ICResult, StatisticalValidator, ValidationResult

__all__ = [
    "AlphaHypothesisGenerator",
    "HypothesisSpec",
    "StatisticalValidator",
    "ICResult",
    "ValidationResult",
    "AlphaQualityScorer",
    "AlphaQualityScoreConfig",
    "SignalClustering",
    "ClusterConfig",
    "CapacityModel",
    "CapacityResult",
    "DecayMonitor",
    "DecayConfig",
    "EdgePreservationRules",
    "ResearchPipeline",
    "PipelineConfig",
]
