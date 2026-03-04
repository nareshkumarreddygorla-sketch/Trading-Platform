"""
Alpha Research & Edge Discovery Engine.
Automated hypothesis generation, statistical validation, quality scoring,
clustering, capacity modeling, decay monitoring, edge preservation, pipeline.
"""
from .hypothesis import AlphaHypothesisGenerator, HypothesisSpec
from .validation import StatisticalValidator, ICResult, ValidationResult
from .scoring import AlphaQualityScorer, AlphaQualityScoreConfig
from .clustering import SignalClustering, ClusterConfig
from .capacity import CapacityModel, CapacityResult
from .decay import DecayMonitor, DecayConfig
from .rules import EdgePreservationRules
from .pipeline import ResearchPipeline, PipelineConfig

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
