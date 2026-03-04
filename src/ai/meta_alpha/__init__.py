"""
Meta-alpha layer: predicts when primary model is wrong, when confidence is inflated,
when regime will flip. Trained on historical errors, miscalibration, regime transitions.
"""
from .predictor import MetaAlphaPredictor, MetaAlphaOutput

__all__ = ["MetaAlphaPredictor", "MetaAlphaOutput"]
