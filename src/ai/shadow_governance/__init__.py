"""
Phase 5: Shadow model governance engine.
Production | Candidate -> Shadow (live predictions, no execution) -> Promote / Rollback;
registry metadata: model_id, version, training_window, metrics, stability, promotion_ts.
"""
from .governance import ShadowModelGovernance, ModelMetadata, ShadowResult

__all__ = ["ShadowModelGovernance", "ModelMetadata", "ShadowResult"]
