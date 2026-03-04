"""
Label engineering: triple-barrier, meta-label, cost-aware, regime-conditioned.
No label leakage; aligned with intraday execution.
"""
from .triple_barrier import TripleBarrierLabeler, TripleBarrierConfig
from .meta_label import MetaLabeler
from .cost_aware import cost_aware_barriers

__all__ = [
    "TripleBarrierLabeler",
    "TripleBarrierConfig",
    "MetaLabeler",
    "cost_aware_barriers",
]
