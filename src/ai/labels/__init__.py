"""
Label engineering: triple-barrier, meta-label, cost-aware, regime-conditioned.
No label leakage; aligned with intraday execution.
"""

from .cost_aware import cost_aware_barriers
from .meta_label import MetaLabeler
from .triple_barrier import TripleBarrierConfig, TripleBarrierLabeler

__all__ = [
    "TripleBarrierLabeler",
    "TripleBarrierConfig",
    "MetaLabeler",
    "cost_aware_barriers",
]
