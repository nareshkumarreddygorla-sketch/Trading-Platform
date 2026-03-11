"""
Phase 4: Regime-specialist model architecture.
Trend, mean reversion, high vol breakout, low liquidity defensive;
activate by regime; blend outputs with regime weights.
"""

from .registry import RegimeSpecialist, RegimeSpecialistRegistry, SpecialistOutput

__all__ = ["RegimeSpecialistRegistry", "RegimeSpecialist", "SpecialistOutput"]
