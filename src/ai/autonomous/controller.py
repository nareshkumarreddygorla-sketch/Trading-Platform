"""
Autonomous trading controller: applies LLM exposure multiplier to risk,
and meta_alpha recommendation (reduce_size / filter_signal) as scale for allocator.
Phase 9: meta_alpha influences allocation BEFORE order entry — block, size reduction, Kelly cut.
"""
import logging

logger = logging.getLogger(__name__)


class AutonomousTradingController:
    """
    Single place to apply:
    1. LLM advisory exposure_multiplier -> risk_manager.set_exposure_multiplier
    2. Meta-alpha recommendation -> meta_alpha_scale for MetaAllocator.allocate()
    3. Phase 9: block trade if P(primary_wrong) > threshold; size *= (1 - P_wrong);
       reduce Kelly when confidence inflation; tighten stop when P(regime_flip) high.
    """

    def __init__(self, risk_manager=None, risk_gate=None, block_primary_wrong_threshold: float = 0.7):
        self.risk_manager = risk_manager
        self.risk_gate = risk_gate
        self.block_primary_wrong_threshold = block_primary_wrong_threshold
        self._last_exposure_mult = 1.0
        self._last_meta_alpha_scale = 1.0

    def apply_llm_advisory(self, exposure_multiplier: float) -> None:
        """Set risk exposure from LLM advisory (0.5--1.5)."""
        mult = max(0.5, min(1.5, float(exposure_multiplier)))
        if self.risk_manager is not None and hasattr(self.risk_manager, "set_exposure_multiplier"):
            self.risk_manager.set_exposure_multiplier(mult)
            self._last_exposure_mult = mult
            logger.info("Autonomous: exposure_multiplier set to %.2f", mult)
        elif self.risk_gate is not None and hasattr(self.risk_gate, "set_exposure_multiplier"):
            self.risk_gate.set_exposure_multiplier(mult)
            self._last_exposure_mult = mult
            logger.info("Autonomous: exposure_multiplier set to %.2f via risk_gate", mult)

    def meta_alpha_scale_for_allocator(self, recommendation: str, prob_primary_wrong: float) -> float:
        """
        Return scale in (0, 1] for MetaAllocator.allocate(meta_alpha_scale=...).
        reduce_size -> scale by (1 - prob_primary_wrong); filter_signal -> 0.5; hold -> 1.0.
        """
        if recommendation == "reduce_size":
            self._last_meta_alpha_scale = max(0.2, 1.0 - prob_primary_wrong)
        elif recommendation == "filter_signal":
            self._last_meta_alpha_scale = 0.5
        else:
            self._last_meta_alpha_scale = 1.0
        return self._last_meta_alpha_scale

    def block_trade_from_meta_alpha(self, prob_primary_wrong: float) -> bool:
        """Phase 9: block trade if P(primary_wrong) > block_threshold. Call BEFORE order entry."""
        return prob_primary_wrong >= self.block_primary_wrong_threshold

    def size_multiplier_from_meta_alpha(
        self,
        prob_primary_wrong: float,
        prob_confidence_inflated: float,
        recommendation: str,
    ) -> float:
        """
        Phase 9: size *= (1 - P_wrong); if confidence inflated, further *= (1 - inflation).
        Call before order entry; apply to quantity.
        """
        mult = 1.0 - prob_primary_wrong
        if prob_confidence_inflated > 0.5:
            mult *= (1.0 - prob_confidence_inflated * 0.5)
        if recommendation == "filter_signal":
            mult *= 0.5
        return max(0.1, min(1.0, mult))

    def regime_flip_stop_multiplier(self, prob_regime_flip: float) -> float:
        """Phase 9: when P(regime_flip) high, tighten stop (e.g. stop_mult < 1). Return multiplier for stop distance."""
        if prob_regime_flip >= 0.6:
            return 0.7
        if prob_regime_flip >= 0.4:
            return 0.85
        return 1.0
