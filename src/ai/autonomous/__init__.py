"""
Autonomous trading controller: ties LLM advisory, meta-alpha, and risk.
Use: after LLM advisory returns exposure_multiplier -> set on risk;
     after meta_alpha returns recommendation -> pass meta_alpha_scale to allocator.
"""
from .controller import AutonomousTradingController

__all__ = ["AutonomousTradingController"]
