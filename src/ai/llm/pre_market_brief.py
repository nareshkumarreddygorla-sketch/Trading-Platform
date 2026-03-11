"""
Pre-Market Briefing Engine:
Generates AI-powered pre-market analysis at 9:00 IST.

Aggregates:
- Overnight global market moves
- News sentiment
- Sector rotation signals
- Risk assessment for the day
- Recommended sector biases and exposure multiplier
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from src.ai.llm.client import LLMClient

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))

PRE_MARKET_SYSTEM_PROMPT = """You are an expert quantitative trading analyst for the Indian NSE market.
Generate a concise pre-market briefing based on the provided data.

Your analysis must include:
1. MARKET OUTLOOK: Overall directional bias (bullish/bearish/neutral) with confidence (0-1)
2. KEY RISKS: Top 3 risks for today's session
3. SECTOR BIAS: Which sectors to overweight/underweight
4. EXPOSURE RECOMMENDATION: Suggested exposure multiplier (0.0 to 1.5)
   - 0.0 = stay flat (extreme risk)
   - 0.5 = half position sizes
   - 1.0 = normal
   - 1.5 = increase exposure (strong conviction)
5. KEY LEVELS: Important Nifty/BankNifty levels to watch

Respond in valid JSON format:
{
    "outlook": "bullish|bearish|neutral",
    "outlook_confidence": 0.0-1.0,
    "exposure_multiplier": 0.0-1.5,
    "key_risks": ["risk1", "risk2", "risk3"],
    "sector_bias": {"IT": "overweight", "Banking": "neutral", ...},
    "key_levels": {"nifty_support": 0, "nifty_resistance": 0},
    "summary": "2-3 sentence market summary",
    "actionable_insights": ["insight1", "insight2"]
}"""


class PreMarketBriefing:
    """
    Generates pre-market intelligence using LLM analysis.
    Called at 9:00 IST before market open (9:15).
    """

    def __init__(self, llm_client: LLMClient | None = None):
        self._llm = llm_client
        self._latest_brief: dict[str, Any] | None = None
        self._latest_time: datetime | None = None

    async def generate_briefing(
        self,
        news_headlines: str = "",
        global_markets: dict[str, float] | None = None,
        yesterday_performance: dict[str, Any] | None = None,
        current_regime: str = "unknown",
        vix_level: float = 0.0,
    ) -> dict[str, Any]:
        """
        Generate pre-market briefing.

        Args:
            news_headlines: Formatted news headlines string
            global_markets: Dict of market -> overnight change %
            yesterday_performance: Our portfolio performance yesterday
            current_regime: Current market regime classification
            vix_level: India VIX level

        Returns:
            Briefing dict with outlook, exposure, risks, sectors
        """
        # Build context
        context_parts = []

        context_parts.append(f"Date: {datetime.now(_IST).strftime('%Y-%m-%d %A')}")
        context_parts.append(f"Current Regime: {current_regime}")

        if vix_level > 0:
            context_parts.append(f"India VIX: {vix_level:.2f}")
            if vix_level > 25:
                context_parts.append("WARNING: VIX elevated - high volatility expected")

        if global_markets:
            context_parts.append("\nGLOBAL MARKETS (overnight):")
            for market, change in global_markets.items():
                direction = "+" if change >= 0 else ""
                context_parts.append(f"  {market}: {direction}{change:.2f}%")

        if yesterday_performance:
            context_parts.append("\nYESTERDAY'S PERFORMANCE:")
            for key, val in yesterday_performance.items():
                context_parts.append(f"  {key}: {val}")

        if news_headlines:
            context_parts.append(f"\nTOP NEWS:\n{news_headlines}")

        context = "\n".join(context_parts)

        # Get LLM analysis
        if self._llm:
            try:
                response = await self._llm.complete(
                    system=PRE_MARKET_SYSTEM_PROMPT,
                    user=context,
                    max_tokens=1024,
                )
                brief = self._parse_response(response)
            except Exception as e:
                logger.error("LLM briefing failed: %s", e)
                brief = self._generate_fallback(vix_level, current_regime)
        else:
            brief = self._generate_fallback(vix_level, current_regime)

        # Add metadata
        brief["generated_at"] = datetime.now(_IST).isoformat()
        brief["context_used"] = {
            "news_count": len(news_headlines.split("\n")) if news_headlines else 0,
            "global_markets": bool(global_markets),
            "vix": vix_level,
            "regime": current_regime,
        }

        self._latest_brief = brief
        self._latest_time = datetime.now(_IST)

        logger.info(
            "Pre-market brief: outlook=%s, confidence=%.2f, exposure=%.2f",
            brief.get("outlook", "unknown"),
            brief.get("outlook_confidence", 0),
            brief.get("exposure_multiplier", 1.0),
        )
        return brief

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse LLM JSON response."""
        try:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

        logger.warning("Failed to parse LLM response, using fallback")
        return self._generate_fallback(0, "unknown")

    def _generate_fallback(
        self,
        vix_level: float,
        regime: str,
    ) -> dict[str, Any]:
        """Generate rule-based fallback when LLM is unavailable."""
        # Conservative defaults
        exposure = 1.0
        outlook = "neutral"
        confidence = 0.3

        if vix_level > 30:
            exposure = 0.3
            outlook = "bearish"
            confidence = 0.6
        elif vix_level > 20:
            exposure = 0.7
            outlook = "neutral"
            confidence = 0.4

        if regime == "crisis":
            exposure = 0.0
            outlook = "bearish"
            confidence = 0.8
        elif regime == "trending_up":
            exposure = min(exposure + 0.2, 1.3)
            if outlook == "neutral":
                outlook = "bullish"
        elif regime == "trending_down":
            exposure = max(exposure - 0.3, 0.3)
            outlook = "bearish"

        return {
            "outlook": outlook,
            "outlook_confidence": confidence,
            "exposure_multiplier": round(exposure, 2),
            "key_risks": [
                "LLM unavailable - using rule-based analysis",
                f"VIX at {vix_level:.1f}" if vix_level > 0 else "VIX data unavailable",
                f"Regime: {regime}",
            ],
            "sector_bias": {},
            "key_levels": {},
            "summary": f"Rule-based analysis: {outlook} outlook, exposure={exposure:.1f}x",
            "actionable_insights": [
                "Monitor VIX for volatility changes",
                "Review overnight global cues at market open",
            ],
        }

    @property
    def latest_briefing(self) -> dict[str, Any] | None:
        return self._latest_brief

    @property
    def exposure_multiplier(self) -> float:
        """Get current recommended exposure multiplier."""
        if self._latest_brief:
            return self._latest_brief.get("exposure_multiplier", 1.0)
        return 1.0
