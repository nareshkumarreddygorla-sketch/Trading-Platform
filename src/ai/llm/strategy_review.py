"""
Strategy Performance Review via LLM: summarize why strategies underperform,
suggest parameter adjustments. LLM does NOT place trades; suggestions are applied
only after risk check and optional human approval.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_REVIEW = """You are a quantitative strategy analyst. Given strategy performance stats, respond in JSON only:
{
  "summary": "1-2 sentence summary of why strategy may be underperforming",
  "suggestions": ["suggestion1", "suggestion2"],
  "parameter_adjustments": {"param_name": "value or null"} or {},
  "recommendation": "continue" | "reduce_weight" | "disable"
}
No other text."""


@dataclass
class StrategyReviewResult:
    summary: str
    suggestions: List[str]
    parameter_adjustments: Dict[str, Any]
    recommendation: str


class StrategyReviewService:
    """LLM summarizes strategy performance and suggests parameter/weight changes."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def review(self, strategy_id: str, stats: Dict[str, Any], recent_trades: List[Dict]) -> Optional[StrategyReviewResult]:
        user = f"Strategy: {strategy_id}\nStats: {stats}\nRecent trades (sample): {recent_trades[:20]}"
        raw = await self.llm.complete(SYSTEM_REVIEW, user, max_tokens=512)
        if not raw:
            return None
        try:
            import json
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            d = json.loads(raw)
            return StrategyReviewResult(
                summary=d.get("summary", ""),
                suggestions=d.get("suggestions", []),
                parameter_adjustments=d.get("parameter_adjustments", {}),
                recommendation=d.get("recommendation", "continue"),
            )
        except Exception as e:
            logger.warning("Strategy review parse failed: %s", e)
            return None
