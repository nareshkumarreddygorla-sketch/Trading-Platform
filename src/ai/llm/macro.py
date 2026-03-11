"""
Macro Risk Assessment via LLM: systemic risk, recommend risk reduction.
Output is advisory; risk engine still enforces hard limits.
"""

import logging
from dataclasses import dataclass

from .client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_MACRO = """You are a macro risk analyst. Given current macro indicators and headlines, respond in JSON only:
{
  "systemic_risk_level": "low" | "medium" | "high" | "critical",
  "recommendation": "none" | "reduce_leverage" | "reduce_position_size" | "increase_cash" | "halt_new_trades",
  "max_position_pct_suggestion": number or null,
  "reason": "brief reason"
}
No other text."""


@dataclass
class MacroRiskResult:
    systemic_risk_level: str
    recommendation: str
    max_position_pct_suggestion: float | None
    reason: str


class MacroRiskService:
    """LLM-based macro risk assessment. Output feeds into config/risk params only."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def assess(self, indicators: str, headlines: list[str]) -> MacroRiskResult | None:
        user = f"Indicators:\n{indicators}\n\nHeadlines:\n" + "\n".join(headlines[:10])
        raw = await self.llm.complete(SYSTEM_MACRO, user, max_tokens=256)
        if not raw:
            return None
        try:
            import json

            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            d = json.loads(raw)
            mpct = d.get("max_position_pct_suggestion")
            return MacroRiskResult(
                systemic_risk_level=d.get("systemic_risk_level", "low"),
                recommendation=d.get("recommendation", "none"),
                max_position_pct_suggestion=float(mpct) if mpct is not None else None,
                reason=d.get("reason", ""),
            )
        except Exception as e:
            logger.warning("Macro parse failed: %s", e)
            return None
