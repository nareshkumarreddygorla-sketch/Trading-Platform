"""
LLM advisory: multi-source aggregation, event severity, risk multiplier (0.5x–1.5x).
Guardrails: no trade instructions; cap multiplier; require source for material claims.
"""

import logging
from dataclasses import dataclass

from .client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_ADVISORY = """You are a risk analyst. Given aggregated news/events, respond in JSON only:
{
  "event_severity": "low" | "medium" | "high",
  "exposure_multiplier": 0.5 to 1.5 (1.0 = normal; 0.5 = reduce; 1.5 = allow slightly more only if justified),
  "reason": "brief reason",
  "sources_cited": ["source1", "source2"] or []
}
Rules: Never instruct trades. Only suggest exposure multiplier. Cap multiplier in [0.5, 1.5]. Cite source for material claims. No other text."""


@dataclass
class AdvisoryResult:
    event_severity: str  # low | medium | high
    exposure_multiplier: float  # 0.5 to 1.5
    reason: str
    sources_cited: list[str]


def _clamp_multiplier(x: float, low: float = 0.5, high: float = 1.5) -> float:
    return max(low, min(high, float(x)))


class AdvisoryService:
    """
    Multi-source aggregation → event severity + risk multiplier.
    Guardrails: cap multiplier; no trade instructions; optional source requirement.
    """

    def __init__(self, llm: LLMClient, require_sources_for_extreme: bool = True):
        self.llm = llm
        self.require_sources_for_extreme = require_sources_for_extreme

    async def aggregate_and_advise(
        self,
        items: list[dict],
        max_tokens: int = 256,
    ) -> AdvisoryResult | None:
        """
        items: list of { "source": str, "text": str, "timestamp": str }.
        Returns event_severity, exposure_multiplier (0.5–1.5), reason, sources_cited.
        """
        if not items:
            return AdvisoryResult("low", 1.0, "No inputs", [])
        combined = "\n\n".join(f"[{x.get('source', 'unknown')}] {x.get('text', '')[:500]}" for x in items[:20])
        user = f"Aggregated inputs:\n{combined[:6000]}"
        raw = await self.llm.complete(SYSTEM_ADVISORY, user, max_tokens=max_tokens)
        if not raw:
            return None
        try:
            import json

            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            d = json.loads(raw)
            mult = _clamp_multiplier(float(d.get("exposure_multiplier", 1.0)))
            severity = str(d.get("event_severity", "low")).lower()
            if severity not in ("low", "medium", "high"):
                severity = "low"
            sources = d.get("sources_cited") or []
            if self.require_sources_for_extreme and (mult <= 0.6 or mult >= 1.4) and not sources:
                mult = _clamp_multiplier(mult, 0.6, 1.4)
            return AdvisoryResult(
                event_severity=severity,
                exposure_multiplier=mult,
                reason=str(d.get("reason", "")),
                sources_cited=sources if isinstance(sources, list) else [],
            )
        except Exception as e:
            logger.warning("Advisory parse failed: %s", e)
            return None
