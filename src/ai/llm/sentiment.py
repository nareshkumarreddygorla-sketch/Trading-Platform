"""
News Sentiment Analysis via LLM: earnings, RBI, budget, geopolitical.
Output: sentiment score and optional risk parameter suggestion (e.g. reduce exposure).
LLM does NOT place trades.
"""

import logging
from dataclasses import dataclass

from .client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_SENTIMENT = """You are a financial news analyst. Analyze the given news text and respond in JSON only:
{
  "sentiment": "positive" | "negative" | "neutral",
  "score": 0.0 to 1.0 (1=most positive),
  "risk_reduction_suggestion": "none" | "reduce_exposure" | "halt_new_positions" | "increase_cash",
  "reason": "brief reason"
}
No other text."""


@dataclass
class SentimentResult:
    sentiment: str
    score: float
    risk_reduction_suggestion: str
    reason: str


class NewsSentimentService:
    """Call LLM for news sentiment; return structured result and optional risk suggestion."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def analyze(self, news_text: str, source: str = "") -> SentimentResult | None:
        user = f"Source: {source}\n\nText:\n{news_text[:4000]}"
        raw = await self.llm.complete(SYSTEM_SENTIMENT, user, max_tokens=256)
        if not raw:
            return None
        try:
            import json

            # Strip markdown code block if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            d = json.loads(raw)
            return SentimentResult(
                sentiment=d.get("sentiment", "neutral"),
                score=float(d.get("score", 0.5)),
                risk_reduction_suggestion=d.get("risk_reduction_suggestion", "none"),
                reason=d.get("reason", ""),
            )
        except Exception as e:
            logger.warning("Sentiment parse failed: %s", e)
            return None
