"""
LLM client: OpenAI and Claude. Used for sentiment, macro risk, strategy review only.
LLM does NOT place trades; only suggests risk parameters or strategy weights.
"""
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    provider: str  # openai | anthropic
    api_key: str
    model: str = "gpt-4o-mini"  # or claude-3-haiku
    max_tokens: int = 1024


class LLMClient:
    """Unified client for OpenAI and Anthropic. Async."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Any = None

    async def _get_client(self):
        if self._client is not None:
            return self._client
        if self.config.provider == "openai":
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.config.api_key)
            except ImportError:
                logger.warning("openai not installed")
        elif self.config.provider == "anthropic":
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(api_key=self.config.api_key)
            except ImportError:
                logger.warning("anthropic not installed")
        return self._client

    async def complete(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Single completion. Returns response text."""
        client = await self._get_client()
        if client is None:
            return ""
        max_tokens = max_tokens or self.config.max_tokens
        try:
            if self.config.provider == "openai":
                r = await client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    max_tokens=max_tokens,
                )
                return r.choices[0].message.content or ""
            elif self.config.provider == "anthropic":
                r = await client.messages.create(
                    model=self.config.model,
                    system=system,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": user}],
                )
                return r.content[0].text if r.content else ""
        except Exception as e:
            logger.exception("LLM complete failed: %s", e)
            return ""
        return ""
