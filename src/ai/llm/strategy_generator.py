"""LLM-powered natural-language → executable trading strategy generator.

Converts plain English descriptions (e.g. "buy when RSI < 30 and MACD crosses up")
into validated Python strategy code that plugs into the StrategyRunner pipeline.
"""

import json
import logging
import re
import textwrap
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

STRATEGY_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert quant developer. The user will describe a trading strategy in
plain English.  You must produce a JSON object with these keys:

{
  "name": "<snake_case strategy name>",
  "description": "<one-line human description>",
  "indicators": [{"name": "<indicator>", "params": {<params>}}],
  "entry_rules": [{"condition": "<python expression>", "side": "long"|"short"}],
  "exit_rules":  [{"condition": "<python expression>", "side": "close_long"|"close_short"|"close_all"}],
  "risk": {"stop_loss_pct": <float>, "take_profit_pct": <float>, "max_position_pct": <float>},
  "timeframe": "1m"|"5m"|"15m"|"1h"|"1d",
  "universe_filter": "<optional filter expression>"
}

Available indicators (use exactly these names):
RSI, EMA, SMA, MACD, BBANDS, ATR, VWAP, ADX, STOCH, OBV, CCI, WILLR, MFI, SUPERTREND

In conditions, reference indicators as `ind.<name>` and price as `bar.close`,
`bar.high`, `bar.low`, `bar.open`, `bar.volume`.

Cross-over helper: `crossover(ind.X, ind.Y)`, `crossunder(ind.X, ind.Y)`.

Return ONLY valid JSON, no markdown fences, no explanation.
""")

STRATEGY_CODE_TEMPLATE = textwrap.dedent('''\
"""Auto-generated strategy: {description}"""
from src.strategy_engine.base import BaseStrategy, Signal


class {class_name}(BaseStrategy):
    """Generated from: {prompt}"""

    name = "{name}"
    timeframe = "{timeframe}"

    def __init__(self):
        super().__init__()
        self.indicators = {indicators}
        self.risk = {risk}

    def on_bar(self, bar, indicators, portfolio):
        signals = []
        ind = type("Ind", (), indicators)()

        # Entry rules
{entry_code}

        # Exit rules
{exit_code}

        return signals
''')


@dataclass
class GeneratedStrategy:
    name: str
    description: str
    code: str
    config: dict[str, Any]
    prompt: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    validated: bool = False
    errors: list[str] = field(default_factory=list)


class StrategyGenerator:
    """Generate executable trading strategies from natural language via LLM."""

    def __init__(self, llm_client=None):
        self._llm = llm_client
        self._generated: dict[str, GeneratedStrategy] = {}

    async def generate(self, prompt: str) -> GeneratedStrategy:
        """Convert a natural-language strategy description to executable code."""
        config = await self._get_strategy_config(prompt)
        if not config:
            return GeneratedStrategy(
                name="error",
                description="Failed to parse",
                code="",
                config={},
                prompt=prompt,
                errors=["LLM did not return valid config"],
            )

        code = self._build_code(config, prompt)
        errors = self._validate(code, config)

        strat = GeneratedStrategy(
            name=config.get("name", "unnamed"),
            description=config.get("description", prompt[:80]),
            code=code,
            config=config,
            prompt=prompt,
            validated=len(errors) == 0,
            errors=errors,
        )
        self._generated[strat.name] = strat
        return strat

    async def _get_strategy_config(self, prompt: str) -> dict | None:
        """Ask LLM to parse the prompt into structured config."""
        if self._llm is None:
            return self._fallback_parse(prompt)

        try:
            raw = await self._llm.complete(
                system=STRATEGY_SYSTEM_PROMPT,
                user=f"Strategy request: {prompt}",
                max_tokens=1024,
            )
            # Strip markdown fences if present
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
            return json.loads(raw)
        except Exception as e:
            logger.warning("LLM strategy parse failed: %s — using fallback", e)
            return self._fallback_parse(prompt)

    def _fallback_parse(self, prompt: str) -> dict:
        """Rule-based fallback when LLM is unavailable."""
        prompt_lower = prompt.lower()
        indicators = []
        entry_rules = []
        exit_rules = []

        # Detect common indicators
        if "rsi" in prompt_lower:
            indicators.append({"name": "RSI", "params": {"period": 14}})
            if "below" in prompt_lower or "oversold" in prompt_lower or "< 30" in prompt_lower:
                entry_rules.append({"condition": "ind.RSI < 30", "side": "long"})
                exit_rules.append({"condition": "ind.RSI > 70", "side": "close_long"})
            elif "above" in prompt_lower or "overbought" in prompt_lower or "> 70" in prompt_lower:
                entry_rules.append({"condition": "ind.RSI > 70", "side": "short"})
                exit_rules.append({"condition": "ind.RSI < 30", "side": "close_short"})

        if "macd" in prompt_lower:
            indicators.append({"name": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}})
            if "cross" in prompt_lower:
                entry_rules.append({"condition": "crossover(ind.MACD, ind.MACD_signal)", "side": "long"})
                exit_rules.append({"condition": "crossunder(ind.MACD, ind.MACD_signal)", "side": "close_long"})

        if "ema" in prompt_lower or "moving average" in prompt_lower:
            indicators.append({"name": "EMA", "params": {"period": 20}})
            indicators.append({"name": "SMA", "params": {"period": 50}})
            entry_rules.append({"condition": "crossover(ind.EMA_20, ind.SMA_50)", "side": "long"})
            exit_rules.append({"condition": "crossunder(ind.EMA_20, ind.SMA_50)", "side": "close_long"})

        if "bollinger" in prompt_lower or "bbands" in prompt_lower:
            indicators.append({"name": "BBANDS", "params": {"period": 20, "std": 2}})
            entry_rules.append({"condition": "bar.close < ind.BBANDS_lower", "side": "long"})
            exit_rules.append({"condition": "bar.close > ind.BBANDS_upper", "side": "close_long"})

        if "vwap" in prompt_lower:
            indicators.append({"name": "VWAP", "params": {}})
            entry_rules.append({"condition": "bar.close > ind.VWAP * 1.002", "side": "long"})
            exit_rules.append({"condition": "bar.close < ind.VWAP * 0.998", "side": "close_long"})

        if "supertrend" in prompt_lower:
            indicators.append({"name": "SUPERTREND", "params": {"period": 10, "multiplier": 3}})
            entry_rules.append({"condition": "bar.close > ind.SUPERTREND", "side": "long"})
            exit_rules.append({"condition": "bar.close < ind.SUPERTREND", "side": "close_long"})

        # Defaults if nothing detected
        if not indicators:
            indicators = [
                {"name": "RSI", "params": {"period": 14}},
                {"name": "EMA", "params": {"period": 20}},
            ]
            entry_rules = [{"condition": "ind.RSI < 35 and bar.close > ind.EMA_20", "side": "long"}]
            exit_rules = [{"condition": "ind.RSI > 65", "side": "close_long"}]

        # Parse timeframe
        timeframe = "5m"
        for tf in ["1d", "1h", "15m", "5m", "1m"]:
            if tf in prompt_lower:
                timeframe = tf
                break

        # Parse risk
        sl = 2.0
        tp = 4.0
        sl_match = re.search(r"stop.?loss[:\s]*(\d+(?:\.\d+)?)\s*%", prompt_lower)
        if sl_match:
            sl = float(sl_match.group(1))
        tp_match = re.search(r"take.?profit[:\s]*(\d+(?:\.\d+)?)\s*%", prompt_lower)
        if tp_match:
            tp = float(tp_match.group(1))

        name_parts = re.sub(r"[^a-z0-9\s]", "", prompt_lower).split()[:4]
        name = "_".join(name_parts) if name_parts else "custom_strategy"

        return {
            "name": name,
            "description": prompt[:100],
            "indicators": indicators,
            "entry_rules": entry_rules,
            "exit_rules": exit_rules,
            "risk": {"stop_loss_pct": sl, "take_profit_pct": tp, "max_position_pct": 5.0},
            "timeframe": timeframe,
        }

    def _build_code(self, config: dict, prompt: str) -> str:
        """Build executable Python code from parsed config."""
        name = config.get("name", "custom")
        class_name = "".join(w.capitalize() for w in name.split("_")) + "Strategy"

        # Build entry code
        entry_lines = []
        for rule in config.get("entry_rules", []):
            cond = rule["condition"]
            side = rule.get("side", "long")
            signal_dir = "Signal.BUY" if side == "long" else "Signal.SELL"
            entry_lines.append(f"        if {cond}:")
            entry_lines.append(f"            signals.append({signal_dir})")
        entry_code = "\n".join(entry_lines) if entry_lines else "        pass  # no entry rules"

        # Build exit code
        exit_lines = []
        for rule in config.get("exit_rules", []):
            cond = rule["condition"]
            side = rule.get("side", "close_long")
            if "short" in side:
                signal_dir = "Signal.BUY"  # close short = buy
            else:
                signal_dir = "Signal.SELL"  # close long = sell
            exit_lines.append(f"        if {cond}:")
            exit_lines.append(f"            signals.append({signal_dir})")
        exit_code = "\n".join(exit_lines) if exit_lines else "        pass  # no exit rules"

        return STRATEGY_CODE_TEMPLATE.format(
            description=config.get("description", ""),
            class_name=class_name,
            prompt=prompt[:120].replace('"', '\\"'),
            name=name,
            timeframe=config.get("timeframe", "5m"),
            indicators=repr(config.get("indicators", [])),
            risk=repr(config.get("risk", {})),
            entry_code=entry_code,
            exit_code=exit_code,
        )

    def _validate(self, code: str, config: dict) -> list[str]:
        """Validate generated strategy code."""
        errors = []

        # Syntax check
        try:
            compile(code, "<strategy>", "exec")
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")

        # Check required fields
        if not config.get("entry_rules"):
            errors.append("No entry rules defined")
        if not config.get("indicators"):
            errors.append("No indicators specified")

        # Check risk params
        risk = config.get("risk", {})
        if risk.get("stop_loss_pct", 0) <= 0:
            errors.append("stop_loss_pct must be positive")
        if risk.get("max_position_pct", 0) > 25:
            errors.append("max_position_pct exceeds 25% safety limit")

        # Disallow dangerous patterns
        dangerous = ["import os", "import subprocess", "exec(", "eval(", "__import__", "open("]
        for pat in dangerous:
            if pat in code:
                errors.append(f"Dangerous pattern detected: {pat}")

        return errors

    def list_generated(self) -> list[GeneratedStrategy]:
        return list(self._generated.values())

    def get_strategy(self, name: str) -> GeneratedStrategy | None:
        return self._generated.get(name)

    def delete_strategy(self, name: str) -> bool:
        return self._generated.pop(name, None) is not None
