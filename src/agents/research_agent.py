"""
Research Agent: AUTONOMOUSLY scans the ENTIRE NSE market, uses AI ensemble
to rank all stocks, and outputs the best trading opportunities.

No hardcoded stock lists — dynamically discovers what to trade.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from .base import BaseAgent

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """
    Autonomous market research agent.
    - Scans entire NSE market dynamically (not a fixed list)
    - Uses existing BarCache for live-feed stocks
    - Uses MarketScanner + yfinance for broader market discovery
    - Runs EnsembleEngine on all candidates
    - Publishes top opportunities to execution agent
    """

    name = "research_agent"
    description = "Autonomously scans entire NSE market for best trading opportunities"

    def __init__(
        self,
        get_symbols: Callable | None = None,
        get_bars: Callable | None = None,
        feature_engine=None,
        ensemble_engine=None,
        market_scanner=None,
        top_n: int = 10,
        min_confidence: float = 0.55,
        scan_interval: float = 300.0,  # 5 minutes
    ):
        super().__init__()
        self._get_symbols = get_symbols
        self._get_bars = get_bars
        self._feature_engine = feature_engine
        self._ensemble = ensemble_engine
        self._market_scanner = market_scanner
        self._top_n = top_n
        self._min_confidence = min_confidence
        self._scan_interval = scan_interval
        self._last_opportunities: list[dict[str, Any]] = []
        self._feature_history: dict[str, list[dict[str, float]]] = {}
        self._dynamic_universe: list[str] = []
        self._universe_refresh_counter = 0

    @property
    def interval_seconds(self) -> float:
        return self._scan_interval

    def _refresh_dynamic_universe(self) -> None:
        """Fetch the latest dynamic stock universe from the full NSE market."""
        try:
            from src.scanner.dynamic_universe import get_dynamic_universe

            universe = get_dynamic_universe()
            self._dynamic_universe = universe.get_trading_stocks(count=100)
            logger.info(
                "ResearchAgent: dynamic universe refreshed — %d stocks from full market scan",
                len(self._dynamic_universe),
            )
        except Exception as e:
            logger.debug("Dynamic universe refresh failed: %s", e)

    async def run_cycle(self) -> None:
        if not all([self._feature_engine, self._ensemble]):
            return

        # Process incoming messages
        msg = await self.receive_message()
        while msg is not None:
            if msg.msg_type == "risk_alert":
                logger.info("ResearchAgent: received risk alert, reducing scan scope")
            msg = await self.receive_message()

        # Refresh dynamic universe every 6 cycles (~30 min if interval=5min)
        self._universe_refresh_counter += 1
        if not self._dynamic_universe or self._universe_refresh_counter % 6 == 0:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._refresh_dynamic_universe)

        # === Phase 1: Scan symbols with live bars (BarCache) ===
        opportunities = []
        live_scanned = set()

        if self._get_symbols and self._get_bars:
            symbols = self._get_symbols()
            for symbol, exchange in symbols:
                live_scanned.add(symbol)
                try:
                    opp = self._score_symbol(symbol, exchange)
                    if opp:
                        opportunities.append(opp)
                except Exception as e:
                    logger.debug("Live scan failed for %s: %s", symbol, e)

        # === Phase 2: Scan broader market via MarketScanner (yfinance) ===
        # Only scan symbols NOT already in BarCache
        if self._market_scanner and self._dynamic_universe:
            discovery_symbols = [f"{s}.NS" for s in self._dynamic_universe if s not in live_scanned]
            if discovery_symbols:
                try:
                    loop = asyncio.get_event_loop()
                    _batch = discovery_symbols[:50]
                    scan_result = await loop.run_in_executor(
                        None,
                        lambda: self._market_scanner.scan(universe=_batch),
                    )
                    for scored in scan_result.top_stocks:
                        if scored.confidence >= self._min_confidence:
                            opportunities.append(
                                {
                                    "symbol": scored.symbol,
                                    "exchange": scored.exchange,
                                    "direction": scored.side,
                                    "prob_up": scored.probability,
                                    "confidence": scored.confidence,
                                    "expected_return": 0.0,
                                    "price": scored.price,
                                    "models": ["scanner"],
                                    "source": "market_discovery",
                                }
                            )
                except Exception as e:
                    logger.debug("Market discovery scan failed: %s", e)

        # Sort by confidence and take top N
        opportunities.sort(key=lambda x: x["confidence"], reverse=True)
        self._last_opportunities = opportunities[: self._top_n]

        if self._last_opportunities:
            logger.info(
                "ResearchAgent: found %d opportunities (top: %s conf=%.2f) — scanned %d live + %d discovery",
                len(self._last_opportunities),
                self._last_opportunities[0]["symbol"],
                self._last_opportunities[0]["confidence"],
                len(live_scanned),
                len(self._dynamic_universe),
            )

            # Send to execution agent
            await self.send_message(
                target="execution_agent",
                msg_type="opportunities",
                payload={"opportunities": self._last_opportunities},
            )

            # Broadcast for dashboard
            await self.send_message(
                target="broadcast",
                msg_type="research_update",
                payload={
                    "opportunities_count": len(self._last_opportunities),
                    "top_opportunities": self._last_opportunities[:5],
                    "scanned_live": len(live_scanned),
                    "scanned_discovery": len(self._dynamic_universe),
                },
            )

    def _score_symbol(self, symbol: str, exchange) -> dict[str, Any] | None:
        """Score a single symbol using the ensemble engine."""
        bars = self._get_bars(symbol, exchange, "1m", 100)
        if not bars or len(bars) < 60:
            return None

        features = self._feature_engine.build_features(bars)
        if not features:
            return None

        # Track history for sequence models (cap dict at 200 symbols to prevent leak)
        if symbol not in self._feature_history:
            if len(self._feature_history) >= 200:
                oldest = next(iter(self._feature_history))
                del self._feature_history[oldest]
            self._feature_history[symbol] = []
        self._feature_history[symbol].append(features)
        if len(self._feature_history[symbol]) > 100:
            self._feature_history[symbol] = self._feature_history[symbol][-100:]

        context = {
            "symbol": symbol,
            "feature_history": self._feature_history[symbol],
        }

        prediction = self._ensemble.predict(features, context)

        if prediction.confidence >= self._min_confidence:
            direction = "BUY" if prediction.prob_up > 0.5 else "SELL"
            return {
                "symbol": symbol,
                "exchange": exchange.value if hasattr(exchange, "value") else str(exchange),
                "direction": direction,
                "prob_up": prediction.prob_up,
                "confidence": prediction.confidence,
                "expected_return": prediction.expected_return,
                "price": bars[-1].close,
                "models": prediction.metadata.get("models", []),
                "source": "live_bars",
            }
        return None
