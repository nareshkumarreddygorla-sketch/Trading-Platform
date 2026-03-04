"""
Full-market AI scanner: fetch bars for entire NSE universe, run FeatureEngine + AlphaModel,
rank all stocks by prediction confidence, return top opportunities.

Used by the AutonomousLoop to dynamically discover what to trade.
"""
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScoredStock:
    """A stock scored by the AI model."""
    symbol: str
    exchange: str
    confidence: float          # 0-1, model prediction strength
    side: str                  # "BUY" or "SELL"
    probability: float         # raw model output
    price: float               # latest close price
    volume: float              # latest volume
    features: Dict[str, float] = field(default_factory=dict)
    regime: Optional[str] = None


@dataclass
class ScanResult:
    """Result of a full market scan."""
    scanned: int               # total symbols scanned
    signals_found: int         # stocks with confidence above threshold
    top_stocks: List[ScoredStock]
    scan_time_ms: float
    timestamp: str


class MarketScanner:
    """
    Scan the full market universe:
    1. For each symbol, fetch latest bars (from cache or yfinance)
    2. Build features via FeatureEngine
    3. Predict via AlphaModel
    4. Rank by confidence, return top N

    Works in two modes:
    - Live: uses BarCache (fed by MarketDataService WebSocket)
    - Polling: fetches from yfinance on each scan (for paper mode without broker feed)
    """

    def __init__(
        self,
        alpha_model,
        feature_engine=None,
        *,
        top_n: int = 20,
        min_confidence: float = 0.55,
        min_volume: float = 10_000,
        scan_batch_size: int = 50,
        yfinance_interval: str = "5m",
        yfinance_period: str = "5d",
    ):
        self.alpha_model = alpha_model
        self.feature_engine = feature_engine
        if self.feature_engine is None:
            from src.ai.feature_engine import FeatureEngine
            self.feature_engine = FeatureEngine()

        self.top_n = top_n
        self.min_confidence = min_confidence
        self.min_volume = min_volume
        self.scan_batch_size = scan_batch_size
        self.yfinance_interval = yfinance_interval
        self.yfinance_period = yfinance_period
        self._market_ctx = None  # cached per scan cycle

    def _fetch_bars_yfinance(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch latest bars for multiple symbols from yfinance (batch)."""
        import yfinance as yf
        from src.core.events import Bar, Exchange

        result = {}
        # Batch download
        try:
            data = yf.download(
                symbols,
                period=self.yfinance_period,
                interval=self.yfinance_interval,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            if data.empty:
                return result

            for symbol in symbols:
                try:
                    if len(symbols) == 1:
                        df = data
                    else:
                        df = data[symbol] if symbol in data.columns.get_level_values(0) else None

                    if df is None or df.empty:
                        continue

                    # Rename columns to lowercase
                    df = df.rename(columns=str.lower)
                    if "close" not in df.columns:
                        continue

                    # Drop NaN rows
                    df = df.dropna(subset=["close"])
                    if len(df) < 30:
                        continue

                    bars = []
                    for idx, row in df.iterrows():
                        ts = idx.to_pydatetime()
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        bars.append(
                            Bar(
                                symbol=symbol.replace(".NS", ""),
                                exchange=Exchange.NSE,
                                interval=self.yfinance_interval,
                                open=float(row.get("open", 0)),
                                high=float(row.get("high", 0)),
                                low=float(row.get("low", 0)),
                                close=float(row["close"]),
                                volume=float(row.get("volume", 0)),
                                ts=ts,
                            )
                        )
                    if len(bars) >= 30:
                        result[symbol] = bars
                except Exception as e:
                    logger.debug("Failed to parse %s: %s", symbol, e)
        except Exception as e:
            logger.warning("yfinance batch download failed: %s", e)

        return result

    def _score_bars(self, symbol: str, bars: list) -> Optional[ScoredStock]:
        """Run feature engine + alpha model on bars, return scored stock."""
        if len(bars) < 30:
            return None

        try:
            features = self.feature_engine.build_features(bars)

            # Inject market context
            if self._market_ctx:
                features.update(self._market_ctx)

            probability = self.alpha_model.predict(features)
            confidence = abs(probability - 0.5) * 2.0

            # Regime-based confidence adjustment:
            # Reduce BUY confidence in bearish market, SELL in bullish
            if self._market_ctx:
                nifty_trend = self._market_ctx.get("nifty_trend", 0.0)
                nifty_rsi = self._market_ctx.get("nifty_rsi", 50.0)
                if probability >= 0.5 and nifty_trend < 0 and nifty_rsi < 40:
                    confidence *= 0.7  # reduce BUY confidence in bearish market
                elif probability < 0.5 and nifty_trend > 0 and nifty_rsi > 60:
                    confidence *= 0.7  # reduce SELL confidence in bullish market

            if confidence < self.min_confidence:
                return None

            latest = bars[-1]
            if latest.volume < self.min_volume:
                return None

            side = "BUY" if probability >= 0.5 else "SELL"

            return ScoredStock(
                symbol=symbol.replace(".NS", ""),
                exchange="NSE",
                confidence=confidence,
                side=side,
                probability=probability,
                price=latest.close,
                volume=latest.volume,
                features=features,
            )
        except Exception as e:
            logger.debug("Score failed for %s: %s", symbol, e)
            return None

    def scan(
        self,
        universe: Optional[List[str]] = None,
        bar_cache=None,
    ) -> ScanResult:
        """
        Scan the full market universe.

        Args:
            universe: list of symbols to scan (default: NIFTY 500 from nse_universe)
            bar_cache: optional BarCache for live data; falls back to yfinance

        Returns:
            ScanResult with top-ranked stocks
        """
        t0 = time.time()

        if universe is None:
            # Dynamic: auto-scan entire NSE market, pick best by liquidity
            try:
                from src.scanner.dynamic_universe import get_dynamic_universe
                universe = get_dynamic_universe().get_yfinance_symbols(count=300)
            except Exception:
                from src.scanner.nse_universe import get_universe
                universe = get_universe(yfinance_suffix=True)

        logger.info("MarketScanner: scanning %d symbols...", len(universe))

        # Refresh market context once per scan cycle
        try:
            from src.ai.market_context import fetch_market_context
            self._market_ctx = fetch_market_context()
            logger.info("Market context: NIFTY RSI=%.1f trend=%.0f vol=%.4f",
                        self._market_ctx.get('nifty_rsi', 0),
                        self._market_ctx.get('nifty_trend', 0),
                        self._market_ctx.get('nifty_volatility', 0))
        except Exception as e:
            logger.debug("Market context unavailable: %s", e)
            self._market_ctx = None

        scored: List[ScoredStock] = []

        # Try BarCache first for symbols that have live data
        cache_hits = set()
        if bar_cache is not None:
            from src.core.events import Exchange
            for symbol in universe:
                raw_sym = symbol.replace(".NS", "")
                bars = bar_cache.get_bars(raw_sym, Exchange.NSE, "1m", 100)
                if bars and len(bars) >= 30:
                    cache_hits.add(symbol)
                    result = self._score_bars(symbol, bars)
                    if result:
                        scored.append(result)

        # Fetch remaining from yfinance in batches
        remaining = [s for s in universe if s not in cache_hits]
        if remaining:
            for i in range(0, len(remaining), self.scan_batch_size):
                batch = remaining[i : i + self.scan_batch_size]
                try:
                    bars_map = self._fetch_bars_yfinance(batch)
                    for symbol, bars in bars_map.items():
                        result = self._score_bars(symbol, bars)
                        if result:
                            scored.append(result)
                except Exception as e:
                    logger.warning("Batch scan failed: %s", e)

        # Rank by confidence (descending)
        scored.sort(key=lambda s: s.confidence, reverse=True)
        top = scored[: self.top_n]

        elapsed_ms = (time.time() - t0) * 1000

        if top:
            logger.info(
                "MarketScanner: scanned %d symbols, %d signals found, top %d returned (%.0fms)",
                len(universe), len(scored), len(top), elapsed_ms,
            )
            for s in top[:5]:
                logger.info("  %s %s  confidence=%.3f  price=%.2f  vol=%.0f",
                            s.side, s.symbol, s.confidence, s.price, s.volume)
        else:
            logger.info("MarketScanner: scanned %d symbols, no signals above threshold (%.0fms)",
                        len(universe), elapsed_ms)

        return ScanResult(
            scanned=len(universe),
            signals_found=len(scored),
            top_stocks=top,
            scan_time_ms=elapsed_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def scan_to_signals(
        self,
        universe: Optional[List[str]] = None,
        bar_cache=None,
    ) -> list:
        """
        Scan and convert top stocks to platform Signal objects for the AutonomousLoop.
        """
        from src.core.events import Exchange, Signal, SignalSide

        result = self.scan(universe=universe, bar_cache=bar_cache)
        signals = []
        for stock in result.top_stocks:
            sig = Signal(
                strategy_id="ai_alpha",
                symbol=stock.symbol,
                exchange=Exchange.NSE,
                side=SignalSide.BUY if stock.side == "BUY" else SignalSide.SELL,
                score=stock.confidence,
                portfolio_weight=stock.confidence * 0.1,
                risk_level="NORMAL",
                reason=f"scanner_prob={stock.probability:.3f}",
                price=stock.price,
                ts=datetime.now(timezone.utc),
                metadata={"probability": stock.probability, "scanner": True},
            )
            signals.append(sig)
        return signals
