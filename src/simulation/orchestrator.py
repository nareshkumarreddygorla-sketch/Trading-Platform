"""
Simulation Orchestrator: coordinates nightly simulation pipeline.

Pipeline:
1. Wait for market close (15:30 IST)
2. Fetch latest OHLCV data
3. Get tradeable universe
4. Generate permutations
5. Run simulations
6. Store results
7. Update strategy registry for next day
8. Send notification
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.simulation.nightly_simulator import NightlySimulator, SimulationResult

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))


class SimulationOrchestrator:
    """
    End-to-end nightly simulation pipeline.
    Triggered at 16:30 IST after market close.
    """

    def __init__(
        self,
        simulator: Optional[NightlySimulator] = None,
        ohlcv_repo=None,
        strategy_registry=None,
        notifier=None,
        max_workers: int = 4,
        top_n: int = 5,
    ):
        self.simulator = simulator or NightlySimulator(
            max_workers=max_workers,
            top_n=top_n,
        )
        self._ohlcv_repo = ohlcv_repo
        self._strategy_registry = strategy_registry
        self._notifier = notifier
        self._running = False
        self._last_results: List[SimulationResult] = []

    async def run_nightly_pipeline(
        self,
        symbols: Optional[List[str]] = None,
        intervals: List[str] = None,
    ) -> List[SimulationResult]:
        """
        Execute the full nightly simulation pipeline.

        Args:
            symbols: Override symbol list (default: fetch from universe scanner)
            intervals: Timeframes to test (default: ["15m"])

        Returns:
            Ranked simulation results
        """
        if self._running:
            logger.warning("Simulation already running, skipping")
            return []

        self._running = True
        try:
            logger.info("=== NIGHTLY SIMULATION PIPELINE START ===")

            # Step 1: Get tradeable universe
            if symbols is None:
                symbols = await self._get_tradeable_symbols()
            if not symbols:
                logger.error("No symbols available for simulation")
                return []

            intervals = intervals or ["15m"]

            # Step 2: Fetch historical data
            bars_data = await self._fetch_bars_data(symbols, intervals)
            if not bars_data:
                logger.error("No bar data available for simulation")
                return []

            # Step 3: Generate permutations
            permutations = self.simulator.generate_permutations(
                symbols=symbols,
                intervals=intervals,
            )

            # Step 4: Run simulations
            results = await self.simulator.run_simulations(bars_data, permutations)

            # Step 5: Store results
            await self._store_results(results)

            # Step 6: Update strategy registry
            selected = self.simulator.get_selected_strategies()
            await self._update_registry(selected)

            # Step 7: Notify
            await self._send_notification(results, selected)

            self._last_results = results
            logger.info("=== NIGHTLY SIMULATION PIPELINE COMPLETE: %d results ===", len(results))
            return results

        except Exception as e:
            logger.exception("Nightly simulation pipeline failed: %s", e)
            return []
        finally:
            self._running = False

    async def _get_tradeable_symbols(self) -> List[str]:
        """Get symbols from dynamic universe scanner."""
        try:
            from src.scanner.dynamic_universe import DynamicUniverse
            scanner = DynamicUniverse()
            symbols = await scanner.get_tradeable_stocks(max_count=100)
            logger.info("Universe scanner returned %d symbols", len(symbols))
            return symbols
        except Exception as e:
            logger.warning("Universe scanner failed, using defaults: %s", e)
            return [
                "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
                "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
                "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "TITAN",
                "ASIANPAINT", "SUNPHARMA", "WIPRO", "HCLTECH", "ULTRACEMCO",
            ]

    async def _fetch_bars_data(
        self,
        symbols: List[str],
        intervals: List[str],
    ) -> Dict[str, list]:
        """
        Fetch bar data from OHLCV repo or yfinance fallback.
        Returns dict of symbol -> list of serializable bar dicts.
        """
        bars_data: Dict[str, list] = {}

        # Try OHLCV repo first
        if self._ohlcv_repo:
            try:
                for symbol in symbols:
                    bars = self._ohlcv_repo.get_bars(symbol, interval="1d", limit=500)
                    if bars:
                        bars_data[symbol] = [
                            {
                                "open": b.open, "high": b.high, "low": b.low,
                                "close": b.close, "volume": b.volume,
                                "timestamp": b.timestamp.isoformat() if hasattr(b.timestamp, 'isoformat') else str(b.timestamp),
                            }
                            for b in bars
                        ]
                if bars_data:
                    logger.info("Loaded bars from OHLCV repo for %d symbols", len(bars_data))
                    return bars_data
            except Exception as e:
                logger.warning("OHLCV repo fetch failed: %s", e)

        # Fallback: yfinance
        try:
            import yfinance as yf
            logger.info("Fetching bars from yfinance for %d symbols", len(symbols))

            for symbol in symbols:
                try:
                    ticker = yf.Ticker(f"{symbol}.NS")
                    df = ticker.history(period="6mo", interval="1d")
                    if df.empty:
                        continue
                    bars_data[symbol] = [
                        {
                            "open": float(row["Open"]),
                            "high": float(row["High"]),
                            "low": float(row["Low"]),
                            "close": float(row["Close"]),
                            "volume": int(row["Volume"]),
                            "timestamp": idx.isoformat(),
                        }
                        for idx, row in df.iterrows()
                    ]
                except Exception as e:
                    logger.debug("yfinance error for %s: %s", symbol, e)
                    continue

            logger.info("yfinance: loaded data for %d/%d symbols", len(bars_data), len(symbols))
        except ImportError:
            logger.error("yfinance not installed")

        return bars_data

    async def _store_results(self, results: List[SimulationResult]) -> None:
        """Store simulation results in database."""
        try:
            from src.persistence.database import session_scope
            from src.persistence.models_market_data import SimulationResultModel

            now = datetime.now(_IST)
            with session_scope() as session:
                for r in results[:100]:  # Store top 100
                    model = SimulationResultModel(
                        run_date=now,
                        strategy_id=r.strategy_id,
                        strategy_params=json.dumps(r.params),
                        symbols=json.dumps(r.symbols[:10]),
                        interval=r.interval,
                        lookback_days=r.lookback_days,
                        total_return_pct=r.total_return_pct,
                        sharpe_ratio=r.sharpe_ratio,
                        sortino_ratio=r.sortino_ratio,
                        max_drawdown_pct=r.max_drawdown_pct,
                        win_rate=r.win_rate,
                        profit_factor=r.profit_factor,
                        total_trades=r.total_trades,
                        rank=r.rank,
                        selected=1 if r.selected else 0,
                    )
                    session.add(model)
                session.commit()
            logger.info("Stored %d simulation results", min(len(results), 100))
        except Exception as e:
            logger.error("Failed to store simulation results: %s", e)

    async def _update_registry(self, selected: List[SimulationResult]) -> None:
        """Update strategy registry with selected strategies for next day."""
        if not self._strategy_registry or not selected:
            return
        try:
            # Disable all, then enable selected
            for strat_id in self._strategy_registry.list_strategies():
                self._strategy_registry.disable(strat_id)

            for r in selected:
                self._strategy_registry.enable(r.strategy_id)
                logger.info(
                    "Selected for tomorrow: %s (Sharpe=%.2f, WR=%.1f%%, DD=%.1f%%)",
                    r.strategy_id, r.sharpe_ratio, r.win_rate, r.max_drawdown_pct,
                )
        except Exception as e:
            logger.error("Failed to update strategy registry: %s", e)

    async def _send_notification(
        self,
        results: List[SimulationResult],
        selected: List[SimulationResult],
    ) -> None:
        """Send simulation summary notification."""
        if not selected:
            return
        try:
            summary = f"Nightly Simulation Complete\n"
            summary += f"Total permutations tested: {len(results)}\n"
            summary += f"Selected {len(selected)} strategies for tomorrow:\n\n"
            for r in selected:
                summary += (
                    f"  {r.strategy_id} [{r.interval}]\n"
                    f"    Sharpe: {r.sharpe_ratio:.2f} | Win: {r.win_rate:.0f}% | "
                    f"DD: {r.max_drawdown_pct:.1f}% | Trades: {r.total_trades}\n"
                    f"    Params: {json.dumps(r.params)}\n\n"
                )
            logger.info(summary)

            if self._notifier:
                await self._notifier.notify("simulation_complete", summary)
        except Exception as e:
            logger.error("Notification failed: %s", e)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_results(self) -> List[SimulationResult]:
        return self._last_results
