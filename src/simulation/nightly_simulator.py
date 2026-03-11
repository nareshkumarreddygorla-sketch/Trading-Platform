"""
Nightly Simulation Engine ("Holly AI" equivalent):
Run thousands of strategy permutations after market close.
Select top performers for the next trading day.

Runs at 16:30 IST (after NSE close at 15:30 + settlement buffer).
"""

import asyncio
import itertools
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class StrategyPermutation:
    """A single strategy + parameter combination to simulate."""

    strategy_id: str
    params: dict[str, Any]
    symbols: list[str]
    interval: str = "15m"
    lookback_days: int = 30


@dataclass
class SimulationResult:
    """Outcome of a single permutation backtest."""

    strategy_id: str
    params: dict[str, Any]
    symbols: list[str]
    interval: str
    lookback_days: int
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    rank: int = 0
    selected: bool = False


# --------------------------------------------------------------------------
# Strategy parameter grids
# --------------------------------------------------------------------------
STRATEGY_GRIDS: dict[str, dict[str, list]] = {
    "ema_crossover": {
        "fast_period": [5, 9, 12, 21],
        "slow_period": [21, 26, 50, 100],
        "rsi_filter": [True, False],
    },
    "macd_strategy": {
        "fast": [8, 12, 16],
        "slow": [21, 26, 34],
        "signal": [7, 9, 12],
    },
    "rsi_mean_reversion": {
        "period": [7, 14, 21],
        "oversold": [20, 25, 30],
        "overbought": [70, 75, 80],
    },
    "momentum_breakout": {
        "lookback": [10, 20, 30, 50],
        "breakout_pct": [1.0, 1.5, 2.0, 3.0],
        "volume_confirm": [True, False],
    },
    "bollinger_squeeze": {
        "period": [15, 20, 30],
        "num_std": [1.5, 2.0, 2.5],
        "squeeze_threshold": [0.02, 0.03, 0.05],
    },
    "ai_alpha": {
        "confidence_threshold": [0.55, 0.60, 0.65, 0.70],
        "regime_filter": [True, False],
    },
}


def _generate_param_combinations(grid: dict[str, list]) -> list[dict[str, Any]]:
    """Generate all parameter combinations from a grid."""
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    return combinations


def _filter_valid_ema(params: dict) -> bool:
    """Ensure EMA fast < slow."""
    if "fast_period" in params and "slow_period" in params:
        return params["fast_period"] < params["slow_period"]
    if "fast" in params and "slow" in params:
        return params["fast"] < params["slow"]
    return True


# --------------------------------------------------------------------------
# Simulation logic (runs in process pool for CPU parallelism)
# --------------------------------------------------------------------------
def _run_single_simulation(
    bars_data: dict[str, list],  # symbol -> list of bar dicts
    strategy_id: str,
    params: dict[str, Any],
    symbols: list[str],
    interval: str,
    initial_capital: float = 100000.0,
) -> dict[str, Any]:
    """
    Run a single strategy simulation on historical data.
    This function runs in a subprocess - must be self-contained.
    Returns result dict.
    """
    import numpy as np

    capital = initial_capital
    equity_curve = [capital]
    trades = []
    position = None  # {symbol, side, qty, entry_price, entry_idx}

    for symbol in symbols:
        bar_list = bars_data.get(symbol, [])
        if len(bar_list) < 50:
            continue

        closes = np.array([b["close"] for b in bar_list])
        highs = np.array([b["high"] for b in bar_list])
        lows = np.array([b["low"] for b in bar_list])
        volumes = np.array([b["volume"] for b in bar_list])

        position = None
        for i in range(50, len(closes)):
            signal = _evaluate_signal(
                strategy_id, params, closes[: i + 1], highs[: i + 1], lows[: i + 1], volumes[: i + 1]
            )

            if signal > 0 and position is None:
                # Enter long
                entry_price = closes[i] * 1.0005  # slippage
                qty = int((capital * 0.05) / entry_price)  # 5% position size
                if qty > 0:
                    position = {"side": "BUY", "qty": qty, "entry_price": entry_price, "entry_idx": i}

            elif signal < 0 and position is not None:
                # Exit
                exit_price = closes[i] * 0.9995  # slippage
                pnl = (exit_price - position["entry_price"]) * position["qty"]
                capital += pnl
                equity_curve.append(capital)
                trades.append(
                    {
                        "symbol": symbol,
                        "pnl": pnl,
                        "bars_held": i - position["entry_idx"],
                    }
                )
                position = None

        # Force close at end
        if position is not None:
            exit_price = closes[-1]
            pnl = (exit_price - position["entry_price"]) * position["qty"]
            capital += pnl
            equity_curve.append(capital)
            trades.append({"symbol": symbol, "pnl": pnl, "bars_held": len(closes) - position["entry_idx"]})

    # Compute metrics
    equity = np.array(equity_curve)
    total_return = (equity[-1] / equity[0] - 1) * 100 if len(equity) > 1 else 0.0

    # Sharpe
    if len(equity) > 2:
        returns = np.diff(equity) / equity[:-1]
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-12) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Sortino
    if len(equity) > 2:
        returns = np.diff(equity) / equity[:-1]
        downside = returns[returns < 0]
        sortino = float(np.mean(returns) / (np.std(downside) + 1e-12) * np.sqrt(252)) if len(downside) > 0 else sharpe
    else:
        sortino = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-12) * 100
    max_dd = float(np.min(dd))

    # Win rate
    wins = sum(1 for t in trades if t["pnl"] > 0)
    win_rate = (wins / len(trades) * 100) if trades else 0.0

    # Profit factor
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    profit_factor = gross_profit / (gross_loss + 1e-12)

    return {
        "strategy_id": strategy_id,
        "params": params,
        "symbols": symbols,
        "interval": interval,
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "total_trades": len(trades),
    }


def _evaluate_signal(
    strategy_id: str,
    params: dict,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
) -> int:
    """
    Evaluate a strategy signal at the current bar.
    Returns: 1 (buy), -1 (sell/exit), 0 (hold).
    """
    import numpy as np

    n = len(closes)

    if strategy_id == "ema_crossover":
        fast = params.get("fast_period", 12)
        slow = params.get("slow_period", 26)
        if n < slow + 1:
            return 0
        ema_f = _simple_ema(closes, fast)
        ema_s = _simple_ema(closes, slow)
        ema_f_prev = _simple_ema(closes[:-1], fast)
        ema_s_prev = _simple_ema(closes[:-1], slow)

        if params.get("rsi_filter", False):
            rsi = _simple_rsi(closes, 14)
            if ema_f > ema_s and ema_f_prev <= ema_s_prev and rsi < 70:
                return 1
            elif ema_f < ema_s and ema_f_prev >= ema_s_prev:
                return -1
        else:
            if ema_f > ema_s and ema_f_prev <= ema_s_prev:
                return 1
            elif ema_f < ema_s and ema_f_prev >= ema_s_prev:
                return -1

    elif strategy_id == "rsi_mean_reversion":
        period = params.get("period", 14)
        oversold = params.get("oversold", 30)
        overbought = params.get("overbought", 70)
        rsi = _simple_rsi(closes, period)
        if rsi < oversold:
            return 1
        elif rsi > overbought:
            return -1

    elif strategy_id == "momentum_breakout":
        lookback = params.get("lookback", 20)
        breakout_pct = params.get("breakout_pct", 2.0)
        if n < lookback + 1:
            return 0
        high_n = np.max(highs[-lookback - 1 : -1])
        change_pct = (closes[-1] / high_n - 1) * 100
        if change_pct > breakout_pct:
            if params.get("volume_confirm", True):
                avg_vol = np.mean(volumes[-lookback:])
                if volumes[-1] > avg_vol * 1.5:
                    return 1
            else:
                return 1
        elif closes[-1] < np.mean(closes[-lookback:]) * 0.97:
            return -1

    elif strategy_id == "bollinger_squeeze":
        period = params.get("period", 20)
        num_std = params.get("num_std", 2.0)
        squeeze_thresh = params.get("squeeze_threshold", 0.03)
        if n < period:
            return 0
        window = closes[-period:]
        sma = np.mean(window)
        std = np.std(window)
        bandwidth = (2 * num_std * std) / (sma + 1e-12)
        if bandwidth < squeeze_thresh and closes[-1] > sma:
            return 1
        elif closes[-1] < sma - num_std * std:
            return -1

    elif strategy_id == "macd_strategy":
        fast_p = params.get("fast", 12)
        slow_p = params.get("slow", 26)
        sig_p = params.get("signal", 9)
        if n < slow_p + sig_p:
            return 0
        ema_fast = _simple_ema(closes, fast_p)
        ema_slow = _simple_ema(closes, slow_p)
        macd = ema_fast - ema_slow
        # Approximate signal line
        ema_fast_prev = _simple_ema(closes[:-1], fast_p)
        ema_slow_prev = _simple_ema(closes[:-1], slow_p)
        macd_prev = ema_fast_prev - ema_slow_prev
        if macd > 0 and macd_prev <= 0:
            return 1
        elif macd < 0 and macd_prev >= 0:
            return -1

    elif strategy_id == "ai_alpha":
        # Simplified: use momentum + RSI composite
        conf_thresh = params.get("confidence_threshold", 0.6)
        if n < 30:
            return 0
        rsi = _simple_rsi(closes, 14)
        momentum = closes[-1] / closes[-20] - 1
        # Pseudo-confidence from normalized indicators
        confidence = 0.5 + momentum * 2 + (50 - rsi) / 200
        confidence = max(0, min(1, confidence))
        if confidence > conf_thresh:
            return 1
        elif confidence < (1 - conf_thresh):
            return -1

    return 0


def _simple_ema(data: np.ndarray, period: int) -> float:
    """Fast EMA calculation."""
    if len(data) < period:
        return float(data[-1])
    mult = 2.0 / (period + 1)
    ema = float(np.mean(data[:period]))
    for i in range(period, len(data)):
        ema = (data[i] - ema) * mult + ema
    return ema


def _simple_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Fast RSI calculation."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-period - 1 :])
    gains = np.mean(np.where(deltas > 0, deltas, 0))
    losses = np.mean(np.where(deltas < 0, -deltas, 0))
    if losses < 1e-12:
        return 100.0 if gains > 0 else 50.0
    rs = gains / losses
    return float(100.0 - 100.0 / (1.0 + rs))


class NightlySimulator:
    """
    Orchestrates nightly strategy simulations.

    After market close:
    1. Generate all strategy permutations
    2. Run backtests in parallel (multiprocessing)
    3. Rank results by composite score
    4. Select top N strategies for next trading day
    5. Store results and update strategy registry
    """

    def __init__(
        self,
        max_workers: int = 4,
        top_n: int = 5,
        min_trades: int = 5,
        min_sharpe: float = 0.5,
    ):
        self.max_workers = max_workers
        self.top_n = top_n
        self.min_trades = min_trades
        self.min_sharpe = min_sharpe
        self._results: list[SimulationResult] = []
        self._last_run: datetime | None = None

    def generate_permutations(
        self,
        symbols: list[str],
        intervals: list[str] = None,
        strategies: list[str] = None,
    ) -> list[StrategyPermutation]:
        """Generate all strategy-parameter-symbol permutations."""
        intervals = intervals or ["15m"]
        strategies = strategies or list(STRATEGY_GRIDS.keys())
        permutations = []

        for strategy_id in strategies:
            grid = STRATEGY_GRIDS.get(strategy_id, {})
            if not grid:
                continue

            param_combos = _generate_param_combinations(grid)

            for params in param_combos:
                # Filter invalid combinations
                if not _filter_valid_ema(params):
                    continue

                for interval in intervals:
                    permutations.append(
                        StrategyPermutation(
                            strategy_id=strategy_id,
                            params=params,
                            symbols=symbols,
                            interval=interval,
                        )
                    )

        logger.info("Generated %d permutations across %d strategies", len(permutations), len(strategies))
        return permutations

    async def run_simulations(
        self,
        bars_data: dict[str, list],
        permutations: list[StrategyPermutation],
    ) -> list[SimulationResult]:
        """
        Run all permutations in parallel using process pool.

        Args:
            bars_data: Dict of symbol -> list of bar dicts (serializable)
            permutations: List of strategy permutations to test

        Returns:
            Ranked list of SimulationResult
        """
        logger.info("Starting nightly simulation: %d permutations", len(permutations))
        start = datetime.now()

        loop = asyncio.get_event_loop()
        results = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            futures = []
            for perm in permutations:
                future = loop.run_in_executor(
                    pool,
                    _run_single_simulation,
                    bars_data,
                    perm.strategy_id,
                    perm.params,
                    perm.symbols,
                    perm.interval,
                )
                futures.append(future)

            raw_results = await asyncio.gather(*futures, return_exceptions=True)

        for raw in raw_results:
            if isinstance(raw, Exception):
                logger.error("Simulation failed: %s", raw)
                continue
            if isinstance(raw, dict):
                results.append(SimulationResult(**raw))

        # Filter: minimum trades and quality
        qualified = [r for r in results if r.total_trades >= self.min_trades and r.sharpe_ratio >= self.min_sharpe]

        # Rank by composite score: Sharpe * 0.4 + Sortino * 0.3 + WinRate * 0.2 + (100 - |MaxDD|) * 0.1
        for r in qualified:
            r._score = (
                r.sharpe_ratio * 0.4
                + r.sortino_ratio * 0.3
                + r.win_rate * 0.002  # normalize to ~0-0.2
                + (100 - abs(r.max_drawdown_pct)) * 0.001  # normalize
            )
        qualified.sort(key=lambda r: getattr(r, "_score", 0), reverse=True)

        # Assign ranks and select top N
        for i, r in enumerate(qualified):
            r.rank = i + 1
            r.selected = i < self.top_n

        self._results = qualified
        self._last_run = datetime.now(_IST)

        elapsed = (datetime.now() - start).total_seconds()
        logger.info(
            "Nightly simulation complete: %d results, %d qualified, top Sharpe=%.2f, elapsed=%.1fs",
            len(results),
            len(qualified),
            qualified[0].sharpe_ratio if qualified else 0,
            elapsed,
        )
        return qualified

    def get_selected_strategies(self) -> list[SimulationResult]:
        """Return strategies selected for next trading day."""
        return [r for r in self._results if r.selected]

    def get_all_results(self) -> list[SimulationResult]:
        """Return all ranked results from last run."""
        return self._results

    @property
    def last_run_time(self) -> datetime | None:
        return self._last_run
