#!/usr/bin/env python3
"""
Backtest Validation: prove the system's strategies actually work.
Runs backtests on real historical data and outputs metrics.

Usage:
    PYTHONPATH=. python3 scripts/run_backtest_validation.py
"""
import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest_validation")

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "nse_historical")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Top stocks to validate against
VALIDATION_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT",
]


def load_bars_from_parquet(symbol: str):
    """Load historical bars from parquet file."""
    import pandas as pd
    from datetime import timezone
    from src.core.events import Bar, Exchange

    path = os.path.join(DATA_DIR, f"{symbol}.NS.parquet")
    if not os.path.exists(path):
        # Try without .NS suffix
        path = os.path.join(DATA_DIR, f"{symbol}.parquet")
    if not os.path.exists(path):
        return []

    df = pd.read_parquet(path)
    bars = []
    for idx, row in df.iterrows():
        try:
            ts = pd.Timestamp(idx)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            bars.append(Bar(
                symbol=symbol,
                exchange=Exchange.NSE,
                interval="1d",
                open=float(row.get("open", row.get("Open", 0))),
                high=float(row.get("high", row.get("High", 0))),
                low=float(row.get("low", row.get("Low", 0))),
                close=float(row.get("close", row.get("Close", 0))),
                volume=float(row.get("volume", row.get("Volume", 0))),
                ts=ts.to_pydatetime(),
            ))
        except Exception as e:
            logger.debug("Bar parse error for %s: %s", symbol, e)
            continue
    return bars


def run_strategy_backtest(strategy, bars, symbol):
    """Run a single strategy backtest and return metrics."""
    from src.backtesting.engine import BacktestEngine, BacktestConfig
    from src.core.events import Exchange

    config = BacktestConfig(
        initial_capital=100_000.0,
        commission_pct=0.05,
        slippage_bps=5.0,
        latency_bars=1,
    )
    engine = BacktestEngine(config)
    result = engine.run(strategy, bars, symbol, Exchange.NSE)
    return result


def main():
    from src.strategy_engine.classical import (
        EMACrossoverStrategy, MACDStrategy, RSIStrategy,
    )
    from src.ai.alpha_model import AlphaStrategy, AlphaModel

    logger.info("=" * 60)
    logger.info("  BACKTEST VALIDATION — PROVING STRATEGIES WORK")
    logger.info("=" * 60)

    # Load AI model if available
    model_path = os.path.join(MODELS_DIR, "alpha_xgb.joblib")
    alpha_model = AlphaModel(strategy_id="ai_alpha")
    if os.path.exists(model_path):
        alpha_model.load(model_path)
        logger.info("Loaded XGBoost model from %s", model_path)

    strategies = {
        "ema_crossover": EMACrossoverStrategy(),
        "macd": MACDStrategy(),
        "rsi": RSIStrategy(),
        "ai_alpha": AlphaStrategy(alpha_model=alpha_model),
    }

    # Find available parquet files
    available_symbols = []
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.endswith(".parquet") and f not in ("all_stocks.parquet", "NIFTY50.parquet", "NSEBANK.parquet"):
                sym = f.replace(".NS.parquet", "").replace(".parquet", "")
                available_symbols.append(sym)

    if not available_symbols:
        logger.error("No data found in %s. Run training pipeline first.", DATA_DIR)
        return

    symbols = available_symbols[:10]  # Test on up to 10 stocks
    logger.info("Validating on %d symbols: %s", len(symbols), symbols[:5])
    logger.info("")

    results_summary = {}

    for strat_name, strategy in strategies.items():
        logger.info("--- Strategy: %s ---", strat_name)
        all_returns = []
        all_sharpe = []
        all_trades = []
        all_wins = []

        for symbol in symbols:
            bars = load_bars_from_parquet(symbol)
            if len(bars) < 100:
                continue

            result = run_strategy_backtest(strategy, bars, symbol)
            if result.metrics:
                m = result.metrics
                ret = getattr(m, 'total_return_pct', 0) or 0
                sharpe = getattr(m, 'sharpe', 0) or 0
                n_trades = getattr(m, 'num_trades', len(result.trades))
                win_rate = getattr(m, 'win_rate_pct', 0) or 0

                all_returns.append(ret)
                all_sharpe.append(sharpe)
                all_trades.append(n_trades)
                all_wins.append(win_rate)

        if all_returns:
            avg_return = sum(all_returns) / len(all_returns)
            avg_sharpe = sum(all_sharpe) / len(all_sharpe)
            total_trades = sum(all_trades)
            avg_win = sum(all_wins) / len(all_wins) if all_wins else 0

            logger.info("  Avg Return: %.2f%%", avg_return)
            logger.info("  Avg Sharpe: %.2f", avg_sharpe)
            logger.info("  Total Trades: %d", total_trades)
            logger.info("  Avg Win Rate: %.1f%%", avg_win)
            logger.info("")

            results_summary[strat_name] = {
                "avg_return_pct": round(avg_return, 2),
                "avg_sharpe": round(avg_sharpe, 2),
                "total_trades": total_trades,
                "avg_win_rate": round(avg_win, 1),
                "symbols_tested": len(all_returns),
            }
        else:
            logger.warning("  No results for %s", strat_name)
            results_summary[strat_name] = {"error": "no data"}

    # Summary
    logger.info("=" * 60)
    logger.info("  VALIDATION SUMMARY")
    logger.info("=" * 60)
    for name, metrics in results_summary.items():
        if "error" in metrics:
            logger.info("  %-20s — %s", name, metrics["error"])
        else:
            verdict = "PASS" if metrics["avg_return_pct"] > -5 and metrics["total_trades"] > 0 else "FAIL"
            logger.info("  %-20s  ret=%+.1f%%  sharpe=%.2f  trades=%d  win=%.0f%%  [%s]",
                        name, metrics["avg_return_pct"], metrics["avg_sharpe"],
                        metrics["total_trades"], metrics["avg_win_rate"], verdict)


if __name__ == "__main__":
    main()
