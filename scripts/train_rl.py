#!/usr/bin/env python3
"""
Train RL (PPO) agent with walk-forward validation on NIFTY50 stocks.

Data is fetched from Yahoo Finance.  Walk-forward scheme:
  - Train on 6 months of data
  - Test (out-of-sample) on the next 2 months
  - Report out-of-sample Sharpe ratio

Saves the final model to models/rl_agent.zip.

Usage:
    PYTHONPATH=. python scripts/train_rl.py
    PYTHONPATH=. python scripts/train_rl.py --timesteps 1000000
    PYTHONPATH=. python scripts/train_rl.py --quick              # fast sanity run
    PYTHONPATH=. python scripts/train_rl.py --symbols RELIANCE.NS,TCS.NS
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_rl")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# NIFTY50 representative tickers (Yahoo Finance format)
NIFTY50_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "KOTAKBANK.NS",
    "ITC.NS",
    "LT.NS",
    "AXISBANK.NS",
    "BAJFINANCE.NS",
    "ASIANPAINT.NS",
    "MARUTI.NS",
    "HCLTECH.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "WIPRO.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "TATAMOTORS.NS",
    "POWERGRID.NS",
    "M&M.NS",
    "TATASTEEL.NS",
    "ONGC.NS",
    "JSWSTEEL.NS",
    "ADANIPORTS.NS",
    "COALINDIA.NS",
    "BAJAJFINSV.NS",
    "TECHM.NS",
    "LTIM.NS",
    "INDUSINDBK.NS",
    "HINDALCO.NS",
    "GRASIM.NS",
    "CIPLA.NS",
    "DRREDDY.NS",
    "DIVISLAB.NS",
    "EICHERMOT.NS",
    "BPCL.NS",
    "BRITANNIA.NS",
    "HEROMOTOCO.NS",
    "APOLLOHOSP.NS",
    "TATACONSUM.NS",
    "SBILIFE.NS",
    "HDFCLIFE.NS",
    "BAJAJ-AUTO.NS",
    "UPL.NS",
    "SHRIRAMFIN.NS",
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def fetch_yahoo_data(
    tickers: List[str],
    start: str,
    end: str,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch daily OHLCV from Yahoo Finance for each ticker.
    Returns dict of ticker -> DataFrame with columns [open, high, low, close, volume].
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed.  Run:  pip install yfinance")
        sys.exit(1)

    data: Dict[str, pd.DataFrame] = {}
    logger.info("Fetching %d tickers from Yahoo Finance (%s to %s) ...", len(tickers), start, end)

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df is None or len(df) < 60:
                logger.debug("Skipping %s -- too few rows (%d)", ticker, len(df) if df is not None else 0)
                continue
            # Normalise column names
            df.columns = [c.lower() for c in df.columns]
            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
            required = {"open", "high", "low", "close", "volume"}
            if not required.issubset(set(df.columns)):
                logger.debug("Skipping %s -- missing columns: %s", ticker, required - set(df.columns))
                continue
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            data[ticker] = df
            logger.info("  %s: %d bars", ticker, len(df))
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", ticker, exc)

    logger.info("Fetched %d / %d tickers", len(data), len(tickers))
    return data


def build_features_from_df(df: pd.DataFrame, symbol: str = "UNKNOWN") -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a DataFrame with OHLCV columns, compute FeatureEngine features.
    Returns (bars_ohlcv, features) both as np.ndarray, row-aligned.
    """
    from src.ai.feature_engine import FeatureEngine
    from src.ai.models.lstm_predictor import FEATURE_KEYS
    from src.core.events import Bar

    fe = FeatureEngine()
    lookback = 100  # FeatureEngine needs ~100 bars of history

    # Build Bar objects
    bars: List[Bar] = []
    for idx, row in df.iterrows():
        bars.append(
            Bar(
                symbol=symbol,
                exchange="NSE",
                interval="1d",
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                ts=idx if isinstance(idx, datetime) else datetime.now(),
            )
        )

    # Compute features starting from index 20 (need history for indicators)
    warmup = 20
    feature_rows = []
    for i in range(warmup, len(bars)):
        window = bars[max(0, i - lookback) : i + 1]
        feat_dict = fe.build_features(window)
        row_feats = [feat_dict.get(k, 0.0) for k in FEATURE_KEYS]
        feature_rows.append(row_feats)

    if len(feature_rows) < 60:
        return np.empty((0, 5)), np.empty((0, len(FEATURE_KEYS)))

    bars_ohlcv = df[["open", "high", "low", "close", "volume"]].values[warmup:].astype(np.float64)
    features = np.array(feature_rows, dtype=np.float32)

    # Align lengths
    min_len = min(len(bars_ohlcv), len(features))
    return bars_ohlcv[:min_len], features[:min_len]


def normalise_features(
    features_list: List[np.ndarray],
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Z-score normalise features across all segments.
    Returns (normalised list, means, stds).
    """
    all_feats = np.concatenate(features_list, axis=0)
    means = all_feats.mean(axis=0)
    stds = all_feats.std(axis=0)
    stds[stds < 1e-8] = 1.0

    normed = [(f - means) / stds for f in features_list]
    return normed, means, stds


# ---------------------------------------------------------------------------
# Walk-forward split
# ---------------------------------------------------------------------------
def walk_forward_split(
    df: pd.DataFrame,
    train_months: int = 6,
    test_months: int = 2,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate walk-forward (train, test) splits from a time-indexed DataFrame.
    Each fold: train on *train_months*, test on the next *test_months*.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    start = df.index.min()
    end = df.index.max()
    folds = []
    cursor = start

    while True:
        train_end = cursor + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break

        train_df = df.loc[cursor:train_end]
        test_df = df.loc[train_end:test_end]

        if len(train_df) >= 60 and len(test_df) >= 20:
            folds.append((train_df, test_df))

        # Slide forward by test_months
        cursor = train_end
    return folds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_agent(model, bars_ohlcv: np.ndarray, features: np.ndarray, **env_kwargs) -> Dict[str, float]:
    """
    Run the trained agent on a test segment and compute performance metrics.
    Returns dict with sharpe_ratio, total_return_pct, max_drawdown_pct, n_trades.
    """
    from src.ai.models.rl_environment import TradingGymEnv

    env = TradingGymEnv(bars_ohlcv, features, **env_kwargs)
    obs, info = env.reset()

    portfolio_values = [info["portfolio_value"]]
    n_trades = 0
    prev_action = 0

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        if action != prev_action and action != 0:
            n_trades += 1
        prev_action = action

        obs, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        done = terminated or truncated

    portfolio_values = np.array(portfolio_values)

    # Total return
    total_return_pct = (portfolio_values[-1] / portfolio_values[0] - 1.0) * 100

    # Daily returns for Sharpe
    daily_returns = np.diff(portfolio_values) / (portfolio_values[:-1] + 1e-12)
    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns)
    # Annualised Sharpe (assume ~252 trading days)
    sharpe = (mean_ret / (std_ret + 1e-12)) * np.sqrt(252) if std_ret > 1e-12 else 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (running_max - portfolio_values) / (running_max + 1e-12)
    max_drawdown_pct = float(np.max(drawdowns) * 100)

    return {
        "sharpe_ratio": float(sharpe),
        "total_return_pct": float(total_return_pct),
        "max_drawdown_pct": max_drawdown_pct,
        "n_trades": n_trades,
        "n_steps": len(portfolio_values) - 1,
    }


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train(
    tickers: Optional[List[str]] = None,
    total_timesteps: int = 500_000,
    train_months: int = 6,
    test_months: int = 2,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    quick: bool = False,
):
    """
    Full training pipeline:
      1. Fetch data from Yahoo Finance
      2. Build features
      3. Walk-forward validation folds
      4. Train PPO on training folds, evaluate on test folds
      5. Train final model on all data and save
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        logger.error(
            "stable-baselines3 is required.  Install with:\n  pip install 'stable-baselines3[extra]' gymnasium"
        )
        sys.exit(1)

    from src.ai.models.rl_environment import TradingGymEnv, make_vec_env

    if tickers is None:
        tickers = NIFTY50_TICKERS[:10] if quick else NIFTY50_TICKERS[:30]

    # Quick mode overrides
    if quick:
        total_timesteps = min(total_timesteps, 50_000)
        n_envs = 2
        logger.info("=== QUICK MODE: %d timesteps, %d envs, %d tickers ===", total_timesteps, n_envs, len(tickers))

    # ----- 1. Fetch data -----
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")  # 2 years
    stock_data = fetch_yahoo_data(tickers, start=start_date, end=end_date)
    if not stock_data:
        logger.error("No data fetched. Check your internet connection and tickers.")
        return

    # ----- 2. Build features per stock -----
    logger.info("Building features for %d stocks ...", len(stock_data))
    stock_bars: Dict[str, np.ndarray] = {}
    stock_features: Dict[str, np.ndarray] = {}
    stock_dfs: Dict[str, pd.DataFrame] = {}

    for ticker, df in stock_data.items():
        bars_arr, feats_arr = build_features_from_df(df, symbol=ticker)
        if len(bars_arr) < 60:
            continue
        stock_bars[ticker] = bars_arr
        stock_features[ticker] = feats_arr
        # Keep aligned DF for walk-forward splitting
        stock_dfs[ticker] = df.iloc[20 : 20 + len(bars_arr)]

    logger.info("Features built for %d stocks", len(stock_bars))
    if not stock_bars:
        logger.error("No usable stocks after feature computation.")
        return

    # ----- 3. Walk-forward validation -----
    logger.info("=" * 60)
    logger.info("WALK-FORWARD VALIDATION  (train=%dm, test=%dm)", train_months, test_months)
    logger.info("=" * 60)

    all_oos_sharpes: List[float] = []

    # Use first stock with enough data as a representative for walk-forward demo
    wf_ticker = max(stock_dfs, key=lambda t: len(stock_dfs[t]))
    wf_df = stock_data[wf_ticker]
    folds = walk_forward_split(wf_df, train_months=train_months, test_months=test_months)
    logger.info("Walk-forward folds for %s: %d folds", wf_ticker, len(folds))

    for fold_idx, (train_df, test_df) in enumerate(folds):
        logger.info(
            "--- Fold %d: train %s -> %s  |  test %s -> %s ---",
            fold_idx + 1,
            train_df.index[0].strftime("%Y-%m-%d"),
            train_df.index[-1].strftime("%Y-%m-%d"),
            test_df.index[0].strftime("%Y-%m-%d"),
            test_df.index[-1].strftime("%Y-%m-%d"),
        )

        # Build features for this fold
        train_bars, train_feats = build_features_from_df(train_df, symbol=wf_ticker)
        test_bars, test_feats = build_features_from_df(test_df, symbol=wf_ticker)

        if len(train_bars) < 60 or len(test_bars) < 20:
            logger.warning(
                "  Fold %d skipped: insufficient data (train=%d, test=%d)",
                fold_idx + 1,
                len(train_bars),
                len(test_bars),
            )
            continue

        # Normalise using training stats
        train_mean = train_feats.mean(axis=0)
        train_std = train_feats.std(axis=0)
        train_std[train_std < 1e-8] = 1.0
        train_feats_n = (train_feats - train_mean) / train_std
        test_feats_n = (test_feats - train_mean) / train_std

        # Train a PPO agent on this fold
        fold_timesteps = total_timesteps // max(len(folds), 1)
        fold_timesteps = max(fold_timesteps, 10_000)

        train_env = TradingGymEnv(train_bars, train_feats_n)
        fold_model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=min(2048, len(train_bars) - 1),
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0,
        )
        fold_model.learn(total_timesteps=fold_timesteps)

        # Evaluate out-of-sample
        metrics = evaluate_agent(fold_model, test_bars, test_feats_n)
        all_oos_sharpes.append(metrics["sharpe_ratio"])

        logger.info(
            "  Fold %d OOS:  Sharpe=%.3f  Return=%.2f%%  MaxDD=%.2f%%  Trades=%d",
            fold_idx + 1,
            metrics["sharpe_ratio"],
            metrics["total_return_pct"],
            metrics["max_drawdown_pct"],
            metrics["n_trades"],
        )

    if all_oos_sharpes:
        avg_sharpe = np.mean(all_oos_sharpes)
        std_sharpe = np.std(all_oos_sharpes)
        logger.info("=" * 60)
        logger.info(
            "WALK-FORWARD SUMMARY:  Mean OOS Sharpe = %.3f (+/- %.3f)  over %d folds",
            avg_sharpe,
            std_sharpe,
            len(all_oos_sharpes),
        )
        logger.info("=" * 60)
    else:
        logger.warning("No walk-forward folds completed.")

    # ----- 4. Final model: train on ALL data -----
    logger.info("Training FINAL model on all %d stocks for %d timesteps ...", len(stock_bars), total_timesteps)

    all_bars_list = list(stock_bars.values())
    all_feats_list = list(stock_features.values())

    # Normalise across all stocks
    all_feats_normed, feat_means, feat_stds = normalise_features(all_feats_list)

    vec_env = make_vec_env(
        bars_list=all_bars_list,
        features_list=all_feats_normed,
        n_envs=n_envs,
        use_subproc=False,
    )

    final_model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )

    t0 = time.time()
    final_model.learn(total_timesteps=total_timesteps)
    elapsed = time.time() - t0
    logger.info(
        "Final training completed in %.1f seconds (%.0f steps/sec)", elapsed, total_timesteps / (elapsed + 1e-6)
    )

    # ----- 5. Save -----
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "rl_agent")
    final_model.save(model_path)
    logger.info("Model saved to %s.zip", model_path)

    # Save normalisation stats for inference
    stats_path = os.path.join(MODELS_DIR, "rl_agent_stats.npz")
    np.savez(stats_path, means=feat_means, stds=feat_stds)
    logger.info("Feature stats saved to %s", stats_path)

    # ----- 6. Quick final evaluation on each stock -----
    logger.info("=" * 60)
    logger.info("FINAL MODEL EVALUATION (in-sample)")
    logger.info("=" * 60)
    all_sharpes = []
    for ticker, bars_arr in stock_bars.items():
        feats_n = (stock_features[ticker] - feat_means) / feat_stds
        metrics = evaluate_agent(final_model, bars_arr, feats_n)
        all_sharpes.append(metrics["sharpe_ratio"])
        logger.info(
            "  %-15s  Sharpe=%.3f  Return=%.2f%%  MaxDD=%.2f%%  Trades=%d",
            ticker,
            metrics["sharpe_ratio"],
            metrics["total_return_pct"],
            metrics["max_drawdown_pct"],
            metrics["n_trades"],
        )

    logger.info("-" * 60)
    logger.info("Mean in-sample Sharpe: %.3f", np.mean(all_sharpes))
    logger.info("=" * 60)

    vec_env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train PPO RL agent on NIFTY50 stocks (Yahoo Finance data)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  PYTHONPATH=. python scripts/train_rl.py
  PYTHONPATH=. python scripts/train_rl.py --quick
  PYTHONPATH=. python scripts/train_rl.py --timesteps 1000000 --symbols RELIANCE.NS,TCS.NS,INFY.NS
        """,
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps for the final model (default: 500000)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated Yahoo Finance tickers (default: NIFTY50 top 30)",
    )
    parser.add_argument(
        "--train-months",
        type=int,
        default=6,
        help="Walk-forward training window in months (default: 6)",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=2,
        help="Walk-forward test window in months (default: 2)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="PPO learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick sanity run: 50K steps, 2 envs, 10 tickers",
    )

    args = parser.parse_args()
    tickers = args.symbols.split(",") if args.symbols else None

    train(
        tickers=tickers,
        total_timesteps=args.timesteps,
        train_months=args.train_months,
        test_months=args.test_months,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
