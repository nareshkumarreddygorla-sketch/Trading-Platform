"""
Train RL (PPO) agent for trading decisions.
Uses custom TradingEnv with stable-baselines3.

Usage:
    PYTHONPATH=. python scripts/train_rl_agent.py
    PYTHONPATH=. python scripts/train_rl_agent.py --timesteps 200000
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "nse_historical")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def build_training_data(symbols=None):
    """Build bars and features arrays for RL training."""
    from src.ai.feature_engine import FeatureEngine
    from src.ai.models.lstm_predictor import FEATURE_KEYS
    from src.core.events import Bar

    fe = FeatureEngine()

    if symbols is None:
        parquet_files = [f for f in os.listdir(DATA_DIR)
                         if f.endswith(".parquet") and f not in ("all_stocks.parquet", "NIFTY50.parquet")]
        symbols = [f.replace(".parquet", "") for f in parquet_files[:10]]  # Use 10 stocks for RL

    all_bars = []
    all_features = []

    for symbol in symbols:
        path = os.path.join(DATA_DIR, f"{symbol}.parquet")
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path)
        if len(df) < 100:
            continue

        # OHLCV array
        bars_arr = df[["open", "high", "low", "close", "volume"]].values.astype(np.float32)

        # Build features
        bars = []
        for idx, row in df.iterrows():
            bars.append(Bar(
                symbol=symbol, exchange="NSE", interval="1d",
                open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]),
                volume=float(row["volume"]),
                ts=idx if hasattr(idx, 'tzinfo') else datetime.now(timezone.utc),
            ))

        feature_list = []
        for i in range(20, len(bars)):
            features = fe.build_features(bars[max(0, i - 100):i + 1])
            row_feats = [features.get(k, 0.0) for k in FEATURE_KEYS]
            feature_list.append(row_feats)

        if len(feature_list) < 100:
            continue

        # Align bars and features (features start at index 20)
        aligned_bars = bars_arr[20:]
        min_len = min(len(aligned_bars), len(feature_list))
        all_bars.append(aligned_bars[:min_len])
        all_features.append(np.array(feature_list[:min_len], dtype=np.float32))
        logger.info("Loaded %s: %d bars", symbol, min_len)

    if not all_bars:
        return None, None

    # Concatenate all stocks (train across stocks)
    bars_combined = np.concatenate(all_bars, axis=0)
    features_combined = np.concatenate(all_features, axis=0)

    # Normalize features
    means = features_combined.mean(axis=0)
    stds = features_combined.std(axis=0)
    stds[stds < 1e-8] = 1.0
    features_combined = (features_combined - means) / stds

    return bars_combined, features_combined


def train(total_timesteps=100000, symbols=None):
    try:
        from stable_baselines3 import PPO
        from src.ai.models.rl_agent import GymTradingEnv
    except ImportError:
        logger.error("stable-baselines3 not installed. Run: pip install stable-baselines3 gymnasium")
        return

    logger.info("Building training data...")
    bars, features = build_training_data(symbols)
    if bars is None:
        logger.error("No training data. Run: PYTHONPATH=. python scripts/download_nse_data.py")
        return

    logger.info("Training data: %d bars, %d features", len(bars), features.shape[1])

    env = GymTradingEnv.create(bars, features)
    if env is None:
        logger.error("Failed to create training environment")
        return

    logger.info("Training PPO agent for %d timesteps...", total_timesteps)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "rl_agent")
    model.save(model_path)
    logger.info("RL agent saved to %s.zip", model_path)

    # Quick evaluation
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False
    while not done and steps < 5000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        steps += 1

    logger.info("Evaluation: total_reward=%.4f over %d steps, final_balance=%.2f",
                total_reward, steps, info.get("balance", 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--symbols", type=str, default=None)
    args = parser.parse_args()
    symbols = args.symbols.split(",") if args.symbols else None
    train(total_timesteps=args.timesteps, symbols=symbols)


if __name__ == "__main__":
    main()
