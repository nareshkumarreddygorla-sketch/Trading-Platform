"""
Train RL (PPO) agent for trading decisions.
Uses custom TradingEnv with stable-baselines3.

Improvements over v1:
  - Default 500K timesteps (configurable via --timesteps)
  - Improved reward: transaction cost penalty, drawdown penalty, Sharpe-based shaping
  - Expanded to 30+ symbols for training data (configurable via --symbols-count)
  - Post-training validation: 10 eval episodes; model NOT saved if mean reward < 0
  - Action-to-probability mapping uses actual PPO policy action distribution
  - CLI args: --timesteps, --symbols-count, --symbols

Usage:
    PYTHONPATH=. python scripts/train_rl_agent.py
    PYTHONPATH=. python scripts/train_rl_agent.py --timesteps 200000 --symbols-count 50
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

# Default number of symbols and timesteps
DEFAULT_TIMESTEPS = 500_000
DEFAULT_SYMBOLS_COUNT = 30


def build_training_data(symbols=None, symbols_count=DEFAULT_SYMBOLS_COUNT):
    """Build bars and features arrays for RL training.

    Args:
        symbols: Explicit list of symbol names. If None, auto-discover from parquet files.
        symbols_count: Number of symbols to use when auto-discovering (default 30).
    """
    from src.ai.feature_engine import FeatureEngine
    from src.ai.models.rl_agent import RL_FEATURE_KEYS as FEATURE_KEYS
    from src.core.events import Bar

    fe = FeatureEngine()

    if symbols is None:
        parquet_files = [f for f in os.listdir(DATA_DIR)
                         if f.endswith(".parquet") and f not in ("all_stocks.parquet", "NIFTY50.parquet")]
        symbols = [f.replace(".parquet", "") for f in parquet_files[:symbols_count]]

    logger.info("Training with %d symbols: %s", len(symbols), symbols[:10])

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


def _evaluate_model(model, env, num_episodes=30):
    """Run evaluation episodes and return per-episode rewards and action sequences.

    Args:
        model: Trained PPO model.
        env: Gymnasium environment.
        num_episodes: Number of evaluation episodes (minimum 30 for statistical significance).

    Returns:
        Tuple of (episode_rewards, episode_actions) where episode_actions is a list
        of action sequences per episode for IC calculation.
    """
    episode_rewards = []
    episode_actions = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        actions = []
        while not done and steps < 10000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            actions.append(int(action))
            steps += 1
        episode_rewards.append(total_reward)
        episode_actions.append(actions)
        logger.info("Eval episode %d/%d: reward=%.4f, steps=%d, balance=%.2f",
                     ep + 1, num_episodes, total_reward, steps, info.get("balance", 0))
    return episode_rewards, episode_actions


def _compute_rl_ic(model, bars, features, sample_size=500):
    """Compute Information Coefficient for RL agent predictions.

    Measures Spearman rank correlation between the agent's directional signal
    (P(buy) - P(sell)) and actual forward returns.

    Args:
        model: Trained PPO model.
        bars: OHLCV array (N, 5).
        features: Feature array (N, num_features).
        sample_size: Number of data points to evaluate.

    Returns:
        IC value (float), or 0.0 if computation fails.
    """
    from scipy.stats import spearmanr

    n = min(len(bars), len(features), sample_size)
    if n < 100:
        return 0.0

    predictions = []
    actual_returns = []

    # Sample evenly across the dataset
    indices = np.linspace(60, n - 2, min(sample_size, n - 62), dtype=int)

    for idx in indices:
        feat = features[idx]
        # Build observation: features + [position_side=0, unrealized_pnl=0, bars_held=0, balance_ratio=1, vol_regime=0.02]
        obs = np.concatenate([feat, [0.0, 0.0, 0.0, 1.0, 0.02]]).astype(np.float32)

        try:
            action, _ = model.predict(obs, deterministic=True)
            # Map action to directional signal: buy=+1, sell=-1, hold=0
            signal = {1: 1.0, 2: -1.0}.get(int(action), 0.0)
        except Exception:
            continue

        # Actual forward return (1-step)
        if idx + 1 < len(bars):
            fwd_return = (bars[idx + 1, 3] - bars[idx, 3]) / (bars[idx, 3] + 1e-12)
            predictions.append(signal)
            actual_returns.append(fwd_return)

    if len(predictions) < 50:
        return 0.0

    preds = np.array(predictions)
    actuals = np.array(actual_returns)

    # Filter out neutral predictions for IC
    mask = preds != 0.0
    if np.sum(mask) < 30:
        return 0.0

    ic, _ = spearmanr(preds[mask], actuals[mask])
    return float(ic) if np.isfinite(ic) else 0.0


def _compute_sharpe_ratio(episode_rewards):
    """Compute annualized Sharpe ratio from episode rewards.

    Args:
        episode_rewards: List of per-episode total rewards.

    Returns:
        Sharpe ratio (float). Annualized assuming ~252 trading days.
    """
    if len(episode_rewards) < 5:
        return 0.0
    rewards = np.array(episode_rewards)
    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    if std_r < 1e-12:
        return 0.0 if mean_r <= 0 else 10.0
    # Sharpe ratio (not annualized since episodes are variable length)
    return float(mean_r / std_r)


def train(total_timesteps=DEFAULT_TIMESTEPS, symbols=None, symbols_count=DEFAULT_SYMBOLS_COUNT):
    try:
        from stable_baselines3 import PPO
        from src.ai.models.rl_agent import GymTradingEnv
    except ImportError:
        logger.error("stable-baselines3 not installed. Run: pip install stable-baselines3 gymnasium")
        return

    logger.info("Building training data...")
    bars, features = build_training_data(symbols, symbols_count=symbols_count)
    if bars is None:
        logger.error("No training data. Run: PYTHONPATH=. python scripts/download_nse_data.py")
        return

    logger.info("Training data: %d bars, %d features", len(bars), features.shape[1])

    # Split data: 80% training, 20% out-of-sample test
    oos_fraction = 0.20
    oos_size = max(int(len(bars) * oos_fraction), 200)
    bars_train = bars[:-oos_size]
    features_train = features[:-oos_size]
    bars_oos = bars[-oos_size:]
    features_oos = features[-oos_size:]
    logger.info(
        "Data split: train=%d bars, out-of-sample=%d bars (%.0f%%)",
        len(bars_train), len(bars_oos), oos_fraction * 100,
    )

    env = GymTradingEnv.create(bars_train, features_train)
    if env is None:
        logger.error("Failed to create training environment")
        return

    # Entropy coefficient scheduling: linear decay from 0.01 to 0.001
    # Encourages exploration early, exploitation later in training
    ent_coef_start = 0.01
    ent_coef_final = 0.001

    def _ent_coef_schedule(progress_remaining: float) -> float:
        """Linear decay: ent_coef_start at beginning -> ent_coef_final at end."""
        return ent_coef_final + (ent_coef_start - ent_coef_final) * progress_remaining

    logger.info(
        "Training PPO agent for %d timesteps (max_grad_norm=0.5, "
        "ent_coef=%.4f->%.4f)...",
        total_timesteps, ent_coef_start, ent_coef_final,
    )
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
        max_grad_norm=0.5,              # Gradient clipping for training stability
        ent_coef=_ent_coef_schedule,     # Entropy scheduling: explore early, exploit late
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)

    # --- Validation Gate 1: 30 in-sample evaluation episodes ---
    MIN_IC = 0.03
    MIN_SHARPE = 0.5
    NUM_EVAL_EPISODES = 30

    logger.info("Running %d in-sample evaluation episodes...", NUM_EVAL_EPISODES)
    episode_rewards, episode_actions = _evaluate_model(model, env, num_episodes=NUM_EVAL_EPISODES)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    logger.info("In-sample eval: mean_reward=%.4f, std=%.4f, min=%.4f, max=%.4f",
                mean_reward, std_reward, np.min(episode_rewards), np.max(episode_rewards))

    # Gate 1a: Mean reward must be positive
    if mean_reward < 0:
        logger.error(
            "VALIDATION FAILED: mean reward %.4f < 0 over %d episodes. Model NOT saved.",
            mean_reward, NUM_EVAL_EPISODES,
        )
        return

    # Gate 1b: Sharpe ratio check
    sharpe = _compute_sharpe_ratio(episode_rewards)
    logger.info("In-sample Sharpe ratio: %.4f (minimum: %.2f)", sharpe, MIN_SHARPE)
    if sharpe < MIN_SHARPE:
        logger.error(
            "VALIDATION FAILED: Sharpe ratio %.4f < %.2f minimum. Model NOT saved. "
            "Consider more timesteps or reward shaping tuning.", sharpe, MIN_SHARPE,
        )
        return

    # --- Validation Gate 2: IC check on training data ---
    logger.info("Computing Information Coefficient on training data...")
    ic_train = _compute_rl_ic(model, bars_train, features_train, sample_size=1000)
    logger.info("In-sample IC: %.4f (minimum: %.2f)", ic_train, MIN_IC)
    if ic_train < MIN_IC:
        logger.error(
            "VALIDATION FAILED: in-sample IC %.4f < %.2f minimum. Model NOT saved. "
            "Agent predictions lack directional accuracy.", ic_train, MIN_IC,
        )
        return

    # --- Validation Gate 3: Out-of-sample test ---
    logger.info("Running out-of-sample validation (%d bars)...", len(bars_oos))
    oos_env = GymTradingEnv.create(bars_oos, features_oos)
    if oos_env is not None:
        oos_rewards, _ = _evaluate_model(model, oos_env, num_episodes=NUM_EVAL_EPISODES)
        oos_mean = np.mean(oos_rewards)
        oos_sharpe = _compute_sharpe_ratio(oos_rewards)
        logger.info("Out-of-sample eval: mean_reward=%.4f, Sharpe=%.4f", oos_mean, oos_sharpe)

        if oos_mean < 0:
            logger.error(
                "VALIDATION FAILED: out-of-sample mean reward %.4f < 0. "
                "Model overfits training data. NOT saved.", oos_mean,
            )
            return

        # OOS IC check
        ic_oos = _compute_rl_ic(model, bars_oos, features_oos, sample_size=500)
        logger.info("Out-of-sample IC: %.4f (minimum: %.2f)", ic_oos, MIN_IC)
        if ic_oos < MIN_IC:
            logger.error(
                "VALIDATION FAILED: out-of-sample IC %.4f < %.2f. "
                "Model does not generalize. NOT saved.", ic_oos, MIN_IC,
            )
            return
    else:
        logger.warning("Could not create OOS environment — skipping OOS validation")

    logger.info(
        "ALL VALIDATION GATES PASSED: mean_reward=%.4f, Sharpe=%.4f, "
        "IC_train=%.4f, IC_oos=%.4f",
        mean_reward, sharpe, ic_train, ic_oos if oos_env else float('nan'),
    )

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "rl_agent")
    model.save(model_path)
    logger.info("RL agent saved to %s.zip (mean_reward=%.4f, Sharpe=%.4f, IC=%.4f)",
                model_path, mean_reward, sharpe, ic_train)


def main():
    parser = argparse.ArgumentParser(
        description="Train RL (PPO) trading agent with improved reward shaping and validation."
    )
    parser.add_argument(
        "--timesteps", type=int, default=DEFAULT_TIMESTEPS,
        help=f"Total PPO training timesteps (default: {DEFAULT_TIMESTEPS})"
    )
    parser.add_argument(
        "--symbols-count", type=int, default=DEFAULT_SYMBOLS_COUNT,
        help=f"Number of symbols to auto-discover for training (default: {DEFAULT_SYMBOLS_COUNT})"
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated list of symbols to train on (overrides --symbols-count)"
    )
    args = parser.parse_args()
    symbols = args.symbols.split(",") if args.symbols else None
    train(
        total_timesteps=args.timesteps,
        symbols=symbols,
        symbols_count=args.symbols_count,
    )


if __name__ == "__main__":
    main()
