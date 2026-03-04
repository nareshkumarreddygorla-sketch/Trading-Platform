"""
Deep Reinforcement Learning trading agent.
Uses PPO from stable-baselines3 with a custom TradingEnv.
Implements BasePredictor contract for EnsembleEngine integration.
"""
import logging
import os
from typing import Any, Dict, Optional

import numpy as np

from .base import BasePredictor, PredictionOutput
from .lstm_predictor import FEATURE_KEYS, NUM_FEATURES, SEQ_LEN

logger = logging.getLogger(__name__)

# RL observation: latest features + position info
OBS_SIZE = NUM_FEATURES + 3  # features + [position_side, unrealized_pnl_pct, bars_held]


def _try_import_sb3():
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
        return gym, PPO
    except ImportError:
        return None, None


class TradingEnv:
    """
    Custom trading environment for RL training.
    Uses gymnasium interface without inheriting from gym.Env to avoid
    import issues when gymnasium is not installed.
    """

    def __init__(self, bars_data: np.ndarray, features_data: np.ndarray,
                 initial_balance: float = 100000.0, commission: float = 0.001):
        """
        Args:
            bars_data: (N, 5) array of OHLCV
            features_data: (N, num_features) array from FeatureEngine
            initial_balance: starting capital
            commission: transaction cost as fraction
        """
        self.bars = bars_data
        self.features = features_data
        self.initial_balance = initial_balance
        self.commission = commission
        self.n_steps = len(bars_data)

        # State
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # -1, 0, 1
        self.entry_price = 0.0
        self.bars_held = 0
        self.total_pnl = 0.0
        self.peak_balance = initial_balance

    @property
    def observation_space_shape(self):
        return (OBS_SIZE,)

    @property
    def action_space_n(self):
        return 3  # 0=hold, 1=buy, 2=sell

    def reset(self):
        self.current_step = SEQ_LEN
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.bars_held = 0
        self.total_pnl = 0.0
        self.peak_balance = self.initial_balance
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        if self.current_step >= len(self.features):
            feats = np.zeros(NUM_FEATURES, dtype=np.float32)
        else:
            feats = self.features[self.current_step].astype(np.float32)

        position_side = float(self.position)
        unrealized_pnl = 0.0
        if self.position != 0 and self.entry_price > 0:
            current_price = self.bars[min(self.current_step, len(self.bars) - 1), 3]  # close
            unrealized_pnl = (current_price / self.entry_price - 1.0) * self.position

        extra = np.array([position_side, unrealized_pnl, self.bars_held / 100.0], dtype=np.float32)
        return np.concatenate([feats, extra])

    def step(self, action: int):
        """Execute action: 0=hold, 1=buy, 2=sell. Returns (obs, reward, done, info)."""
        current_price = self.bars[min(self.current_step, len(self.bars) - 1), 3]  # close
        reward = 0.0
        done = False

        # Execute action
        if action == 1 and self.position <= 0:  # Buy
            if self.position == -1:  # Close short
                pnl = (self.entry_price - current_price) / self.entry_price
                pnl -= self.commission
                reward = pnl
                self.total_pnl += pnl * self.balance
                self.balance *= (1 + pnl)
            self.position = 1
            self.entry_price = current_price
            self.bars_held = 0
            self.balance *= (1 - self.commission)

        elif action == 2 and self.position >= 0:  # Sell
            if self.position == 1:  # Close long
                pnl = (current_price - self.entry_price) / self.entry_price
                pnl -= self.commission
                reward = pnl
                self.total_pnl += pnl * self.balance
                self.balance *= (1 + pnl)
            self.position = -1
            self.entry_price = current_price
            self.bars_held = 0
            self.balance *= (1 - self.commission)

        else:  # Hold
            if self.position != 0 and self.entry_price > 0:
                unrealized = (current_price / self.entry_price - 1.0) * self.position
                reward = unrealized * 0.01  # Small reward for unrealized gains
            self.bars_held += 1

        # Track peak for drawdown penalty
        self.peak_balance = max(self.peak_balance, self.balance)
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        if drawdown > 0.05:
            reward -= drawdown * 0.5  # Penalize drawdown

        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        return self._get_obs(), reward, done, {"balance": self.balance, "pnl": self.total_pnl}


class GymTradingEnv:
    """Wrapper that creates a proper gymnasium environment for SB3 training."""

    @staticmethod
    def create(bars_data: np.ndarray, features_data: np.ndarray, **kwargs):
        gym, _ = _try_import_sb3()
        if gym is None:
            return None

        env = TradingEnv(bars_data, features_data, **kwargs)

        class _GymEnv(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self_env):
                super().__init__()
                self_env.env = env
                self_env.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=env.observation_space_shape, dtype=np.float32,
                )
                self_env.action_space = gym.spaces.Discrete(env.action_space_n)

            def reset(self_env, seed=None, options=None):
                super().reset(seed=seed)
                obs = self_env.env.reset()
                return obs, {}

            def step(self_env, action):
                obs, reward, done, info = self_env.env.step(action)
                return obs, reward, done, False, info

        return _GymEnv()


class RLPredictor(BasePredictor):
    """RL agent for trading decisions. Maps actions to PredictionOutput.

    Supports:
      - Loading pre-trained PPO model from disk
      - Training from bar data via `train_from_bars()`
      - Graceful degradation: returns confidence=0 when model not loaded
    """

    model_id = "rl_ppo"
    version = "v1"

    def __init__(self, model_path: str = ""):
        self.path = model_path
        self._model = None
        self._loaded = False

        _, PPO = _try_import_sb3()
        if PPO is not None and model_path and os.path.exists(model_path):
            try:
                self._model = PPO.load(model_path)
                self._loaded = True
                logger.info("RL agent loaded from %s", model_path)
            except Exception as e:
                logger.warning("Failed to load RL agent from %s: %s", model_path, e)

    def train_from_bars(
        self,
        bars_data: np.ndarray,
        features_data: np.ndarray,
        total_timesteps: int = 50_000,
        save_path: Optional[str] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 256,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        verbose: int = 0,
    ) -> bool:
        """
        Train PPO agent from bar/feature data.

        Args:
            bars_data: (N, 5) OHLCV array
            features_data: (N, num_features) feature matrix
            total_timesteps: PPO training timesteps
            save_path: Where to save trained model. Defaults to self.path or models/rl_ppo.zip
            learning_rate: PPO learning rate
            n_steps: PPO rollout buffer size
            batch_size: PPO minibatch size
            n_epochs: PPO epochs per update
            gamma: Discount factor
            verbose: SB3 verbosity

        Returns:
            True if training succeeded and model was saved.
        """
        gym, PPO = _try_import_sb3()
        if gym is None or PPO is None:
            logger.warning("RL train_from_bars: stable-baselines3/gymnasium not installed")
            return False

        if len(bars_data) < 100:
            logger.warning("RL train_from_bars: insufficient bars (%d < 100)", len(bars_data))
            return False

        try:
            # Create gymnasium environment
            env = GymTradingEnv.create(bars_data, features_data)
            if env is None:
                logger.warning("RL train_from_bars: failed to create GymTradingEnv")
                return False

            # Train PPO
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                n_steps=min(n_steps, len(bars_data) // 2),
                batch_size=min(batch_size, n_steps),
                n_epochs=n_epochs,
                gamma=gamma,
                verbose=verbose,
            )

            logger.info("RL training started (timesteps=%d, bars=%d)", total_timesteps, len(bars_data))
            model.learn(total_timesteps=total_timesteps)
            logger.info("RL training complete")

            # Save model
            save_to = save_path or self.path or os.path.join("models", "rl_ppo.zip")
            os.makedirs(os.path.dirname(save_to) if os.path.dirname(save_to) else ".", exist_ok=True)
            model.save(save_to)
            logger.info("RL model saved to %s", save_to)

            # Hot-swap: load the newly trained model
            self._model = model
            self._loaded = True
            self.path = save_to
            return True

        except Exception as e:
            logger.exception("RL train_from_bars failed: %s", e)
            return False

    def predict(self, features: Dict[str, float], context: Optional[Dict[str, Any]] = None) -> PredictionOutput:
        if self._model is None:
            return PredictionOutput(
                prob_up=0.5, expected_return=0.0, confidence=0.0,
                model_id=self.model_id, version=self.version,
                metadata={"reason": "model_not_loaded"},
            )

        # Build observation
        feat_array = np.array([features.get(k, 0.0) for k in FEATURE_KEYS], dtype=np.float32)

        # Add position info from context
        position_side = 0.0
        unrealized_pnl = 0.0
        bars_held = 0.0
        if context:
            position_side = float(context.get("position_side", 0))
            unrealized_pnl = float(context.get("unrealized_pnl_pct", 0.0))
            bars_held = float(context.get("bars_held", 0)) / 100.0

        obs = np.concatenate([feat_array, [position_side, unrealized_pnl, bars_held]])

        try:
            action, _ = self._model.predict(obs, deterministic=True)
            action = int(action)
        except Exception as e:
            logger.debug("RL predict failed: %s", e)
            return PredictionOutput(
                prob_up=0.5, expected_return=0.0, confidence=0.0,
                model_id=self.model_id, version=self.version,
                metadata={"reason": f"predict_error: {e}"},
            )

        # Map action to PredictionOutput
        # action: 0=hold, 1=buy, 2=sell
        if action == 1:  # Buy
            prob_up = 0.75
            expected_return = 0.01
            confidence = 0.7
        elif action == 2:  # Sell
            prob_up = 0.25
            expected_return = -0.01
            confidence = 0.7
        else:  # Hold
            prob_up = 0.5
            expected_return = 0.0
            confidence = 0.3

        if not self._loaded:
            confidence *= 0.3

        return PredictionOutput(
            prob_up=prob_up,
            expected_return=expected_return,
            confidence=confidence,
            model_id=self.model_id,
            version=self.version,
            metadata={"action": action, "loaded": self._loaded},
        )
