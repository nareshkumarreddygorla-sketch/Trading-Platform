"""
Deep Reinforcement Learning trading agent.
Uses PPO from stable-baselines3 with a custom TradingEnv.
Implements BasePredictor contract for EnsembleEngine integration.

v2 improvements:
  - Improved reward: transaction cost penalty, progressive drawdown, Sharpe shaping
  - predict() uses actual PPO policy action distribution instead of hardcoded probs
"""
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch as _torch
except ImportError:
    _torch = None

from .base import BasePredictor, PredictionOutput, estimate_empirical_return
from .lstm_predictor import BASE_FEATURE_KEYS, SEQ_LEN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RL-specific features: focused on market microstructure and execution context.
# RL agents need real-time market state to learn optimal trade timing.
# ---------------------------------------------------------------------------
RL_SPECIFIC_FEATURES = [
    # Bid-ask spread proxy (2)
    "bid_ask_spread_proxy", "effective_spread_proxy",
    # Intraday volatility features (2)
    "intraday_volatility", "intraday_range_ratio",
    # Time-of-day encoding (2)
    "time_of_day_sin", "time_of_day_cos",
    # Position heat: how long current position has been held (1)
    "position_heat",
    # Execution quality proxies (2)
    "slippage_estimate", "market_impact_estimate",
    # Urgency / momentum-at-execution (2)
    "tick_velocity", "quote_intensity",
    # Inventory risk (1)
    "inventory_risk_score",
]

RL_FEATURE_KEYS = BASE_FEATURE_KEYS + RL_SPECIFIC_FEATURES
RL_NUM_FEATURES = len(RL_FEATURE_KEYS)  # 39 base + 12 RL-specific = 51

# RL observation: latest features + position info
OBS_SIZE = RL_NUM_FEATURES + 5  # features + [position_side, unrealized_pnl_pct, bars_held, balance_ratio, vol_regime]


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

    Reward shaping (v2):
      - Transaction cost penalty on every trade
      - Drawdown penalty (progressive, starting at 3%)
      - Sharpe-based reward shaping via rolling reward tracking
      - Small negative reward for excessive holding without position
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

        # Sharpe reward tracking: rolling window of step returns
        self._reward_history: list = []
        self._sharpe_window = 50  # steps for rolling Sharpe estimate

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
        # Keep cross-episode reward history for meaningful Sharpe calculation
        # Only trim to last 500 entries to bound memory
        self._reward_history = self._reward_history[-500:] if self._reward_history else []
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        if self.current_step >= len(self.features):
            feats = np.zeros(RL_NUM_FEATURES, dtype=np.float32)
        else:
            feats = self.features[self.current_step].astype(np.float32)

        position_side = float(self.position)
        unrealized_pnl = 0.0
        if self.position != 0 and self.entry_price > 0:
            current_price = self.bars[min(self.current_step, len(self.bars) - 1), 3]  # close
            unrealized_pnl = (current_price / self.entry_price - 1.0) * self.position

        # Include portfolio context: balance ratio and vol regime
        balance_ratio = self.balance / self.initial_balance  # Relative capital context
        recent_vol = 0.02  # Default
        lookback = min(20, self.current_step)
        if lookback >= 5:
            recent_rets = []
            for i in range(max(1, self.current_step - lookback), self.current_step):
                if i < len(self.bars) and i > 0:
                    ret = (self.bars[i, 3] - self.bars[i-1, 3]) / (self.bars[i-1, 3] + 1e-10)
                    recent_rets.append(ret)
            if len(recent_rets) >= 5:
                recent_vol = float(np.std(recent_rets))
        extra = np.array([position_side, unrealized_pnl, self.bars_held / 100.0, balance_ratio, recent_vol], dtype=np.float32)
        return np.concatenate([feats, extra])

    def _sharpe_bonus(self) -> float:
        """Rolling Sharpe-based reward shaping with cap and decay to prevent feedback loop.

        Improvements over v1:
          - Hard cap at 0.1 (absolute value) to prevent reward explosion
          - Exponential decay: bonus shrinks as episode progresses to reduce
            compounding feedback between Sharpe bonus and reward history
          - Minimum window size of 2x sharpe_window before bonus kicks in
        """
        # Require 2x window before computing to ensure stable estimate
        if len(self._reward_history) < self._sharpe_window * 2:
            return 0.0
        recent = np.array(self._reward_history[-self._sharpe_window:])
        mean_r = np.mean(recent)
        std_r = np.std(recent)
        if std_r < 1e-8:
            return 0.0
        sharpe = mean_r / std_r
        # Base bonus proportional to rolling Sharpe
        raw_bonus = sharpe * 0.005
        # Exponential decay: reduce bonus magnitude as episode progresses
        # This prevents the Sharpe bonus from dominating late in episodes
        decay_factor = max(0.1, np.exp(-len(self._reward_history) / 5000.0))
        decayed_bonus = raw_bonus * decay_factor
        # Hard cap at 0.1 to prevent reward explosion / feedback loop
        return float(np.clip(decayed_bonus, -0.1, 0.1))

    def step(self, action: int):
        """Execute action: 0=hold, 1=buy, 2=sell. Returns (obs, reward, done, info)."""
        current_price = self.bars[min(self.current_step, len(self.bars) - 1), 3]  # close
        reward = 0.0
        done = False
        traded = False

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
            traded = True

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
            traded = True

        else:  # Hold
            if self.position != 0 and self.entry_price > 0:
                # Simple mark-to-market delta (no vol scaling to avoid holding bias)
                prev_price = self.bars[max(0, self.current_step - 1), 3]
                step_return = (current_price - prev_price) / (prev_price + 1e-10) * self.position
                reward = step_return * 0.01  # Small incremental reward
            self.bars_held += 1

        # --- Reward shaping components ---
        # NOTE: Transaction costs already deducted in buy/sell logic above (lines 190/197/203/210)
        # Do NOT add additional penalty here (was double-counting before)

        # Drawdown penalty (progressive, starting at 3%)
        self.peak_balance = max(self.peak_balance, self.balance)
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        if drawdown > 0.03:
            reward -= drawdown * 0.8  # Stronger drawdown penalty than v1

        # 3. Sharpe-based reward shaping
        # Record RAW trading reward (before Sharpe bonus) to avoid feedback loop
        raw_reward = reward
        self._reward_history.append(raw_reward)
        reward += self._sharpe_bonus()

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
    version = "v2"

    @classmethod
    def get_feature_keys(cls) -> List[str]:
        """Return the full feature list for this model (base + RL-specific).

        This classmethod allows the training pipeline and feature engine to
        query the exact feature set without instantiating the model.
        """
        return list(RL_FEATURE_KEYS)

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def __repr__(self) -> str:
        return f"<RLPredictor model_id={self.model_id!r} version={self.version!r} loaded={self._loaded}>"

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
        total_timesteps: int = 500_000,
        save_path: Optional[str] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        max_grad_norm: float = 0.5,
        ent_coef: float = 0.01,
        ent_coef_final: float = 0.001,
        verbose: int = 0,
    ) -> bool:
        """
        Train PPO agent from bar/feature data.

        Args:
            bars_data: (N, 5) OHLCV array
            features_data: (N, num_features) feature matrix
            total_timesteps: PPO training timesteps (default 500K for convergence)
            save_path: Where to save trained model. Defaults to self.path or models/rl_ppo.zip
            learning_rate: PPO learning rate
            n_steps: PPO rollout buffer size (increased to 2048 for stability)
            batch_size: PPO minibatch size
            n_epochs: PPO epochs per update
            gamma: Discount factor
            max_grad_norm: Gradient clipping threshold (prevents exploding gradients)
            ent_coef: Initial entropy coefficient (exploration bonus)
            ent_coef_final: Final entropy coefficient (scheduled decay)
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

            # Entropy coefficient scheduling: linear decay from ent_coef to ent_coef_final
            # This encourages exploration early and exploitation later
            def _ent_coef_schedule(progress_remaining: float) -> float:
                """Linear decay: ent_coef at start -> ent_coef_final at end."""
                return ent_coef_final + (ent_coef - ent_coef_final) * progress_remaining

            # Train PPO with gradient clipping and entropy scheduling
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                n_steps=min(n_steps, len(bars_data) // 2),
                batch_size=min(batch_size, n_steps),
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=0.95,
                clip_range=0.2,
                max_grad_norm=max_grad_norm,  # Gradient clipping for training stability
                ent_coef=_ent_coef_schedule,   # Entropy scheduling: explore early, exploit late
                verbose=verbose,
            )

            logger.info(
                "RL training started (timesteps=%d, bars=%d, max_grad_norm=%.2f, "
                "ent_coef=%.4f->%.4f)",
                total_timesteps, len(bars_data), max_grad_norm, ent_coef, ent_coef_final,
            )
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

    def _get_action_probs(self, obs: np.ndarray) -> Optional[np.ndarray]:
        """Extract actual action probabilities from the PPO policy network.

        Returns:
            Array of shape (3,) with probabilities for [hold, buy, sell],
            or None if extraction fails.
        """
        try:
            if _torch is None:
                return None
            # PPO policy exposes get_distribution() which gives the Categorical dist
            dist = self._model.policy.get_distribution(
                self._model.policy.obs_to_tensor(obs.reshape(1, -1))[0]
            )
            probs = dist.distribution.probs.detach().cpu().numpy().flatten()
            return probs  # shape (3,) for [hold, buy, sell]
        except Exception as e:
            logger.debug("Failed to extract action probs: %s", e)
            return None

    def predict(self, features: Dict[str, float], context: Optional[Dict[str, Any]] = None) -> Optional[PredictionOutput]:
        if self._model is None:
            return None

        # Build observation
        feat_array = np.array([features.get(k, np.nan) for k in RL_FEATURE_KEYS], dtype=np.float32)
        # Replace NaN with 0.0 for RL observation (RL needs valid float inputs)
        feat_array = np.nan_to_num(feat_array, nan=0.0)

        # Add position info from context
        position_side = 0.0
        unrealized_pnl = 0.0
        bars_held = 0.0
        if context:
            position_side = float(context.get("position_side", 0))
            unrealized_pnl = float(context.get("unrealized_pnl_pct", 0.0))
            bars_held = float(context.get("bars_held", 0)) / 100.0

        balance_ratio = float(context.get("balance_ratio", 1.0)) if context else 1.0
        vol_regime = float(context.get("vol_regime", 0.02)) if context else 0.02
        obs = np.concatenate([feat_array, [position_side, unrealized_pnl, bars_held, balance_ratio, vol_regime]])

        try:
            action, _ = self._model.predict(obs, deterministic=True)
            action = int(action)
        except Exception as e:
            logger.debug("RL predict failed: %s", e)
            return None

        # --- Use actual PPO policy action distribution ---
        # Extract action probabilities from the policy network instead of
        # hardcoded buy=0.75/sell=0.25/hold=0.5 mapping.
        # Actions: 0=hold, 1=buy, 2=sell
        action_probs = self._get_action_probs(obs)

        if action_probs is not None and len(action_probs) == 3:
            p_hold = float(action_probs[0])
            p_buy = float(action_probs[1])
            p_sell = float(action_probs[2])

            # prob_up = P(buy) / (P(buy) + P(sell)), i.e. directional conviction
            directional_total = p_buy + p_sell
            if directional_total > 1e-8:
                prob_up = p_buy / directional_total
            else:
                prob_up = 0.5  # All probability on hold -> neutral

            # Confidence = 1 - P(hold), scaled: high hold prob means low confidence
            confidence = float(np.clip(1.0 - p_hold, 0.0, 1.0))
            # Further scale by how decisive the directional bet is
            if directional_total > 1e-8:
                decisiveness = abs(p_buy - p_sell) / directional_total
                confidence *= (0.5 + 0.5 * decisiveness)

            metadata = {
                "action": action,
                "action_probs": {"hold": p_hold, "buy": p_buy, "sell": p_sell},
                "loaded": self._loaded,
            }
        else:
            # Fallback: policy probs unavailable (e.g. torch not installed)
            # Use neutral 0.5 base — avoid hardcoded bullish/bearish bias
            if action == 1:  # Buy
                prob_up = 0.5
                confidence = 0.3  # Low confidence without policy distribution
            elif action == 2:  # Sell
                prob_up = 0.5
                confidence = 0.3
            else:  # Hold
                prob_up = 0.5
                confidence = 0.1
            metadata = {"action": action, "loaded": self._loaded, "fallback": True}

        if not self._loaded:
            confidence *= 0.3

        # Validate prediction meets minimum quality standards
        if not self.validate_prediction(prob_up, confidence):
            return None

        # Expected return estimate using empirical calibration
        expected_return = estimate_empirical_return(prob_up, self._empirical_returns)
        if expected_return is None:
            return None

        return PredictionOutput(
            prob_up=prob_up,
            expected_return=expected_return,
            confidence=confidence,
            model_id=self.model_id,
            version=self.version,
            metadata=metadata,
        )
