"""
Gymnasium-compatible trading environment for RL agent training.

State space: FeatureEngine features + position info (side, unrealized P&L, cash ratio)
Action space: Discrete(3) -- {0: hold, 1: buy, 2: sell}
Reward: realised P&L minus realistic India transaction costs (STT, brokerage, GST, etc.)

Supports:
  - Configurable episode length (one stock's price series)
  - Vectorized environments via stable-baselines3 SubprocVecEnv / DummyVecEnv
  - Drawdown penalty and position-holding cost for risk shaping

Usage:
    from src.ai.models.rl_environment import TradingGymEnv, make_vec_env
    env = TradingGymEnv(bars_ohlcv, features_array)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(1)  # buy
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.ai.models.lstm_predictor import NUM_FEATURES
from src.costs.india_costs import IndiaCostCalculator

logger = logging.getLogger(__name__)

# Observation = features + [position_side, unrealized_pnl_pct, cash_ratio]
_EXTRA_OBS = 3
OBS_DIM = NUM_FEATURES + _EXTRA_OBS


# ---------------------------------------------------------------------------
# Core gymnasium environment
# ---------------------------------------------------------------------------
class TradingGymEnv(gym.Env):
    """
    A single-stock, single-episode trading environment.

    Parameters
    ----------
    bars_ohlcv : np.ndarray, shape (T, 5)
        Columns: open, high, low, close, volume.
    features : np.ndarray, shape (T, NUM_FEATURES)
        Pre-computed features aligned with bars_ohlcv (row-for-row).
    initial_cash : float
        Starting capital in INR.
    max_shares : int
        Maximum position size in number of shares.
    episode_length : int or None
        If set, each episode is capped at this many steps.
        If None, an episode spans the full price series.
    product_type : str
        "INTRADAY" or "DELIVERY" -- passed to IndiaCostCalculator.
    exchange : str
        "NSE" or "BSE".
    reward_scaling : float
        Multiply raw reward to keep gradients in a reasonable range.
    drawdown_penalty : float
        Coefficient for penalising portfolio drawdown exceeding 5%.
    hold_cost_per_step : float
        Tiny penalty per step while holding a position, to discourage
        aimless holding when the model is uncertain.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        bars_ohlcv: np.ndarray,
        features: np.ndarray,
        initial_cash: float = 1_000_000.0,
        max_shares: int = 100,
        episode_length: int | None = None,
        product_type: str = "INTRADAY",
        exchange: str = "NSE",
        reward_scaling: float = 1.0,
        drawdown_penalty: float = 0.5,
        hold_cost_per_step: float = 0.0001,
    ):
        super().__init__()
        assert len(bars_ohlcv) == len(features), (
            f"bars ({len(bars_ohlcv)}) and features ({len(features)}) must be same length"
        )
        assert bars_ohlcv.shape[1] >= 4, "bars_ohlcv must have >= 4 columns (O,H,L,C)"

        self._bars = bars_ohlcv.astype(np.float64)
        self._features = features.astype(np.float32)
        self._n_total = len(bars_ohlcv)
        self._initial_cash = float(initial_cash)
        self._max_shares = max_shares
        self._episode_length = episode_length
        self._product_type = product_type
        self._exchange = exchange
        self._reward_scaling = reward_scaling
        self._drawdown_penalty = drawdown_penalty
        self._hold_cost_per_step = hold_cost_per_step

        # India transaction cost model
        self._cost_calc = IndiaCostCalculator()

        # Gymnasium spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell

        # State variables (initialised in reset)
        self._step_idx: int = 0
        self._start_idx: int = 0
        self._end_idx: int = 0
        self._cash: float = 0.0
        self._shares: int = 0  # positive = long, negative = short
        self._entry_price: float = 0.0
        self._peak_portfolio: float = 0.0
        self._portfolio_values: list[float] = []

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # Determine episode window
        if self._episode_length is not None and self._episode_length < self._n_total:
            max_start = self._n_total - self._episode_length
            self._start_idx = self.np_random.integers(0, max_start + 1)
            self._end_idx = self._start_idx + self._episode_length
        else:
            self._start_idx = 0
            self._end_idx = self._n_total

        self._step_idx = self._start_idx
        self._cash = self._initial_cash
        self._shares = 0
        self._entry_price = 0.0
        self._peak_portfolio = self._initial_cash
        self._portfolio_values = [self._initial_cash]

        obs = self._build_obs()
        info = self._build_info()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one step.

        Actions:
            0 -- hold (do nothing)
            1 -- buy  (go long / close short then go long)
            2 -- sell (go short / close long then go short)

        Returns:
            obs, reward, terminated, truncated, info
        """
        price = self._close_price(self._step_idx)
        reward = 0.0
        trade_cost = 0.0

        # ----- Execute action -----
        if action == 1:  # BUY
            if self._shares < 0:
                # Close short first
                reward += self._close_position(price)
            if self._shares == 0:
                # Open long
                trade_cost = self._transaction_cost("BUY", price, self._max_shares)
                self._cash -= price * self._max_shares + trade_cost
                self._shares = self._max_shares
                self._entry_price = price

        elif action == 2:  # SELL
            if self._shares > 0:
                # Close long first
                reward += self._close_position(price)
            if self._shares == 0:
                # Open short
                trade_cost = self._transaction_cost("SELL", price, self._max_shares)
                self._cash += price * self._max_shares - trade_cost
                self._shares = -self._max_shares
                self._entry_price = price

        else:  # HOLD
            # Small penalty for holding a position to discourage idle holding
            if self._shares != 0:
                reward -= self._hold_cost_per_step

        # ----- Portfolio value and reward shaping -----
        portfolio_val = self._portfolio_value(price)
        self._portfolio_values.append(portfolio_val)

        # Track peak for drawdown penalty
        self._peak_portfolio = max(self._peak_portfolio, portfolio_val)
        drawdown = (self._peak_portfolio - portfolio_val) / (self._peak_portfolio + 1e-12)
        if drawdown > 0.05:
            reward -= drawdown * self._drawdown_penalty

        # Step-level return (log return of portfolio)
        prev_val = self._portfolio_values[-2] if len(self._portfolio_values) >= 2 else self._initial_cash
        if prev_val > 0:
            step_return = (portfolio_val - prev_val) / (prev_val + 1e-12)
            reward += step_return

        reward *= self._reward_scaling

        # ----- Advance -----
        self._step_idx += 1
        terminated = self._step_idx >= self._end_idx - 1
        truncated = False

        # Force-close at episode end
        if terminated and self._shares != 0:
            end_price = self._close_price(min(self._step_idx, self._end_idx - 1))
            reward += self._close_position(end_price)

        obs = self._build_obs()
        info = self._build_info()
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _close_price(self, idx: int) -> float:
        """Return close price at given index, clamped to valid range."""
        idx = min(max(idx, 0), self._n_total - 1)
        return float(self._bars[idx, 3])

    def _transaction_cost(self, side: str, price: float, qty: int) -> float:
        """Compute India-realistic transaction cost in INR."""
        notional = abs(price * qty)
        if notional <= 0:
            return 0.0
        breakdown = self._cost_calc.calculate(
            side=side,
            notional=notional,
            product_type=self._product_type,
            exchange=self._exchange,
        )
        return breakdown.total

    def _close_position(self, price: float) -> float:
        """
        Close the current position at *price*.
        Returns realised P&L as fraction of initial capital (after costs).
        """
        if self._shares == 0:
            return 0.0

        qty = abs(self._shares)
        if self._shares > 0:
            # Closing long -> sell
            side = "SELL"
            gross_pnl = (price - self._entry_price) * qty
        else:
            # Closing short -> buy
            side = "BUY"
            gross_pnl = (self._entry_price - price) * qty

        cost = self._transaction_cost(side, price, qty)
        net_pnl = gross_pnl - cost

        # Update cash
        if self._shares > 0:
            self._cash += price * qty - cost
        else:
            self._cash -= price * qty + cost

        self._shares = 0
        self._entry_price = 0.0

        # Return as fraction of initial capital
        return net_pnl / (self._initial_cash + 1e-12)

    def _portfolio_value(self, price: float) -> float:
        """Mark-to-market portfolio value."""
        return self._cash + self._shares * price

    def _build_obs(self) -> np.ndarray:
        """Construct observation vector: features + position info."""
        idx = min(self._step_idx, self._n_total - 1)
        feats = self._features[idx].copy()

        # Position side: -1 (short), 0 (flat), +1 (long)
        if self._shares > 0:
            position_side = 1.0
        elif self._shares < 0:
            position_side = -1.0
        else:
            position_side = 0.0

        # Unrealised P&L as percentage of entry price
        unrealized_pnl_pct = 0.0
        if self._shares != 0 and self._entry_price > 0:
            current_price = self._close_price(idx)
            if self._shares > 0:
                unrealized_pnl_pct = (current_price - self._entry_price) / (self._entry_price + 1e-12)
            else:
                unrealized_pnl_pct = (self._entry_price - current_price) / (self._entry_price + 1e-12)

        # Cash ratio: fraction of initial capital held as cash
        cash_ratio = self._cash / (self._initial_cash + 1e-12)

        extra = np.array([position_side, unrealized_pnl_pct, cash_ratio], dtype=np.float32)
        return np.concatenate([feats, extra])

    def _build_info(self) -> dict[str, Any]:
        """Return diagnostic info dict."""
        idx = min(self._step_idx, self._n_total - 1)
        price = self._close_price(idx)
        portfolio_val = self._portfolio_value(price)
        return {
            "step": self._step_idx - self._start_idx,
            "price": price,
            "cash": self._cash,
            "shares": self._shares,
            "portfolio_value": portfolio_val,
            "return_pct": (portfolio_val / self._initial_cash - 1.0) * 100,
        }


# ---------------------------------------------------------------------------
# Vectorized environment factory
# ---------------------------------------------------------------------------
def make_vec_env(
    bars_list: list[np.ndarray],
    features_list: list[np.ndarray],
    n_envs: int = 4,
    use_subproc: bool = False,
    **env_kwargs,
) -> Any:
    """
    Create a vectorized environment for parallel RL training.

    Parameters
    ----------
    bars_list : list of np.ndarray
        One OHLCV array per stock/segment.
    features_list : list of np.ndarray
        Corresponding feature arrays.
    n_envs : int
        Number of parallel environments.
    use_subproc : bool
        If True, use SubprocVecEnv (separate processes).
        If False, use DummyVecEnv (single process, faster for small envs).
    **env_kwargs
        Passed to TradingGymEnv constructor.

    Returns
    -------
    stable_baselines3.common.vec_env.VecEnv
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    n_segments = len(bars_list)
    assert n_segments == len(features_list), "bars_list and features_list must match"
    assert n_segments > 0, "Need at least one data segment"

    def _make_env(rank: int):
        def _init():
            seg_idx = rank % n_segments
            env = TradingGymEnv(
                bars_ohlcv=bars_list[seg_idx],
                features=features_list[seg_idx],
                **env_kwargs,
            )
            return env

        return _init

    env_fns = [_make_env(i) for i in range(n_envs)]

    if use_subproc and n_envs > 1:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


# ---------------------------------------------------------------------------
# Convenience: environment registration (optional)
# ---------------------------------------------------------------------------
def register_trading_env():
    """Register TradingGymEnv with gymnasium's registry (optional convenience)."""
    try:
        gym.register(
            id="IndiaTrading-v0",
            entry_point="src.ai.models.rl_environment:TradingGymEnv",
        )
        logger.info("Registered IndiaTrading-v0 with gymnasium")
    except gym.error.Error:
        pass  # Already registered
