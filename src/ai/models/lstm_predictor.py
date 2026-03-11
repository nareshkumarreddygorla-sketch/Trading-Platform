"""
LSTM time-series predictor for price direction.
Two-layer LSTM with dropout, trained on FeatureEngine output sequences.
Implements BasePredictor contract for EnsembleEngine integration.
"""
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BasePredictor, PredictionOutput, estimate_empirical_return

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared base features: common to ALL deep learning models (39 features).
# These provide the foundational feature set every model can rely on.
# ---------------------------------------------------------------------------
BASE_FEATURE_KEYS = [
    # Returns (4)
    "returns_1", "returns_5", "returns_10", "returns_20",
    # Volatility (3)
    "rolling_volatility", "atr", "bollinger_bandwidth",
    # Trend indicators (8)
    "rsi", "ema_spread", "macd_line", "macd_signal", "macd_histogram",
    "stochastic_k", "stochastic_d", "adx",
    # Momentum (4)
    "momentum_5", "momentum_10", "momentum_20", "roc_10",
    # Volume indicators (3)
    "volume_spike", "obv_slope", "vwap_distance",
    # Price structure (3)
    "bollinger_pct_b", "price_position", "gap_pct",
    # Candlestick patterns (4)
    "candle_body_ratio", "candle_upper_shadow", "candle_lower_shadow", "candle_engulfing",
    # Additional oscillators (2)
    "williams_r", "mfi",
    # Microstructure features (6)
    "minute_of_day_sin", "minute_of_day_cos",
    "hurst_exponent", "volume_profile_deviation",
    "vol_of_vol", "rolling_vol_20",
    # Raw reference (2, normalized during training)
    "close", "volume",
]

# ---------------------------------------------------------------------------
# LSTM-specific features: focused on sequential/temporal patterns.
# LSTMs excel at capturing autocorrelation and regime persistence in time.
# ---------------------------------------------------------------------------
LSTM_SPECIFIC_FEATURES = [
    # Lagged returns for autoregressive structure (5)
    "lagged_return_1", "lagged_return_2", "lagged_return_3",
    "lagged_return_5", "lagged_return_10",
    # Autocorrelation features (3)
    "autocorr_5", "autocorr_10", "autocorr_20",
    # Regime indicators (2)
    "regime_hmm_state", "regime_volatility_zscore",
    # Orderflow imbalance proxies (2)
    "orderflow_imbalance", "tick_direction_ratio",
]

# Backward-compatible alias: legacy code importing FEATURE_KEYS still works,
# but now returns the LSTM-augmented feature list.
FEATURE_KEYS = BASE_FEATURE_KEYS + LSTM_SPECIFIC_FEATURES
NUM_FEATURES = len(FEATURE_KEYS)  # 39 base + 12 LSTM-specific = 51
SEQ_LEN = 60


def _try_import_torch():
    """Lazy import torch to avoid startup failures if not installed."""
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        return None, None


class LSTMModel:
    """PyTorch LSTM network with self-attention for binary classification (price up/down)."""

    def __init__(self, input_size: int = NUM_FEATURES, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        torch, nn = _try_import_torch()
        if torch is None:
            self._model = None
            self._torch = None
            return

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _Attention(nn.Module):
            """Simple self-attention over LSTM sequence outputs.
            Computes attention weights across time steps and returns
            a weighted context vector."""

            def __init__(self_attn, hidden_size):
                super().__init__()
                self_attn.attn_weights = nn.Linear(hidden_size, 1)

            def forward(self_attn, lstm_output):
                # lstm_output: (batch, seq_len, hidden_size)
                scores = self_attn.attn_weights(lstm_output).squeeze(-1)  # (batch, seq_len)
                attn_weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)
                context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch, hidden_size)
                return context, attn_weights

        class _LSTMNet(nn.Module):
            def __init__(self_net):
                super().__init__()
                self_net.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self_net.attention = _Attention(hidden_size)
                self_net.dropout = nn.Dropout(dropout)
                self_net.fc1 = nn.Linear(hidden_size, 64)
                self_net.relu = nn.ReLU()
                self_net.fc2 = nn.Linear(64, 1)
                self_net.sigmoid = nn.Sigmoid()

            def forward(self_net, x):
                lstm_out, _ = self_net.lstm(x)  # (batch, seq_len, hidden_size)
                context, _ = self_net.attention(lstm_out)  # (batch, hidden_size)
                out = self_net.dropout(context)
                out = self_net.relu(self_net.fc1(out))
                out = self_net.sigmoid(self_net.fc2(out))
                return out.squeeze(-1)

        self._model = _LSTMNet().to(self._device)
        self._model.eval()

    @property
    def available(self) -> bool:
        return self._model is not None

    def predict_proba(self, sequence: np.ndarray) -> float:
        """Predict P(up) from (seq_len, features) array. Returns float 0-1."""
        if not self.available:
            return 0.5
        torch = self._torch
        with torch.no_grad():
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self._device)
            prob = self._model(x).item()
        return float(np.clip(prob, 0.0, 1.0))

    def save(self, path: str) -> None:
        if self.available:
            self._torch.save(self._model.state_dict(), path)
            logger.info("LSTM model saved to %s", path)

    def load(self, path: str) -> bool:
        if not self.available:
            return False
        try:
            torch = self._torch
            state_dict = torch.load(path, map_location=self._device, weights_only=True)
            # Pre-validate shapes to avoid load_state_dict hang (joblib/torch interaction)
            model_sd = self._model.state_dict()
            for key in model_sd:
                if key in state_dict and model_sd[key].shape != state_dict[key].shape:
                    logger.warning(
                        "Failed to load LSTM model from %s: shape mismatch for %s: "
                        "checkpoint=%s vs model=%s",
                        path, key, state_dict[key].shape, model_sd[key].shape,
                    )
                    return False
            missing = set(model_sd.keys()) - set(state_dict.keys())
            if missing:
                logger.warning("Failed to load LSTM model from %s: missing keys: %s", path, missing)
                return False
            self._model.load_state_dict(state_dict)
            self._model.eval()
            logger.info("LSTM model loaded from %s", path)
            return True
        except Exception as e:
            logger.warning("Failed to load LSTM model from %s: %s", path, e)
            return False


class LSTMPredictor(BasePredictor):
    """LSTM for next-step return / direction. Expects sequence in context['sequence']."""

    model_id = "lstm_ts"
    version = "v3"

    @classmethod
    def get_feature_keys(cls) -> List[str]:
        """Return the full feature list for this model (base + LSTM-specific).

        This classmethod allows the training pipeline and feature engine to
        query the exact feature set without instantiating the model.
        """
        return list(BASE_FEATURE_KEYS + LSTM_SPECIFIC_FEATURES)

    @property
    def is_ready(self) -> bool:
        return self._lstm.available

    def __repr__(self) -> str:
        return f"<LSTMPredictor model_id={self.model_id!r} version={self.version!r} available={self._lstm.available}>"

    def __init__(self, model_path: str = ""):
        self._lstm = LSTMModel()
        self.path = model_path
        self._loaded = False
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

        if model_path and os.path.exists(model_path):
            self._loaded = self._lstm.load(model_path)
            # Load normalization stats if available
            stats_path = model_path.replace(".pt", "_stats.npz")
            if os.path.exists(stats_path):
                try:
                    stats = np.load(stats_path)
                    self._feature_means = stats["means"]
                    self._feature_stds = stats["stds"]
                except Exception as e:
                    logger.warning("Failed to load normalization stats from %s: %s — falling back to expanding window normalization", stats_path, e)

    def _normalize_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Z-score normalize using training statistics (no lookahead bias)."""
        if self._feature_means is not None and self._feature_stds is not None:
            # Use training-time statistics (correct approach — no future data leakage)
            stds = np.where(self._feature_stds < 1e-8, 1.0, self._feature_stds)
            return (seq - self._feature_means) / stds
        # Fallback: expanding window normalization (prevents lookahead bias)
        # Each timestep normalized using only data available up to that point
        normed = np.zeros_like(seq)
        for t in range(len(seq)):
            window = seq[:t + 1]
            mean = window.mean(axis=0)
            std = window.std(axis=0)
            std = np.where(std < 1e-8, 1.0, std)
            normed[t] = (seq[t] - mean) / std
        return normed

    def _build_sequence_from_features(self, feature_history: List[Dict[str, float]]) -> Optional[np.ndarray]:
        """Convert list of feature dicts to (seq_len, num_features) array.
        Uses FEATURE_KEYS ordering; missing features default to 0.0.
        Logs warning if too many features are missing for reliable prediction."""
        if len(feature_history) < SEQ_LEN:
            return None
        rows = []
        # Check feature availability on first dict
        sample = feature_history[-1]
        available = [k for k in FEATURE_KEYS if k in sample]
        if len(available) < len(FEATURE_KEYS) * 0.7:
            logger.warning("LSTM: Only %d/%d expected features available (missing: %s)",
                          len(available), len(FEATURE_KEYS),
                          [k for k in FEATURE_KEYS if k not in sample][:5])
        for fdict in feature_history[-SEQ_LEN:]:
            row = [fdict.get(k, np.nan) for k in FEATURE_KEYS]
            rows.append(row)
        arr = np.array(rows, dtype=np.float32)
        # Replace NaN with 0.0 after creating array (NaN-aware normalization handles this better)
        arr = np.nan_to_num(arr, nan=0.0)
        return arr

    def predict(self, features: Dict[str, float], context: Optional[Dict[str, Any]] = None) -> Optional[PredictionOutput]:
        if not self._lstm.available:
            return None

        # Get sequence from context
        seq = None
        if context:
            seq = context.get("sequence")  # Pre-built (T, F) array
            if seq is not None and not isinstance(seq, np.ndarray):
                seq = np.array(seq, dtype=np.float32)
            if seq is None:
                # Build from feature history
                feature_history = context.get("feature_history")
                if feature_history and len(feature_history) >= SEQ_LEN:
                    seq = self._build_sequence_from_features(feature_history)

        # Fallback: build single-step from features dict
        if seq is None:
            return None

        if len(seq) < SEQ_LEN:
            return None

        # Normalize and predict
        seq_norm = self._normalize_sequence(seq)
        prob_up = self._lstm.predict_proba(seq_norm)

        # Confidence: distance from 0.5 (uncertain) scaled to 0-1
        confidence = min(1.0, abs(prob_up - 0.5) * 2.0)
        # If model not loaded (random weights), reduce confidence
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
            metadata={
                "seq_len": len(seq),
                "loaded": self._loaded,
            },
        )
