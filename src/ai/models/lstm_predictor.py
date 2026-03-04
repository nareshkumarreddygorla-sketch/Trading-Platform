"""
LSTM time-series predictor for price direction.
Two-layer LSTM with dropout, trained on FeatureEngine output sequences.
Implements BasePredictor contract for EnsembleEngine integration.
"""
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BasePredictor, PredictionOutput

logger = logging.getLogger(__name__)

# Feature ordering must match training
FEATURE_KEYS = [
    "returns_1", "returns_5", "returns_10", "returns_20",
    "rolling_volatility", "atr", "bollinger_bandwidth",
    "rsi", "ema_spread", "macd_line", "macd_signal", "macd_histogram",
    "stochastic_k", "stochastic_d", "adx",
    "momentum_5", "momentum_10", "momentum_20", "roc_10",
    "volume_spike", "obv_slope", "vwap_distance",
    "bollinger_pct_b", "price_position", "gap_pct",
    "candle_body_ratio", "candle_upper_shadow", "candle_lower_shadow", "candle_engulfing",
    "williams_r", "mfi",
]

NUM_FEATURES = len(FEATURE_KEYS)
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
    """PyTorch LSTM network for binary classification (price up/down)."""

    def __init__(self, input_size: int = NUM_FEATURES, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        torch, nn = _try_import_torch()
        if torch is None:
            self._model = None
            self._torch = None
            return

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                self_net.dropout = nn.Dropout(dropout)
                self_net.fc1 = nn.Linear(hidden_size, 64)
                self_net.relu = nn.ReLU()
                self_net.fc2 = nn.Linear(64, 1)
                self_net.sigmoid = nn.Sigmoid()

            def forward(self_net, x):
                lstm_out, _ = self_net.lstm(x)
                last_hidden = lstm_out[:, -1, :]
                out = self_net.dropout(last_hidden)
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
    version = "v2"

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
                except Exception:
                    pass

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
            row = [fdict.get(k, 0.0) for k in FEATURE_KEYS]
            rows.append(row)
        return np.array(rows, dtype=np.float32)

    def predict(self, features: Dict[str, float], context: Optional[Dict[str, Any]] = None) -> PredictionOutput:
        if not self._lstm.available:
            return PredictionOutput(
                prob_up=0.5, expected_return=0.0, confidence=0.0,
                model_id=self.model_id, version=self.version,
                metadata={"reason": "torch_not_available"},
            )

        # Get sequence from context
        seq = None
        if context:
            seq = context.get("sequence")  # Pre-built (T, F) array
            if seq is None:
                # Build from feature history
                feature_history = context.get("feature_history")
                if feature_history and len(feature_history) >= SEQ_LEN:
                    seq = self._build_sequence_from_features(feature_history)

        # Fallback: build single-step from features dict
        if seq is None:
            return PredictionOutput(
                prob_up=0.5, expected_return=0.0, confidence=0.0,
                model_id=self.model_id, version=self.version,
                metadata={"reason": "no_sequence_data"},
            )

        if len(seq) < SEQ_LEN:
            return PredictionOutput(
                prob_up=0.5, expected_return=0.0, confidence=0.0,
                model_id=self.model_id, version=self.version,
                metadata={"reason": "sequence_too_short", "seq_len": len(seq)},
            )

        # Normalize and predict
        seq_norm = self._normalize_sequence(seq)
        prob_up = self._lstm.predict_proba(seq_norm)

        # Confidence: distance from 0.5 (uncertain) scaled to 0-1
        confidence = min(1.0, abs(prob_up - 0.5) * 2.0)
        # If model not loaded (random weights), reduce confidence
        if not self._loaded:
            confidence *= 0.3

        # Expected return estimate based on probability
        expected_return = (prob_up - 0.5) * 0.02  # +-1% expected

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
