"""
Transformer-based price predictor.
Multi-head self-attention on price sequences for directional prediction.
Implements BasePredictor contract for EnsembleEngine integration.
"""
import logging
import math
import os
from typing import Any, Dict, Optional

import numpy as np

from .base import BasePredictor, PredictionOutput
from .lstm_predictor import FEATURE_KEYS, NUM_FEATURES, SEQ_LEN

logger = logging.getLogger(__name__)


def _try_import_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        return None, None


class TransformerModel:
    """Transformer encoder for time-series binary classification."""

    def __init__(self, input_size: int = NUM_FEATURES, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dropout: float = 0.2):
        torch, nn = _try_import_torch()
        if torch is None:
            self._model = None
            self._torch = None
            return

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _PositionalEncoding(nn.Module):
            def __init__(self_pe, d_model, max_len=200):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
                pe = pe.unsqueeze(0)
                self_pe.register_buffer('pe', pe)

            def forward(self_pe, x):
                return x + self_pe.pe[:, :x.size(1), :]

        class _TransformerNet(nn.Module):
            def __init__(self_net):
                super().__init__()
                self_net.input_proj = nn.Linear(input_size, d_model)
                self_net.pos_encoder = _PositionalEncoding(d_model, max_len=200)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                    dropout=dropout, batch_first=True,
                )
                self_net.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self_net.dropout = nn.Dropout(dropout)
                self_net.fc1 = nn.Linear(d_model, 32)
                self_net.relu = nn.ReLU()
                self_net.fc2 = nn.Linear(32, 1)
                self_net.sigmoid = nn.Sigmoid()

            def forward(self_net, x):
                x = self_net.input_proj(x)
                x = self_net.pos_encoder(x)
                x = self_net.transformer(x)
                # Use mean pooling over sequence
                x = x.mean(dim=1)
                x = self_net.dropout(x)
                x = self_net.relu(self_net.fc1(x))
                x = self_net.sigmoid(self_net.fc2(x))
                return x.squeeze(-1)

        self._model = _TransformerNet().to(self._device)
        self._model.eval()

    @property
    def available(self) -> bool:
        return self._model is not None

    def predict_proba(self, sequence: np.ndarray) -> float:
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

    def load(self, path: str) -> bool:
        if not self.available:
            return False
        try:
            torch = self._torch
            state_dict = torch.load(path, map_location=self._device, weights_only=True)
            self._model.load_state_dict(state_dict)
            self._model.eval()
            logger.info("Transformer model loaded from %s", path)
            return True
        except Exception as e:
            logger.warning("Failed to load Transformer model from %s: %s", path, e)
            return False


class TransformerPredictor(BasePredictor):
    """Transformer for next-step direction prediction."""

    model_id = "transformer_ts"
    version = "v1"

    def __init__(self, model_path: str = ""):
        self._transformer = TransformerModel()
        self.path = model_path
        self._loaded = False
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

        if model_path and os.path.exists(model_path):
            self._loaded = self._transformer.load(model_path)
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
        normed = np.zeros_like(seq)
        for t in range(len(seq)):
            window = seq[:t + 1]
            mean = window.mean(axis=0)
            std = window.std(axis=0)
            std = np.where(std < 1e-8, 1.0, std)
            normed[t] = (seq[t] - mean) / std
        return normed

    def predict(self, features: Dict[str, float], context: Optional[Dict[str, Any]] = None) -> PredictionOutput:
        if not self._transformer.available:
            return PredictionOutput(
                prob_up=0.5, expected_return=0.0, confidence=0.0,
                model_id=self.model_id, version=self.version,
                metadata={"reason": "torch_not_available"},
            )

        seq = None
        if context:
            seq = context.get("sequence")
            if seq is None:
                feature_history = context.get("feature_history")
                if feature_history and len(feature_history) >= SEQ_LEN:
                    rows = []
                    for fdict in feature_history[-SEQ_LEN:]:
                        rows.append([fdict.get(k, 0.0) for k in FEATURE_KEYS])
                    seq = np.array(rows, dtype=np.float32)

        if seq is None or len(seq) < SEQ_LEN:
            return PredictionOutput(
                prob_up=0.5, expected_return=0.0, confidence=0.0,
                model_id=self.model_id, version=self.version,
                metadata={"reason": "no_sequence_data"},
            )

        seq_norm = self._normalize_sequence(seq[-SEQ_LEN:])
        prob_up = self._transformer.predict_proba(seq_norm)

        confidence = min(1.0, abs(prob_up - 0.5) * 2.0)
        if not self._loaded:
            confidence *= 0.3

        expected_return = (prob_up - 0.5) * 0.02

        return PredictionOutput(
            prob_up=prob_up,
            expected_return=expected_return,
            confidence=confidence,
            model_id=self.model_id,
            version=self.version,
            metadata={"seq_len": SEQ_LEN, "loaded": self._loaded},
        )
