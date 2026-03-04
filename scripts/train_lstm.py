"""
Train LSTM model for price direction prediction using walk-forward validation.

Walk-forward scheme:
    - 60-day rolling train window, 10-day test window
    - Rolling forward across all available data
    - Computes Information Coefficient (IC) per window
    - Requires average IC > 0.02 across windows for model acceptance

Target: P(price increase > 0.5% within next 5 bars)

Data source: Yahoo Finance historical daily OHLCV for NIFTY50 stocks.

Usage:
    PYTHONPATH=. python scripts/train_lstm.py
    PYTHONPATH=. python scripts/train_lstm.py --quick
    PYTHONPATH=. python scripts/train_lstm.py --symbols RELIANCE.NS,INFY.NS,TCS.NS
    PYTHONPATH=. python scripts/train_lstm.py --epochs 50 --lr 0.0005
"""
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_lstm")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SEQ_LEN = 20  # 20-day input sequences (as specified)
FORECAST_HORIZON = 5  # predict 5-bar forward return
THRESHOLD_PCT = 0.005  # 0.5% price increase threshold
MIN_HISTORY_DAYS = 126  # ~6 months of trading days minimum

# Default NIFTY50 tickers for Yahoo Finance (suffixed with .NS)
NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "HCLTECH.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    "NESTLEIND.NS", "ONGC.NS", "NTPC.NS", "TATAMOTORS.NS", "M&M.NS",
    "POWERGRID.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "BAJAJFINSV.NS", "COALINDIA.NS", "TECHM.NS", "HDFCLIFE.NS", "GRASIM.NS",
    "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS", "BPCL.NS", "EICHERMOT.NS",
    "APOLLOHOSP.NS", "SBILIFE.NS", "TATACONSUM.NS", "BRITANNIA.NS", "INDUSINDBK.NS",
    "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "HINDALCO.NS", "UPL.NS", "LTIM.NS",
]

# Quick mode: smaller subset for fast iteration
QUICK_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "ITC.NS", "HCLTECH.NS", "WIPRO.NS", "LT.NS",
]


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------
@dataclass
class WindowMetrics:
    """Metrics for a single walk-forward window."""
    window_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_loss: float
    test_loss: float
    train_accuracy: float
    test_accuracy: float
    ic: float  # Information Coefficient (Spearman rank correlation)
    n_train: int
    n_test: int


@dataclass
class TrainingReport:
    """Aggregate training report across all walk-forward windows."""
    model_version: str
    timestamp: str
    total_windows: int
    avg_train_loss: float
    avg_test_loss: float
    avg_train_accuracy: float
    avg_test_accuracy: float
    avg_ic: float
    ic_std: float
    stability_score: float  # fraction of windows with IC > 0
    ic_values: List[float]
    window_metrics: List[dict]
    symbols_used: List[str]
    seq_len: int
    forecast_horizon: int
    threshold_pct: float
    passed_ic_threshold: bool
    total_training_time_sec: float


# ---------------------------------------------------------------------------
# Data fetching via Yahoo Finance
# ---------------------------------------------------------------------------
def fetch_yahoo_data(
    symbols: List[str],
    period: str = "2y",
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """
    Download historical OHLCV data from Yahoo Finance for given symbols.

    Returns dict of symbol -> DataFrame with columns:
        Open, High, Low, Close, Volume (DatetimeIndex).
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error(
            "yfinance not installed. Install with: pip install yfinance"
        )
        sys.exit(1)

    data: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []

    logger.info("Fetching data for %d symbols from Yahoo Finance (period=%s)...", len(symbols), period)

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
            if df is None or len(df) < MIN_HISTORY_DAYS:
                logger.warning(
                    "Skipping %s: insufficient data (%d bars, need %d)",
                    symbol, len(df) if df is not None else 0, MIN_HISTORY_DAYS,
                )
                failed.append(symbol)
                continue

            # Ensure clean columns
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.dropna(inplace=True)

            if len(df) < MIN_HISTORY_DAYS:
                logger.warning("Skipping %s: too few bars after dropna (%d)", symbol, len(df))
                failed.append(symbol)
                continue

            data[symbol] = df
            logger.info("  %s: %d bars [%s .. %s]", symbol, len(df),
                        df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d"))

        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", symbol, exc)
            failed.append(symbol)

    if failed:
        logger.info("Failed/skipped %d symbols: %s", len(failed), ", ".join(failed[:10]))

    if not data:
        logger.error("No data fetched. Check network connection and symbol names.")
        sys.exit(1)

    logger.info("Successfully fetched data for %d / %d symbols.", len(data), len(symbols))
    return data


# ---------------------------------------------------------------------------
# Feature computation (aligned with FeatureEngine / FEATURE_KEYS)
# ---------------------------------------------------------------------------
def compute_features_from_df(df: pd.DataFrame, symbol: str) -> Optional[np.ndarray]:
    """
    Compute features matching FEATURE_KEYS from lstm_predictor.py using
    the FeatureEngine. Returns (n_bars, n_features) array or None.

    We build Bar objects from the DataFrame rows, then call FeatureEngine
    on expanding windows to avoid lookahead bias.
    """
    from src.ai.feature_engine import FeatureEngine
    from src.ai.models.lstm_predictor import FEATURE_KEYS
    from src.core.events import Bar, Exchange

    fe = FeatureEngine()

    # Build Bar objects
    bars: List[Bar] = []
    for idx, row in df.iterrows():
        bars.append(Bar(
            symbol=symbol,
            exchange=Exchange.NSE,
            interval="1d",
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]),
            ts=pd.Timestamp(idx, tz="UTC") if idx.tzinfo is None else pd.Timestamp(idx),
        ))

    # We need at least ~40 bars of history for indicators to stabilize
    min_warmup = 40
    if len(bars) < min_warmup + SEQ_LEN + FORECAST_HORIZON:
        return None

    # Compute features for each bar using expanding lookback (capped at 120 bars)
    feature_rows: List[List[float]] = []
    for i in range(min_warmup, len(bars)):
        lookback_start = max(0, i - 120)
        feats = fe.build_features(bars[lookback_start : i + 1])
        row = [feats.get(k, 0.0) for k in FEATURE_KEYS]
        feature_rows.append(row)

    return np.array(feature_rows, dtype=np.float32)


def build_labels(df: pd.DataFrame, warmup: int = 40) -> np.ndarray:
    """
    Build binary labels: 1 if price increases > 0.5% within next 5 bars.
    Aligned with feature computation (starts at warmup index).

    Returns array of shape (n_valid_bars,).
    """
    closes = df["Close"].values
    n = len(closes)
    labels = []

    for i in range(warmup, n):
        # Look ahead up to FORECAST_HORIZON bars
        future_end = min(i + FORECAST_HORIZON, n - 1)
        if future_end <= i:
            labels.append(np.nan)
            continue
        future_max = np.max(closes[i + 1 : future_end + 1])
        pct_change = (future_max - closes[i]) / (closes[i] + 1e-12)
        labels.append(1.0 if pct_change > THRESHOLD_PCT else 0.0)

    return np.array(labels, dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
def prepare_sequences(
    features: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (SEQ_LEN, n_features) sequences from feature matrix and
    corresponding labels.

    features: (T, F) array
    labels: (T,) array (must be same length as features)

    Returns:
        X: (N, SEQ_LEN, F) array
        y: (N,) array
    """
    assert len(features) == len(labels), (
        f"features and labels length mismatch: {len(features)} vs {len(labels)}"
    )
    T, F = features.shape
    if T < SEQ_LEN:
        return np.empty((0, SEQ_LEN, F), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X_list = []
    y_list = []
    for i in range(SEQ_LEN, T):
        if np.isnan(labels[i]):
            continue
        seq = features[i - SEQ_LEN : i]
        # Skip sequences with NaN/Inf
        if np.any(~np.isfinite(seq)):
            continue
        X_list.append(seq)
        y_list.append(labels[i])

    if not X_list:
        return np.empty((0, SEQ_LEN, F), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def aggregate_data(
    symbol_data: Dict[str, pd.DataFrame],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build feature sequences and labels from all symbols.
    Returns X, y, and list of successfully processed symbols.
    """
    from src.ai.models.lstm_predictor import NUM_FEATURES

    all_X = []
    all_y = []
    used_symbols = []

    for symbol, df in symbol_data.items():
        features = compute_features_from_df(df, symbol)
        if features is None:
            logger.warning("Skipping %s: could not compute features", symbol)
            continue

        labels = build_labels(df, warmup=40)

        # Align lengths (features and labels should both start at warmup)
        min_len = min(len(features), len(labels))
        features = features[:min_len]
        labels = labels[:min_len]

        X, y = prepare_sequences(features, labels)
        if len(X) == 0:
            logger.warning("Skipping %s: no valid sequences", symbol)
            continue

        all_X.append(X)
        all_y.append(y)
        used_symbols.append(symbol)
        logger.info("  %s: %d sequences (positive rate: %.1f%%)",
                     symbol, len(X), np.nanmean(y) * 100)

    if not all_X:
        logger.error("No valid training data produced from any symbol.")
        sys.exit(1)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    logger.info("Total dataset: %d sequences, %d features, positive rate: %.1f%%",
                len(X), X.shape[2], np.nanmean(y) * 100)

    return X, y, used_symbols


# ---------------------------------------------------------------------------
# Information Coefficient (IC) computation
# ---------------------------------------------------------------------------
def compute_ic(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Information Coefficient as Spearman rank correlation
    between predicted probabilities and actual labels.
    """
    from scipy.stats import spearmanr

    if len(predictions) < 5:
        return 0.0

    # Remove NaN values
    mask = np.isfinite(predictions) & np.isfinite(labels)
    predictions = predictions[mask]
    labels = labels[mask]

    if len(predictions) < 5:
        return 0.0

    # Spearman rank correlation
    corr, _ = spearmanr(predictions, labels)
    if np.isnan(corr):
        return 0.0
    return float(corr)


# ---------------------------------------------------------------------------
# Walk-forward training
# ---------------------------------------------------------------------------
def walk_forward_train(
    X: np.ndarray,
    y: np.ndarray,
    train_window: int = 60,
    test_window: int = 10,
    epochs_per_window: int = 15,
    batch_size: int = 64,
    learning_rate: float = 0.001,
) -> Tuple[dict, List[WindowMetrics]]:
    """
    Walk-forward validation training loop.

    1. For each window: train on [i : i+train_window], test on [i+train_window : i+train_window+test_window]
    2. Roll forward by test_window each step.
    3. Accumulate model across windows (warm start).
    4. Returns best model state_dict and per-window metrics.

    The walk-forward is done over the TIME dimension of the data. Since the
    sequences are already built chronologically, we split by sample index
    which preserves temporal ordering.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.ai.models.lstm_predictor import LSTMModel, NUM_FEATURES

    n_samples = len(X)
    n_features = X.shape[2]

    # Calculate number of walk-forward windows
    total_window = train_window + test_window
    n_windows = (n_samples - train_window) // test_window
    if n_windows < 1:
        logger.error(
            "Not enough data for walk-forward: %d samples, need at least %d "
            "(train=%d + test=%d)",
            n_samples, total_window, train_window, test_window,
        )
        sys.exit(1)

    logger.info(
        "Walk-forward: %d windows (train=%d, test=%d, total samples=%d)",
        n_windows, train_window, test_window, n_samples,
    )

    # Initialize model
    lstm = LSTMModel(input_size=n_features)
    if not lstm.available:
        logger.error("PyTorch LSTM model not available.")
        sys.exit(1)

    model = lstm._model
    device = lstm._device
    criterion = nn.BCELoss()

    # Global normalization stats (computed from first train window, updated rolling)
    global_means = None
    global_stds = None

    window_metrics: List[WindowMetrics] = []
    best_avg_ic = -np.inf
    best_state = None
    best_means = None
    best_stds = None

    for w in range(n_windows):
        train_start = w * test_window
        train_end = train_start + train_window
        test_end = train_end + test_window

        if test_end > n_samples:
            break

        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]

        # Skip windows with degenerate labels
        if len(np.unique(y_train)) < 2:
            logger.warning("Window %d: skipping (single-class training set)", w)
            continue

        # Compute normalization from training window
        flat_train = X_train.reshape(-1, n_features)
        means = flat_train.mean(axis=0)
        stds = flat_train.std(axis=0)
        stds[stds < 1e-8] = 1.0

        # Normalize
        X_train_norm = (X_train - means) / stds
        X_test_norm = (X_test - means) / stds

        # Clamp extreme values to prevent gradient explosion
        X_train_norm = np.clip(X_train_norm, -5.0, 5.0)
        X_test_norm = np.clip(X_test_norm, -5.0, 5.0)

        # DataLoaders
        train_ds = TensorDataset(
            torch.FloatTensor(X_train_norm), torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Optimizer (fresh per window with warm-started weights)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-5
        )

        # Train for epochs_per_window
        model.train()
        final_train_loss = 0.0
        for epoch in range(epochs_per_window):
            epoch_loss = 0.0
            n_batches = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            final_train_loss = epoch_loss / max(n_batches, 1)

        # Evaluate on test window
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test_norm).to(device)
            y_test_t = torch.FloatTensor(y_test).to(device)
            test_preds = model(X_test_t)
            test_loss = criterion(test_preds, y_test_t).item()
            test_preds_np = test_preds.cpu().numpy()

            # Accuracy
            test_predicted = (test_preds_np > 0.5).astype(float)
            test_accuracy = float(np.mean(test_predicted == y_test))

            # Train accuracy
            X_train_t = torch.FloatTensor(X_train_norm).to(device)
            train_preds = model(X_train_t).cpu().numpy()
            train_predicted = (train_preds > 0.5).astype(float)
            train_accuracy = float(np.mean(train_predicted == y_train))

        # Information Coefficient
        ic = compute_ic(test_preds_np, y_test)

        metrics = WindowMetrics(
            window_idx=w,
            train_start=f"sample_{train_start}",
            train_end=f"sample_{train_end}",
            test_start=f"sample_{train_end}",
            test_end=f"sample_{test_end}",
            train_loss=final_train_loss,
            test_loss=test_loss,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            ic=ic,
            n_train=len(X_train),
            n_test=len(X_test),
        )
        window_metrics.append(metrics)

        # Track best model by rolling average IC
        recent_ics = [m.ic for m in window_metrics[-10:]]
        avg_recent_ic = np.mean(recent_ics)
        if avg_recent_ic > best_avg_ic:
            best_avg_ic = avg_recent_ic
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_means = means.copy()
            best_stds = stds.copy()

        if (w + 1) % 5 == 0 or w == 0 or w == n_windows - 1:
            logger.info(
                "Window %3d/%d — train_loss=%.4f test_loss=%.4f "
                "train_acc=%.3f test_acc=%.3f IC=%.4f (avg_recent=%.4f)",
                w + 1, n_windows, final_train_loss, test_loss,
                train_accuracy, test_accuracy, ic, avg_recent_ic,
            )

    if best_state is None:
        logger.error("No valid walk-forward windows completed.")
        sys.exit(1)

    # Store normalization stats alongside model weights
    best_state["__normalization_means__"] = torch.FloatTensor(best_means)
    best_state["__normalization_stds__"] = torch.FloatTensor(best_stds)

    return best_state, window_metrics


# ---------------------------------------------------------------------------
# Model saving with metadata
# ---------------------------------------------------------------------------
def save_model(
    state_dict: dict,
    report: TrainingReport,
    model_dir: str = MODELS_DIR,
) -> str:
    """
    Save model weights, normalization stats, and version metadata.
    Returns path to saved model.
    """
    import torch

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "lstm_predictor.pt")
    stats_path = os.path.join(model_dir, "lstm_predictor_stats.npz")
    meta_path = os.path.join(model_dir, "lstm_predictor_meta.json")

    # Extract normalization stats from state_dict before saving model weights
    means = state_dict.pop("__normalization_means__", None)
    stds = state_dict.pop("__normalization_stds__", None)

    # Save model weights
    torch.save(state_dict, model_path)
    logger.info("Model weights saved to %s", model_path)

    # Save normalization stats
    if means is not None and stds is not None:
        np.savez(
            stats_path,
            means=means.numpy() if hasattr(means, "numpy") else means,
            stds=stds.numpy() if hasattr(stds, "numpy") else stds,
        )
        logger.info("Normalization stats saved to %s", stats_path)

    # Save metadata
    meta = {
        "model_id": "lstm_ts",
        "version": report.model_version,
        "timestamp": report.timestamp,
        "seq_len": report.seq_len,
        "forecast_horizon": report.forecast_horizon,
        "threshold_pct": report.threshold_pct,
        "avg_ic": report.avg_ic,
        "ic_std": report.ic_std,
        "stability_score": report.stability_score,
        "avg_test_accuracy": report.avg_test_accuracy,
        "total_windows": report.total_windows,
        "passed_ic_threshold": report.passed_ic_threshold,
        "symbols_used": report.symbols_used,
        "training_time_sec": report.total_training_time_sec,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)

    return model_path


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(
    window_metrics: List[WindowMetrics],
    symbols: List[str],
    training_time: float,
    model_version: str,
) -> TrainingReport:
    """Build aggregate training report from per-window metrics."""

    ic_values = [m.ic for m in window_metrics]
    avg_ic = float(np.mean(ic_values)) if ic_values else 0.0
    ic_std = float(np.std(ic_values)) if ic_values else 0.0
    stability = float(np.mean([1.0 if ic > 0 else 0.0 for ic in ic_values])) if ic_values else 0.0

    report = TrainingReport(
        model_version=model_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_windows=len(window_metrics),
        avg_train_loss=float(np.mean([m.train_loss for m in window_metrics])),
        avg_test_loss=float(np.mean([m.test_loss for m in window_metrics])),
        avg_train_accuracy=float(np.mean([m.train_accuracy for m in window_metrics])),
        avg_test_accuracy=float(np.mean([m.test_accuracy for m in window_metrics])),
        avg_ic=avg_ic,
        ic_std=ic_std,
        stability_score=stability,
        ic_values=ic_values,
        window_metrics=[asdict(m) for m in window_metrics],
        symbols_used=symbols,
        seq_len=SEQ_LEN,
        forecast_horizon=FORECAST_HORIZON,
        threshold_pct=THRESHOLD_PCT,
        passed_ic_threshold=avg_ic > 0.02,
        total_training_time_sec=training_time,
    )
    return report


def print_report(report: TrainingReport) -> None:
    """Pretty-print the training report to the logger."""
    logger.info("=" * 70)
    logger.info("LSTM TRAINING REPORT")
    logger.info("=" * 70)
    logger.info("Model version:       %s", report.model_version)
    logger.info("Timestamp:           %s", report.timestamp)
    logger.info("Symbols used:        %d", len(report.symbols_used))
    logger.info("Walk-forward windows: %d", report.total_windows)
    logger.info("-" * 70)
    logger.info("Avg Train Loss:      %.4f", report.avg_train_loss)
    logger.info("Avg Test Loss:       %.4f", report.avg_test_loss)
    logger.info("Avg Train Accuracy:  %.3f (%.1f%%)", report.avg_train_accuracy,
                report.avg_train_accuracy * 100)
    logger.info("Avg Test Accuracy:   %.3f (%.1f%%)", report.avg_test_accuracy,
                report.avg_test_accuracy * 100)
    logger.info("-" * 70)
    logger.info("Avg IC:              %.4f", report.avg_ic)
    logger.info("IC Std Dev:          %.4f", report.ic_std)
    logger.info("Stability Score:     %.3f (%.0f%% of windows IC > 0)",
                report.stability_score, report.stability_score * 100)
    logger.info("IC Threshold (0.02): %s",
                "PASSED" if report.passed_ic_threshold else "FAILED")
    logger.info("-" * 70)
    logger.info("Training Time:       %.1f seconds", report.total_training_time_sec)
    logger.info("=" * 70)

    if not report.passed_ic_threshold:
        logger.warning(
            "Average IC (%.4f) is below the 0.02 threshold. "
            "Model is saved but may not have predictive power. "
            "Consider: more data, feature engineering, or hyperparameter tuning.",
            report.avg_ic,
        )


# ---------------------------------------------------------------------------
# Main training orchestrator
# ---------------------------------------------------------------------------
def train(
    symbols: Optional[List[str]] = None,
    quick: bool = False,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.001,
    period: str = "2y",
    train_window: int = 60,
    test_window: int = 10,
) -> None:
    """
    End-to-end training pipeline:
      1. Fetch data from Yahoo Finance
      2. Compute features using FeatureEngine
      3. Build sequences and labels
      4. Walk-forward training
      5. Evaluate and report
      6. Save model
    """
    try:
        import torch
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    try:
        from scipy.stats import spearmanr  # noqa: F401
    except ImportError:
        logger.error("scipy not installed. Run: pip install scipy")
        sys.exit(1)

    start_time = time.time()

    # Quick mode overrides
    if quick:
        if symbols is None:
            symbols = QUICK_TICKERS
        period = "1y"
        epochs = 8
        train_window = 40
        test_window = 5
        logger.info("QUICK MODE: using smaller subset (%d symbols), "
                     "shorter period (%s), fewer epochs (%d), "
                     "smaller windows (train=%d, test=%d)",
                     len(symbols), period, epochs, train_window, test_window)
    else:
        if symbols is None:
            symbols = NIFTY50_TICKERS

    # Step 1: Fetch data
    logger.info("=" * 70)
    logger.info("STEP 1: Fetching historical data")
    logger.info("=" * 70)
    symbol_data = fetch_yahoo_data(symbols, period=period)

    # Step 2: Compute features and build sequences
    logger.info("=" * 70)
    logger.info("STEP 2: Computing features and building sequences")
    logger.info("=" * 70)
    X, y, used_symbols = aggregate_data(symbol_data)

    if len(X) < train_window + test_window:
        logger.error(
            "Insufficient sequences (%d) for walk-forward with "
            "train_window=%d + test_window=%d. Need more data or symbols.",
            len(X), train_window, test_window,
        )
        sys.exit(1)

    # Shuffle while preserving temporal structure per-symbol
    # (We concatenated symbols, so the overall ordering has jumps anyway.
    #  For walk-forward to be truly temporal, you'd need per-symbol windows.
    #  Here we do a global walk-forward over the pooled data, which is a
    #  reasonable approximation when all symbols share similar date ranges.)
    logger.info("Dataset ready: %d sequences, shape=%s", len(X), X.shape)

    # Step 3: Walk-forward training
    logger.info("=" * 70)
    logger.info("STEP 3: Walk-forward training")
    logger.info("=" * 70)
    best_state, window_metrics = walk_forward_train(
        X, y,
        train_window=train_window,
        test_window=test_window,
        epochs_per_window=epochs,
        batch_size=batch_size,
        learning_rate=lr,
    )

    training_time = time.time() - start_time

    # Step 4: Generate report
    logger.info("=" * 70)
    logger.info("STEP 4: Results and saving")
    logger.info("=" * 70)
    model_version = f"v2.wf.{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    report = generate_report(window_metrics, used_symbols, training_time, model_version)
    print_report(report)

    # Step 5: Save model
    model_path = save_model(best_state, report)

    # Save full report as JSON
    report_path = os.path.join(MODELS_DIR, "lstm_training_report.json")
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    logger.info("Full training report saved to %s", report_path)

    logger.info("Training complete. Model at: %s", model_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LSTM predictor with walk-forward validation on NIFTY50 stocks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training on all NIFTY50 stocks (2 years of data):
  PYTHONPATH=. python scripts/train_lstm.py

  # Quick retraining (smaller windows, fewer symbols):
  PYTHONPATH=. python scripts/train_lstm.py --quick

  # Custom symbols and parameters:
  PYTHONPATH=. python scripts/train_lstm.py --symbols RELIANCE.NS,INFY.NS,TCS.NS --epochs 20

  # Override walk-forward windows:
  PYTHONPATH=. python scripts/train_lstm.py --train-window 90 --test-window 15
        """,
    )

    parser.add_argument(
        "--quick", action="store_true",
        help="Quick retraining mode: fewer symbols, smaller windows, fewer epochs.",
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated list of Yahoo Finance tickers (e.g. RELIANCE.NS,TCS.NS). "
             "Defaults to NIFTY50.",
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Epochs per walk-forward window (default: 15).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Training batch size (default: 64).",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--period", type=str, default="2y",
        help="Yahoo Finance data period: 6mo, 1y, 2y, 5y, max (default: 2y).",
    )
    parser.add_argument(
        "--train-window", type=int, default=60,
        help="Walk-forward training window size in sequences (default: 60).",
    )
    parser.add_argument(
        "--test-window", type=int, default=10,
        help="Walk-forward test window size in sequences (default: 10).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    train(
        symbols=symbols,
        quick=args.quick,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        period=args.period,
        train_window=args.train_window,
        test_window=args.test_window,
    )


if __name__ == "__main__":
    main()
