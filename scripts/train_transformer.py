"""
Train Transformer model for price direction prediction using walk-forward validation.

Walk-forward scheme (ported from LSTM training pipeline):
    - 60-day rolling train window, 10-day test window
    - Rolling forward across all available data
    - Computes Information Coefficient (IC) per window
    - Cosine annealing LR schedule with warm restarts per window
    - Early stopping per window (patience=5, restores best weights)
    - Validation gate: rejects model if avg IC < 0.03 or avg accuracy < 55%
    - Extended universe: 70 symbols (NIFTY50 + NIFTY Next 50 midcaps)

Target: P(price increase > 0.5% within next 5 bars)

Usage:
    PYTHONPATH=. python scripts/train_transformer.py
    PYTHONPATH=. python scripts/train_transformer.py --quick
    PYTHONPATH=. python scripts/train_transformer.py --epochs 30
    PYTHONPATH=. python scripts/train_transformer.py --symbols RELIANCE.NS,TCS.NS
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_transformer")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Quick mode tickers (same as LSTM training)
QUICK_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "ITC.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
    "LT.NS",
]


# ---------------------------------------------------------------------------
# Metrics dataclass (mirrors LSTM training report structure)
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
    ic: float
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
    stability_score: float
    ic_values: List[float]
    window_metrics: List[dict]
    symbols_used: List[str]
    passed_validation_gate: bool
    total_training_time_sec: float


# ---------------------------------------------------------------------------
# Walk-forward training with cosine annealing
# ---------------------------------------------------------------------------
def walk_forward_train(
    X: np.ndarray,
    y: np.ndarray,
    train_window: int = 60,
    test_window: int = 10,
    epochs_per_window: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.0005,
    early_stopping_patience: int = 5,
) -> Tuple[Optional[dict], List[WindowMetrics], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Walk-forward validation training loop for the Transformer model.

    Features:
        - Cosine annealing LR schedule with warm restarts per window
        - Early stopping per window with best-weight restoration
        - Gradient clipping at max_norm=1.0
        - Returns best model state_dict, per-window metrics, and normalization stats
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.ai.models.transformer_predictor import TransformerModel

    n_samples = len(X)
    n_features = X.shape[2]

    # Calculate number of walk-forward windows
    n_windows = (n_samples - train_window) // test_window
    if n_windows < 1:
        logger.error(
            "Not enough data for walk-forward: %d samples, need at least %d (train=%d + test=%d)",
            n_samples,
            train_window + test_window,
            train_window,
            test_window,
        )
        sys.exit(1)

    logger.info(
        "Walk-forward: %d windows (train=%d, test=%d, total samples=%d, "
        "epochs_per_window=%d, early_stopping_patience=%d)",
        n_windows,
        train_window,
        test_window,
        n_samples,
        epochs_per_window,
        early_stopping_patience,
    )

    # Initialize model
    tf_model = TransformerModel(input_size=n_features)
    if not tf_model.available:
        logger.error("Transformer model not available.")
        sys.exit(1)

    model = tf_model._model
    device = tf_model._device
    criterion = nn.BCELoss()

    # Import IC computation from LSTM training
    from scripts.train_lstm import compute_ic

    window_metrics: List[WindowMetrics] = []
    best_avg_ic = -np.inf
    best_state = None
    best_means = None
    best_stds = None

    for w in range(n_windows):
        train_start = w * test_window
        train_end = train_start + train_window
        test_end_idx = train_end + test_window

        if test_end_idx > n_samples:
            break

        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_test = X[train_end:test_end_idx]
        y_test = y[train_end:test_end_idx]

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
        X_train_norm = np.clip((X_train - means) / stds, -5.0, 5.0)
        X_test_norm = np.clip((X_test - means) / stds, -5.0, 5.0)

        # DataLoaders
        train_ds = TensorDataset(torch.FloatTensor(X_train_norm), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Optimizer (fresh per window with warm-started weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Cosine annealing LR schedule with warm restarts per window
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, epochs_per_window // 3),
            T_mult=2,
            eta_min=1e-6,
        )

        # Early stopping state for this window
        best_window_loss = float("inf")
        best_window_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0

        # Train for epochs_per_window (with early stopping)
        final_train_loss = 0.0
        for epoch in range(epochs_per_window):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step(epoch)
            final_train_loss = epoch_loss / max(n_batches, 1)

            # Evaluate for early stopping
            model.eval()
            with torch.no_grad():
                X_test_t = torch.FloatTensor(X_test_norm).to(device)
                y_test_t = torch.FloatTensor(y_test).to(device)
                val_preds = model(X_test_t)
                val_loss = criterion(val_preds, y_test_t).item()

            if val_loss < best_window_loss:
                best_window_loss = val_loss
                best_window_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.debug(
                        "Window %d: early stopping at epoch %d/%d (val_loss=%.4f, best=%.4f)",
                        w,
                        epoch + 1,
                        epochs_per_window,
                        val_loss,
                        best_window_loss,
                    )
                    break

        # Restore best weights from this window
        model.load_state_dict(best_window_state)
        model.to(device)

        # Final evaluation on test window
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test_norm).to(device)
            y_test_t = torch.FloatTensor(y_test).to(device)
            test_preds = model(X_test_t)
            test_loss = criterion(test_preds, y_test_t).item()
            test_preds_np = test_preds.cpu().numpy()

            test_predicted = (test_preds_np > 0.5).astype(float)
            test_accuracy = float(np.mean(test_predicted == y_test))

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
            test_end=f"sample_{test_end_idx}",
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
                "Window %3d/%d — train_loss=%.4f test_loss=%.4f train_acc=%.3f test_acc=%.3f IC=%.4f (avg_recent=%.4f)",
                w + 1,
                n_windows,
                final_train_loss,
                test_loss,
                train_accuracy,
                test_accuracy,
                ic,
                avg_recent_ic,
            )

    return best_state, window_metrics, best_means, best_stds


# ---------------------------------------------------------------------------
# Main training orchestrator
# ---------------------------------------------------------------------------
def train(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 0.0005,
    symbols: Optional[List[str]] = None,
    quick: bool = False,
    train_window: int = 60,
    test_window: int = 10,
) -> None:
    """
    End-to-end Transformer training pipeline with walk-forward validation:
      1. Fetch data from Yahoo Finance (extended universe)
      2. Compute features using FeatureEngine
      3. Build sequences and labels
      4. Walk-forward training with cosine annealing
      5. Validation gate (reject if IC < 0.03 or accuracy < 55%)
      6. Save model if gate passes
    """
    try:
        import torch
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install torch")
        return

    from scripts.train_lstm import (
        EXTENDED_TICKERS,
        aggregate_data,
        fetch_yahoo_data,
    )

    start_time = time.time()

    # Quick mode overrides
    if quick:
        if symbols is None:
            symbols = QUICK_TICKERS
        period = "1y"
        epochs = min(epochs, 15)
        train_window = 40
        test_window = 5
        logger.info(
            "QUICK MODE: %d symbols, period=%s, epochs=%d, train_window=%d, test_window=%d",
            len(symbols),
            period,
            epochs,
            train_window,
            test_window,
        )
    else:
        period = "2y"
        if symbols is None:
            symbols = EXTENDED_TICKERS  # 70 symbols (NIFTY50 + midcap diversification)

    # Step 1: Fetch data
    logger.info("=" * 70)
    logger.info("STEP 1: Fetching historical data for %d symbols", len(symbols))
    logger.info("=" * 70)
    symbol_data = fetch_yahoo_data(symbols, period=period)

    # Step 2: Compute features and build sequences
    logger.info("=" * 70)
    logger.info("STEP 2: Computing features and building sequences")
    logger.info("=" * 70)
    X, y, used_symbols = aggregate_data(symbol_data)

    if len(X) < train_window + test_window:
        logger.error(
            "Insufficient sequences (%d) for walk-forward with train_window=%d + test_window=%d.",
            len(X),
            train_window,
            test_window,
        )
        return

    logger.info("Dataset ready: %d sequences, shape=%s", len(X), X.shape)

    # Step 3: Walk-forward training
    logger.info("=" * 70)
    logger.info("STEP 3: Walk-forward training (Transformer)")
    logger.info("=" * 70)
    best_state, window_metrics, best_means, best_stds = walk_forward_train(
        X,
        y,
        train_window=train_window,
        test_window=test_window,
        epochs_per_window=epochs,
        batch_size=batch_size,
        learning_rate=lr,
    )

    training_time = time.time() - start_time

    if best_state is None or not window_metrics:
        logger.error("No valid walk-forward windows completed.")
        return

    # Step 4: Generate report
    logger.info("=" * 70)
    logger.info("STEP 4: Results and validation")
    logger.info("=" * 70)

    ic_values = [m.ic for m in window_metrics]
    avg_ic = float(np.mean(ic_values)) if ic_values else 0.0
    ic_std = float(np.std(ic_values)) if ic_values else 0.0
    stability = float(np.mean([1.0 if ic > 0 else 0.0 for ic in ic_values])) if ic_values else 0.0
    avg_test_acc = float(np.mean([m.test_accuracy for m in window_metrics]))
    avg_train_acc = float(np.mean([m.train_accuracy for m in window_metrics]))
    avg_train_loss = float(np.mean([m.train_loss for m in window_metrics]))
    avg_test_loss = float(np.mean([m.test_loss for m in window_metrics]))

    model_version = f"v2.wf.{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # Validation gate
    IC_GATE_THRESHOLD = 0.03
    ACCURACY_GATE_THRESHOLD = 0.55
    gate_passed = True
    gate_reasons = []

    if avg_ic < IC_GATE_THRESHOLD:
        gate_reasons.append(f"avg IC {avg_ic:.4f} < {IC_GATE_THRESHOLD}")
        gate_passed = False
    if avg_test_acc < ACCURACY_GATE_THRESHOLD:
        gate_reasons.append(f"avg test accuracy {avg_test_acc:.3f} < {ACCURACY_GATE_THRESHOLD}")
        gate_passed = False

    report = TrainingReport(
        model_version=model_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_windows=len(window_metrics),
        avg_train_loss=avg_train_loss,
        avg_test_loss=avg_test_loss,
        avg_train_accuracy=avg_train_acc,
        avg_test_accuracy=avg_test_acc,
        avg_ic=avg_ic,
        ic_std=ic_std,
        stability_score=stability,
        ic_values=ic_values,
        window_metrics=[asdict(m) for m in window_metrics],
        symbols_used=used_symbols,
        passed_validation_gate=gate_passed,
        total_training_time_sec=training_time,
    )

    # Print report
    logger.info("=" * 70)
    logger.info("TRANSFORMER TRAINING REPORT")
    logger.info("=" * 70)
    logger.info("Model version:       %s", report.model_version)
    logger.info("Symbols used:        %d", len(report.symbols_used))
    logger.info("Walk-forward windows: %d", report.total_windows)
    logger.info("-" * 70)
    logger.info("Avg Train Loss:      %.4f", report.avg_train_loss)
    logger.info("Avg Test Loss:       %.4f", report.avg_test_loss)
    logger.info("Avg Train Accuracy:  %.3f (%.1f%%)", report.avg_train_accuracy, report.avg_train_accuracy * 100)
    logger.info("Avg Test Accuracy:   %.3f (%.1f%%)", report.avg_test_accuracy, report.avg_test_accuracy * 100)
    logger.info("-" * 70)
    logger.info("Avg IC:              %.4f", report.avg_ic)
    logger.info("IC Std Dev:          %.4f", report.ic_std)
    logger.info(
        "Stability Score:     %.3f (%.0f%% of windows IC > 0)", report.stability_score, report.stability_score * 100
    )
    logger.info("-" * 70)
    logger.info("Training Time:       %.1f seconds", report.total_training_time_sec)
    logger.info("=" * 70)

    os.makedirs(MODELS_DIR, exist_ok=True)

    if not gate_passed:
        logger.warning(
            "MODEL VALIDATION GATE FAILED — model will NOT be saved. Reasons: %s",
            "; ".join(gate_reasons),
        )
        report_path = os.path.join(MODELS_DIR, "transformer_training_report_REJECTED.json")
        with open(report_path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        logger.info("Rejected training report saved to %s", report_path)
        return

    # Step 5: Save model (passed validation gate)
    model_path = os.path.join(MODELS_DIR, "transformer_predictor.pt")
    stats_path = os.path.join(MODELS_DIR, "transformer_predictor_stats.npz")
    meta_path = os.path.join(MODELS_DIR, "transformer_predictor_meta.json")

    torch.save(best_state, model_path)
    logger.info("Model weights saved to %s", model_path)

    if best_means is not None and best_stds is not None:
        np.savez(stats_path, means=best_means, stds=best_stds)
        logger.info("Normalization stats saved to %s", stats_path)

    meta = {
        "model_id": "transformer_ts",
        "version": model_version,
        "timestamp": report.timestamp,
        "avg_ic": avg_ic,
        "ic_std": ic_std,
        "stability_score": stability,
        "avg_test_accuracy": avg_test_acc,
        "total_windows": len(window_metrics),
        "passed_validation_gate": gate_passed,
        "symbols_used": used_symbols,
        "training_time_sec": training_time,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)

    # Save full report
    report_path = os.path.join(MODELS_DIR, "transformer_training_report.json")
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    logger.info("Full training report saved to %s", report_path)

    # ── P0-2: Save OOS validation data for model validation gate ──
    try:
        validation_dir = os.path.join(MODELS_DIR, "validation")
        os.makedirs(validation_dir, exist_ok=True)
        if window_metrics:
            all_ics = np.array([m.ic for m in window_metrics])
            all_accs = np.array([m.test_accuracy for m in window_metrics])
            np.savez(
                os.path.join(validation_dir, "transformer_ts_oos.npz"),
                predictions=all_ics,
                actuals=all_accs,
            )
            logger.info("Transformer OOS validation data saved (%d windows)", len(window_metrics))
    except Exception as e:
        logger.warning("Transformer OOS save failed: %s", e)

    logger.info("Training complete. Model at: %s", model_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train Transformer predictor with walk-forward validation.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per walk-forward window (default: 30).")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size (default: 64).")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate (default: 0.0005).")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated list of Yahoo Finance tickers.")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer symbols, smaller windows.")
    parser.add_argument("--train-window", type=int, default=60, help="Walk-forward training window size (default: 60).")
    parser.add_argument("--test-window", type=int, default=10, help="Walk-forward test window size (default: 10).")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        symbols=symbols,
        quick=args.quick,
        train_window=args.train_window,
        test_window=args.test_window,
    )


if __name__ == "__main__":
    main()
