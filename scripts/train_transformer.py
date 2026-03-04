"""
Train Transformer model for price direction prediction.
Shares data pipeline with LSTM training.

Usage:
    PYTHONPATH=. python scripts/train_transformer.py
    PYTHONPATH=. python scripts/train_transformer.py --epochs 30
"""
import argparse
import logging
import os
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def train(epochs=30, batch_size=64, lr=0.0005, symbols=None):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install torch")
        return

    from scripts.train_lstm import aggregate_data
    from src.ai.models.transformer_predictor import TransformerModel
    from src.ai.models.lstm_predictor import NUM_FEATURES

    logger.info("Loading data...")
    X, y, _used_symbols = aggregate_data(symbols=symbols)
    if len(X) == 0:
        logger.error("No training data. Run: PYTHONPATH=. python scripts/download_nse_data.py")
        return

    logger.info("Dataset: %d sequences", len(X))

    # Split
    n = len(X)
    train_end = int(n * 0.75)
    val_end = int(n * 0.875)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # Normalize
    means = X_train.reshape(-1, NUM_FEATURES).mean(axis=0)
    stds = X_train.reshape(-1, NUM_FEATURES).std(axis=0)
    stds[stds < 1e-8] = 1.0

    X_train = (X_train - means) / stds
    X_val = (X_val - means) / stds
    X_test = (X_test - means) / stds

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Build model
    tf_model = TransformerModel(input_size=NUM_FEATURES)
    if not tf_model.available:
        logger.error("Transformer model not available")
        return

    model = tf_model._model
    device = tf_model._device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                val_loss += criterion(pred, batch_y).item()
                val_correct += ((pred > 0.5).float() == batch_y).sum().item()
                val_total += len(batch_y)

        val_acc = val_correct / val_total if val_total > 0 else 0
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("Epoch %d/%d — val_acc=%.3f (best=%.3f)", epoch + 1, epochs, val_acc, best_val_acc)

    # Test
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(torch.FloatTensor(X_test).to(device))
        test_acc = ((test_pred > 0.5).float() == torch.FloatTensor(y_test).to(device)).float().mean().item()

    logger.info("Test accuracy: %.3f", test_acc)

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "transformer_predictor.pt")
    stats_path = os.path.join(MODELS_DIR, "transformer_predictor_stats.npz")

    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), model_path)
    np.savez(stats_path, means=means, stds=stds)
    logger.info("Model saved to %s", model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--symbols", type=str, default=None)
    args = parser.parse_args()
    symbols = args.symbols.split(",") if args.symbols else None
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, symbols=symbols)


if __name__ == "__main__":
    main()
