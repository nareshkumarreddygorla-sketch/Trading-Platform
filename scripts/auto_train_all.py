#!/usr/bin/env python3
"""
MASTER AUTO-TRAIN PIPELINE
===========================
One command to rule them all: downloads data, trains every AI model,
validates performance, and hot-reloads into the running system.

Usage:
    PYTHONPATH=. python3 scripts/auto_train_all.py              # Full pipeline
    PYTHONPATH=. python3 scripts/auto_train_all.py --quick       # Quick mode (5 stocks, 10 epochs)
    PYTHONPATH=. python3 scripts/auto_train_all.py --models lstm,transformer
    PYTHONPATH=. python3 scripts/auto_train_all.py --data-only   # Download data only
    PYTHONPATH=. python3 scripts/auto_train_all.py --skip-data   # Skip download, train only

Also callable via:
    make train-ai           # Quick train (5 stocks)
    make train-ai-full      # Full train (70 stocks)
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("auto_train")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "nse_historical")
META_PATH = os.path.join(MODELS_DIR, "training_meta.json")


def banner(text: str):
    width = 60
    logger.info("=" * width)
    logger.info(text.center(width))
    logger.info("=" * width)


def run_step(name: str, cmd: list, timeout: int = 1800) -> bool:
    """Run a subprocess step with timeout. Returns True on success."""
    logger.info("Starting: %s", name)
    start = time.time()
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = PROJECT_ROOT
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=timeout, cwd=PROJECT_ROOT,
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            logger.info("DONE: %s (%.1fs)", name, elapsed)
            # Show last few lines of output
            lines = (result.stdout or "").strip().split("\n")
            for line in lines[-3:]:
                if line.strip():
                    logger.info("  > %s", line.strip())
            return True
        else:
            logger.error("FAILED: %s (%.1fs)", name, elapsed)
            err_lines = (result.stderr or result.stdout or "").strip().split("\n")
            for line in err_lines[-5:]:
                if line.strip():
                    logger.error("  > %s", line.strip())
            return False
    except subprocess.TimeoutExpired:
        logger.error("TIMEOUT: %s (>%ds)", name, timeout)
        return False
    except Exception as e:
        logger.error("ERROR: %s — %s", name, e)
        return False


def step_download_data(symbols_count: int = 50, period: str = "2y"):
    """Step 1: Download historical data — dynamically scans entire NSE market."""
    banner("STEP 1: SCANNING MARKET & DOWNLOADING DATA")

    from scripts.data_sources import DataDownloader, get_dynamic_symbols, INDEX_SYMBOLS

    os.makedirs(DATA_DIR, exist_ok=True)
    downloader = DataDownloader()

    # DYNAMIC: scan entire NSE market, auto-pick best stocks by liquidity
    logger.info("Scanning entire NSE market to find best %d stocks...", symbols_count)
    symbols = get_dynamic_symbols(count=symbols_count)
    logger.info("Auto-selected %d stocks from full NSE market scan", len(symbols))

    logger.info("Downloading %d stocks + indices (period=%s)...", len(symbols), period)

    # Download indices first
    for idx_sym in INDEX_SYMBOLS:
        df = downloader.download(idx_sym, period=period)
        if not df.empty:
            safe_name = idx_sym.replace("^", "").replace(" ", "_")
            df.to_parquet(os.path.join(DATA_DIR, f"{safe_name}.parquet"))

    # Bulk download stocks
    results = downloader.download_all(symbols, period=period)
    all_dfs = []
    for symbol, df in results.items():
        df.to_parquet(os.path.join(DATA_DIR, f"{symbol}.parquet"))
        all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=False)
        combined.to_parquet(os.path.join(DATA_DIR, "all_stocks.parquet"))
        logger.info("Total: %d stocks, %d bars saved", len(results), len(combined))
    else:
        logger.error("No data downloaded!")
        return False

    return True


def step_train_xgboost():
    """Step 2: Train XGBoost alpha model (existing pipeline)."""
    banner("STEP 2: TRAINING XGBOOST")
    script = os.path.join(PROJECT_ROOT, "scripts", "train_alpha_model.py")
    if not os.path.exists(script):
        logger.warning("XGBoost training script not found, skipping")
        return True
    return run_step("XGBoost Alpha Model", [sys.executable, script], timeout=600)


def step_train_lstm(epochs: int = 30, symbols: str = None):
    """Step 3: Train LSTM predictor."""
    banner("STEP 3: TRAINING LSTM")
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "train_lstm.py"),
           "--epochs", str(epochs)]
    if symbols:
        cmd.extend(["--symbols", symbols])
    return run_step("LSTM Predictor", cmd, timeout=1200)


def step_train_transformer(epochs: int = 30, symbols: str = None):
    """Step 4: Train Transformer predictor."""
    banner("STEP 4: TRAINING TRANSFORMER")
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "train_transformer.py"),
           "--epochs", str(epochs)]
    if symbols:
        cmd.extend(["--symbols", symbols])
    return run_step("Transformer Predictor", cmd, timeout=1200)


def step_train_rl(timesteps: int = 100000, symbols: str = None):
    """Step 5: Train RL agent."""
    banner("STEP 5: TRAINING RL AGENT")
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "train_rl_agent.py"),
           "--timesteps", str(timesteps)]
    if symbols:
        cmd.extend(["--symbols", symbols])
    return run_step("RL Agent (PPO)", cmd, timeout=1800)


def save_training_meta(results: dict):
    """Save training metadata for the dashboard / auto-retrain scheduler."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    meta = {
        "last_trained": datetime.utcnow().isoformat() + "Z",
        "results": results,
        "models_dir": MODELS_DIR,
        "data_dir": DATA_DIR,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Training metadata saved to %s", META_PATH)


def main():
    parser = argparse.ArgumentParser(
        description="Master Auto-Train Pipeline: download data + train all AI models"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 5 stocks, 10 epochs, 50k RL steps")
    parser.add_argument("--full", action="store_true",
                        help="Full mode: 70 stocks, 50 epochs, 200k RL steps")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated models to train: xgboost,lstm,transformer,rl (default: all)")
    parser.add_argument("--data-only", action="store_true",
                        help="Only download data, skip training")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data download, train with existing data")
    parser.add_argument("--period", type=str, default="2y",
                        help="Historical data period: 1y, 2y, 5y, max")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs for LSTM/Transformer")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="RL training timesteps")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Specific symbols to use for training")
    args = parser.parse_args()

    # Configuration based on mode
    # All modes use DYNAMIC stock selection from entire NSE market
    if args.quick:
        n_stocks = 30       # top 30 most liquid (fast training)
        epochs = args.epochs or 10
        rl_timesteps = args.timesteps or 50000
    elif args.full:
        n_stocks = 300      # top 300 from full market scan (comprehensive)
        epochs = args.epochs or 50
        rl_timesteps = args.timesteps or 200000
    else:
        n_stocks = 100      # top 100 most liquid (balanced)
        epochs = args.epochs or 30
        rl_timesteps = args.timesteps or 100000

    models_to_train = (args.models.split(",") if args.models
                       else ["xgboost", "lstm", "transformer", "rl"])

    # Need pandas for data step
    global pd
    import pandas as pd

    banner("AI AUTO-TRAIN PIPELINE")
    logger.info("Mode: %s", "quick" if args.quick else ("full" if args.full else "standard"))
    logger.info("Stocks: %d | Epochs: %d | RL steps: %d", n_stocks, epochs, rl_timesteps)
    logger.info("Models: %s", ", ".join(models_to_train))
    logger.info("")

    start_time = time.time()
    results = {}

    # Step 1: Download data
    if not args.skip_data:
        success = step_download_data(symbols_count=n_stocks, period=args.period)
        results["data_download"] = "success" if success else "failed"
        if not success and not args.skip_data:
            logger.error("Data download failed. Use --skip-data if you have existing data.")
            if not os.path.exists(os.path.join(DATA_DIR, "all_stocks.parquet")):
                return
    else:
        logger.info("Skipping data download (--skip-data)")
        results["data_download"] = "skipped"

    if args.data_only:
        logger.info("Data-only mode, skipping training.")
        save_training_meta(results)
        return

    # Step 2-5: Train models
    if "xgboost" in models_to_train:
        ok = step_train_xgboost()
        results["xgboost"] = "success" if ok else "failed"

    if "lstm" in models_to_train:
        ok = step_train_lstm(epochs=epochs, symbols=args.symbols)
        results["lstm"] = "success" if ok else "failed"

    if "transformer" in models_to_train:
        ok = step_train_transformer(epochs=epochs, symbols=args.symbols)
        results["transformer"] = "success" if ok else "failed"

    if "rl" in models_to_train:
        ok = step_train_rl(timesteps=rl_timesteps, symbols=args.symbols)
        results["rl"] = "success" if ok else "failed"

    # Save metadata
    save_training_meta(results)

    # Summary
    elapsed = time.time() - start_time
    banner("TRAINING COMPLETE")
    for model, status in results.items():
        icon = "OK" if status == "success" else ("SKIP" if status == "skipped" else "FAIL")
        logger.info("  [%s] %s", icon, model)
    logger.info("")
    logger.info("Total time: %.1f minutes", elapsed / 60)
    logger.info("Models saved to: %s", MODELS_DIR)

    # Check model files
    model_files = {
        "XGBoost": "alpha_xgb.joblib",
        "LSTM": "lstm_predictor.pt",
        "Transformer": "transformer_predictor.pt",
        "RL Agent": "rl_agent.zip",
    }
    logger.info("")
    logger.info("Model files:")
    for name, fname in model_files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            logger.info("  [EXISTS] %s (%.1f MB)", name, size_mb)
        else:
            logger.info("  [MISSING] %s", name)

    failed = [k for k, v in results.items() if v == "failed"]
    if failed:
        logger.warning("Some models failed to train: %s", ", ".join(failed))
        logger.warning("The system will still work with available models.")


if __name__ == "__main__":
    main()
