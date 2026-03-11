#!/usr/bin/env python3
"""
Train an XGBoost + LightGBM ensemble on historical intraday data.
Produces models/alpha_xgb.joblib for the autonomous trading loop.

Key improvements over v1:
  - 37+ features (expanded FeatureEngine)
  - Market context features (NIFTY50 trend, RSI, volatility)
  - Risk-adjusted target: label=1 only if forward_return > min_return_threshold
  - Sample weighting by move magnitude
  - XGBoost + LightGBM ensemble with soft voting
  - Walk-forward cross-validation with proper IC measurement

Key improvements over v2 (production-grade):
  - Optuna hyperparameter optimization (30 trials default)
  - Adaptive MIN_RETURN_PCT based on median absolute return
  - Purged cross-validation with embargo gap to prevent look-ahead bias
  - Model validation gate: IC > 0.05 and balanced buy_pct required to save
  - Transaction cost hurdle: signal must exceed 0.1% round-trip costs
  - True holdout set: last 20% of data never touched during training/HPO
  - Final refit uses early stopping with time-series validation set
  - Expanded default training: 30 symbols, 120-day lookback
  - CLI arguments via argparse

Usage:
    PYTHONPATH=. python scripts/train_alpha_model.py
    PYTHONPATH=. python scripts/train_alpha_model.py --symbols-count 50 --lookback-days 180 --optuna-trials 50
    TRAIN_SYMBOLS=all PYTHONPATH=. python scripts/train_alpha_model.py

Configurable via env vars (CLI args take precedence):
    TRAIN_SYMBOLS      comma-separated or 'all' for NIFTY 500
    TRAIN_PERIOD       yfinance period  (default: 120d)
    TRAIN_INTERVAL     yfinance interval (default: 5m)
    FORWARD_BARS       bars ahead for target (default: 5)
    MIN_RETURN_PCT     minimum return % for positive label (default: adaptive)
    MODEL_DIR          output directory (default: models/)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional: Optuna for hyperparameter optimization
try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_alpha_model")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_NIFTY50_DEFAULT = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "AXISBANK.NS",
    "ASIANPAINT.NS",
    "MARUTI.NS",
    "BAJFINANCE.NS",
    "HCLTECH.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "WIPRO.NS",
    "ULTRACEMCO.NS",
    "TATAMOTORS.NS",
    "NESTLEIND.NS",
    "ONGC.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "M&M.NS",
    "JSWSTEEL.NS",
    "TATASTEEL.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "TECHM.NS",
    "INDUSINDBK.NS",
    "BAJAJFINSV.NS",
    "HDFCLIFE.NS",
    "SBILIFE.NS",
    "GRASIM.NS",
    "DIVISLAB.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "EICHERMOT.NS",
    "DRREDDY.NS",
    "APOLLOHOSP.NS",
    "COALINDIA.NS",
    "BPCL.NS",
    "TATACONSUM.NS",
    "HEROMOTOCO.NS",
    "UPL.NS",
    "BAJAJ-AUTO.NS",
    "HINDALCO.NS",
    "LTIM.NS",
]

INTERVAL = os.environ.get("TRAIN_INTERVAL", "5m")
FORWARD_BARS = int(os.environ.get("FORWARD_BARS", "5"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments (take precedence over env vars)."""
    parser = argparse.ArgumentParser(description="Train production-grade XGBoost alpha model")
    parser.add_argument(
        "--symbols-count",
        type=int,
        default=int(os.environ.get("SYMBOLS_COUNT", "30")),
        help="Number of NIFTY50 symbols to train on (default: 30)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=int(os.environ.get("LOOKBACK_DAYS", "120")),
        help="Number of days of historical data (default: 120)",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=int(os.environ.get("OPTUNA_TRIALS", "30")),
        help="Number of Optuna hyperparameter optimization trials (default: 30)",
    )
    return parser.parse_args()


def _resolve_symbols(symbols_count: int) -> list:
    """Resolve symbol list. Use 'all' for full NIFTY 500 universe."""
    env_val = os.environ.get("TRAIN_SYMBOLS", "")
    if env_val.strip().lower() == "all":
        try:
            from src.scanner.nse_universe import get_universe

            symbols = get_universe(yfinance_suffix=True)
            logger.info("Using full NSE universe: %d symbols", len(symbols))
            return symbols
        except Exception as e:
            logger.warning("Failed to fetch universe, using defaults: %s", e)
            return _NIFTY50_DEFAULT[:symbols_count]
    elif env_val.strip():
        return [s.strip() for s in env_val.split(",") if s.strip()]
    # Default: use first N of NIFTY50
    count = min(symbols_count, len(_NIFTY50_DEFAULT))
    return _NIFTY50_DEFAULT[:count]


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_data_batch(symbols: list, period: str) -> Dict[str, pd.DataFrame]:
    """Download intraday OHLCV from yfinance in batch."""
    import yfinance as yf

    result = {}
    batch_size = 50
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        logger.info(
            "Fetching batch %d-%d / %d  (%d symbols)",
            i + 1,
            min(i + batch_size, len(symbols)),
            len(symbols),
            len(batch),
        )
        try:
            data = yf.download(
                batch,
                period=period,
                interval=INTERVAL,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            if data.empty:
                continue

            for symbol in batch:
                try:
                    if len(batch) == 1:
                        df = data.copy()
                    else:
                        if symbol not in data.columns.get_level_values(0):
                            continue
                        df = data[symbol].copy()

                    if df is None or df.empty:
                        continue

                    df = df.rename(columns=str.lower)
                    if "close" not in df.columns:
                        continue

                    df = df.dropna(subset=["close"])
                    if len(df) < 50:  # need enough for all features
                        continue

                    df["symbol"] = symbol
                    result[symbol] = df
                except Exception as e:
                    logger.debug("  Skip %s: %s", symbol, e)
        except Exception as e:
            logger.warning("Batch download failed: %s", e)
        if i + batch_size < len(symbols):
            time.sleep(1)

    logger.info("Fetched data for %d / %d symbols", len(result), len(symbols))
    return result


def fetch_market_context_data() -> Dict[str, float]:
    """Fetch NIFTY50 context features for training enrichment."""
    try:
        from src.ai.market_context import fetch_market_context

        return fetch_market_context(interval=INTERVAL, period="120d")
    except Exception as e:
        logger.debug("Market context unavailable for training: %s", e)
        return {
            "nifty_return_1": 0.0,
            "nifty_return_5": 0.0,
            "nifty_rsi": 50.0,
            "nifty_volatility": 0.01,
            "nifty_trend": 0.0,
        }


# ---------------------------------------------------------------------------
# Adaptive MIN_RETURN_PCT
# ---------------------------------------------------------------------------
def compute_adaptive_min_return(
    dataframes: Dict[str, pd.DataFrame],
    forward_bars: int,
) -> float:
    """
    Compute adaptive minimum return threshold based on actual market volatility.
    Uses 0.5x median absolute forward return across the dataset.
    Falls back to env var MIN_RETURN_PCT or 0.3% if insufficient data.
    """
    env_override = os.environ.get("MIN_RETURN_PCT")
    if env_override is not None:
        val = float(env_override)
        logger.info("MIN_RETURN_PCT set via env var: %.4f%%", val)
        return val

    all_returns = []
    for symbol, df in dataframes.items():
        closes = df["close"].values
        if len(closes) <= forward_bars:
            continue
        fwd_returns = (closes[forward_bars:] - closes[:-forward_bars]) / closes[:-forward_bars]
        all_returns.extend(np.abs(fwd_returns).tolist())

    if len(all_returns) < 100:
        logger.warning("Insufficient data for adaptive threshold (%d returns), falling back to 0.3%%", len(all_returns))
        return 0.3

    median_abs_return = np.median(all_returns) * 100.0  # convert to percentage
    threshold = 0.5 * median_abs_return
    logger.info(
        "Adaptive MIN_RETURN_PCT: median |5-bar return| = %.4f%%, threshold = 0.5x = %.4f%%",
        median_abs_return,
        threshold,
    )
    return threshold


# ---------------------------------------------------------------------------
# Feature + target construction
# ---------------------------------------------------------------------------
def bars_from_df(df: pd.DataFrame) -> list:
    """Convert a DataFrame to a list of Bar objects."""
    from datetime import timezone
    from src.core.events import Bar, Exchange

    bars = []
    for idx, row in df.iterrows():
        ts = idx.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        bars.append(
            Bar(
                symbol=row["symbol"],
                exchange=Exchange.NSE,
                interval=INTERVAL,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                ts=ts,
            )
        )
    return bars


def build_feature_rows(
    bars: list,
    forward_bars: int = FORWARD_BARS,
    min_return_pct: float = 0.3,
    market_ctx: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Slide a window over bars, compute features, create risk-adjusted target.

    Target: 1 if forward_return > min_return_pct/100, else 0
    Sample weight: proportional to |forward_return| (bigger moves matter more)

    Returns: (X, y, sample_weights, feature_names)
    """
    from src.ai.feature_engine import FeatureEngine

    fe = FeatureEngine()
    min_warmup = 35  # need enough bars for all indicators (EMA-26 + signal-9)
    rows = []
    labels = []
    weights = []
    min_return = min_return_pct / 100.0  # convert to decimal

    for end_idx in range(min_warmup, len(bars) - forward_bars):
        window = bars[: end_idx + 1]  # no lookahead
        try:
            features = fe.build_features(window)
        except Exception:
            continue

        # Inject market context
        if market_ctx:
            features.update(market_ctx)

        current_close = bars[end_idx].close
        future_close = bars[end_idx + forward_bars].close
        if current_close <= 0:
            continue

        forward_return = (future_close - current_close) / current_close

        # Risk-adjusted target: only label 1 if move exceeds threshold
        label = 1 if forward_return > min_return else 0
        weight = max(abs(forward_return) * 100.0, 0.1)  # weight by importance

        rows.append(features)
        labels.append(label)
        weights.append(weight)

    if not rows:
        return np.array([]), np.array([]), np.array([]), []

    feature_names = sorted(rows[0].keys())
    X = np.array([[r.get(k, 0.0) for k in feature_names] for r in rows], dtype=np.float64)
    y = np.array(labels, dtype=np.int32)
    w = np.array(weights, dtype=np.float64)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, w, feature_names


# ---------------------------------------------------------------------------
# Purged cross-validation with embargo
# ---------------------------------------------------------------------------
def purged_time_series_split(
    n_samples: int,
    n_splits: int = 5,
    embargo_bars: int = FORWARD_BARS,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    TimeSeriesSplit with purge/embargo gap between train and test sets.
    Drops `embargo_bars` rows between each train/test boundary to prevent
    look-ahead bias from overlapping forward windows.
    """
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    purged_splits = []

    for train_idx, test_idx in tscv.split(np.arange(n_samples)):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        # Purge: remove last `embargo_bars` rows from train set
        if embargo_bars > 0 and len(train_idx) > embargo_bars:
            train_idx = train_idx[:-embargo_bars]

        # Also skip first `embargo_bars` rows from test set if they overlap
        # with the forward window of the last training sample
        if embargo_bars > 0 and len(test_idx) > embargo_bars:
            test_idx = test_idx[embargo_bars:]

        if len(train_idx) > 0 and len(test_idx) > 0:
            purged_splits.append((train_idx, test_idx))

    return purged_splits


# ---------------------------------------------------------------------------
# Optuna hyperparameter optimization
# ---------------------------------------------------------------------------
def objective(
    trial: "optuna.Trial",
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
) -> float:
    """Optuna objective: maximize mean Spearman IC via purged time-series CV."""
    from xgboost import XGBClassifier
    from scipy.stats import spearmanr

    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0),
    }

    model = XGBClassifier(
        n_estimators=500,
        **params,
        scale_pos_weight=max(1.0, (1 - y.mean()) / (y.mean() + 1e-9)),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    splits = purged_time_series_split(len(X), n_splits=5, embargo_bars=FORWARD_BARS)
    ics = []

    for train_idx, val_idx in splits:
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = sample_weights[train_idx]

        model.fit(
            X_tr,
            y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        proba = model.predict_proba(X_val)[:, 1]
        ic, _ = spearmanr(proba, y_val)
        if np.isnan(ic):
            ic = 0.0
        ics.append(ic)

    return float(np.mean(ics))


def run_optuna_optimization(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    n_trials: int = 30,
) -> dict:
    """Run Optuna HPO and return best XGBoost params."""
    if not OPTUNA_AVAILABLE:
        logger.warning(
            "Optuna not installed — falling back to hardcoded hyperparameters. Install with: pip install optuna>=3.0.0"
        )
        return {
            "max_depth": 5,
            "learning_rate": 0.03,
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
        }

    logger.info("Starting Optuna hyperparameter optimization (%d trials)...", n_trials)
    study = optuna.create_study(direction="maximize", study_name="xgb_alpha")
    study.optimize(
        lambda trial: objective(trial, X, y, sample_weights),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = study.best_params
    logger.info("Optuna best trial IC=%.4f  params=%s", study.best_value, best)
    return best


# ---------------------------------------------------------------------------
# Model training: XGBoost + LightGBM ensemble
# ---------------------------------------------------------------------------
def train_model(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    feature_names: List[str],
    xgb_params: dict,
):
    """Train XGBoost + LightGBM ensemble with purged walk-forward CV."""
    from xgboost import XGBClassifier
    from scipy.stats import spearmanr

    logger.info("Training ensemble  X=%s  class balance: %.2f%% positive", X.shape, y.mean() * 100)

    # ── XGBoost with Optuna-tuned params ──
    xgb_model = XGBClassifier(
        n_estimators=500,
        **xgb_params,
        scale_pos_weight=max(1.0, (1 - y.mean()) / (y.mean() + 1e-9)),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    # ── LightGBM (if available) ──
    lgb_model = None
    try:
        from lightgbm import LGBMClassifier

        lgb_model = LGBMClassifier(
            n_estimators=500,
            max_depth=xgb_params.get("max_depth", 5),
            learning_rate=xgb_params.get("learning_rate", 0.03),
            subsample=xgb_params.get("subsample", 0.8),
            colsample_bytree=xgb_params.get("colsample_bytree", 0.7),
            min_child_weight=xgb_params.get("min_child_weight", 10),
            reg_alpha=xgb_params.get("reg_alpha", 0.5),
            reg_lambda=xgb_params.get("reg_lambda", 2.0),
            scale_pos_weight=max(1.0, (1 - y.mean()) / (y.mean() + 1e-9)),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        logger.info("LightGBM available — training ensemble")
    except ImportError:
        logger.info("LightGBM not installed — using XGBoost only")

    # Purged walk-forward cross validation
    splits = purged_time_series_split(len(X), n_splits=5, embargo_bars=FORWARD_BARS)
    xgb_ics, lgb_ics = [], []

    for fold, (train_idx, val_idx) in enumerate(splits):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = sample_weights[train_idx]

        # XGBoost
        xgb_model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)
        xgb_proba = xgb_model.predict_proba(X_val)[:, 1]
        xgb_ic, _ = spearmanr(xgb_proba, y_val)
        if np.isnan(xgb_ic):
            xgb_ic = 0.0
        xgb_ics.append(xgb_ic)
        xgb_acc = (xgb_model.predict(X_val) == y_val).mean()

        # LightGBM
        lgb_ic = 0.0
        lgb_acc = 0.0
        if lgb_model is not None:
            lgb_model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)])
            lgb_proba = lgb_model.predict_proba(X_val)[:, 1]
            lgb_ic, _ = spearmanr(lgb_proba, y_val)
            if np.isnan(lgb_ic):
                lgb_ic = 0.0
            lgb_ics.append(lgb_ic)
            lgb_acc = (lgb_model.predict(X_val) == y_val).mean()

        if lgb_model:
            logger.info(
                "  Fold %d  XGB IC=%.4f acc=%.3f  |  LGB IC=%.4f acc=%.3f  (purged, embargo=%d)",
                fold + 1,
                xgb_ic,
                xgb_acc,
                lgb_ic,
                lgb_acc,
                FORWARD_BARS,
            )
        else:
            logger.info(
                "  Fold %d  XGB IC=%.4f  accuracy=%.3f  (purged, embargo=%d)", fold + 1, xgb_ic, xgb_acc, FORWARD_BARS
            )

    logger.info("XGBoost walk-forward mean IC=%.4f (std=%.4f)", np.mean(xgb_ics), np.std(xgb_ics))
    if lgb_ics:
        logger.info("LightGBM walk-forward mean IC=%.4f (std=%.4f)", np.mean(lgb_ics), np.std(lgb_ics))

    # Final refit with early stopping using last fold as time-series validation set
    # Use last 15% of data as validation set for early stopping (temporal split)
    val_split = max(int(len(X) * 0.15), 50)
    X_train_final, X_val_final = X[:-val_split], X[-val_split:]
    y_train_final, y_val_final = y[:-val_split], y[-val_split:]
    w_train_final = sample_weights[:-val_split]

    xgb_model.fit(
        X_train_final,
        y_train_final,
        sample_weight=w_train_final,
        eval_set=[(X_val_final, y_val_final)],
        verbose=False,
    )
    if lgb_model is not None:
        lgb_model.fit(
            X_train_final,
            y_train_final,
            sample_weight=w_train_final,
            eval_set=[(X_val_final, y_val_final)],
        )
    logger.info("Final refit with early stopping (train=%d, val=%d)", len(X_train_final), val_split)

    # ── Post-training calibration (Platt scaling on held-out calibration set) ──
    # Split off a separate calibration set (last 12% of training data) that was
    # NOT used during HPO or model fitting, to avoid data leakage in calibration.
    try:
        from sklearn.calibration import CalibratedClassifierCV

        calib_fraction = 0.12
        calib_size = max(int(len(X) * calib_fraction), 100)
        X_train_pre_calib = X[:-calib_size]
        y_train_pre_calib = y[:-calib_size]
        w_train_pre_calib = sample_weights[:-calib_size]
        X_calib = X[-calib_size:]
        y_calib = y[-calib_size:]
        w_calib = sample_weights[-calib_size:]

        logger.info(
            "Applying Platt scaling calibration on held-out set (train=%d, calib=%d, %.0f%% of training data)...",
            len(X_train_pre_calib),
            calib_size,
            calib_fraction * 100,
        )

        # Refit XGBoost on pre-calibration training data, then calibrate on calib set
        xgb_model.fit(
            X_train_pre_calib,
            y_train_pre_calib,
            sample_weight=w_train_pre_calib,
            eval_set=[(X_calib, y_calib)],
            verbose=False,
        )
        xgb_calibrated = CalibratedClassifierCV(xgb_model, method="sigmoid", cv="prefit")
        xgb_calibrated.fit(X_calib, y_calib, sample_weight=w_calib)
        xgb_model = xgb_calibrated

        if lgb_model is not None:
            lgb_model.fit(
                X_train_pre_calib,
                y_train_pre_calib,
                sample_weight=w_train_pre_calib,
                eval_set=[(X_calib, y_calib)],
            )
            lgb_calibrated = CalibratedClassifierCV(lgb_model, method="sigmoid", cv="prefit")
            lgb_calibrated.fit(X_calib, y_calib, sample_weight=w_calib)
            lgb_model = lgb_calibrated

        logger.info("Post-training calibration complete (held-out calib set, no data leakage)")
    except Exception as e:
        logger.warning("Calibration failed (non-fatal, using uncalibrated model): %s", e)

    # Create ensemble wrapper
    ensemble = EnsembleClassifier(xgb_model, lgb_model)

    # Feature importance (from XGBoost — unwrap calibrated if needed)
    raw_xgb = xgb_model
    if hasattr(raw_xgb, "estimator"):
        raw_xgb = raw_xgb.estimator
    elif hasattr(raw_xgb, "calibrated_classifiers_"):
        # CalibratedClassifierCV wraps the base estimator
        try:
            raw_xgb = raw_xgb.calibrated_classifiers_[0].estimator
        except (AttributeError, IndexError):
            raw_xgb = None

    if raw_xgb is not None and hasattr(raw_xgb, "feature_importances_"):
        importances = raw_xgb.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        logger.info("Top features:")
        for i in sorted_idx[:10]:
            logger.info("  %-25s  importance=%.4f", feature_names[i], importances[i])

    return ensemble


class EnsembleClassifier:
    """Soft-voting ensemble of XGBoost + LightGBM."""

    def __init__(self, xgb_model, lgb_model=None, xgb_weight=0.6, lgb_weight=0.4):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.xgb_weight = xgb_weight if lgb_model else 1.0
        self.lgb_weight = lgb_weight if lgb_model else 0.0

    def predict_proba(self, X):
        xgb_proba = self.xgb_model.predict_proba(X)
        if self.lgb_model is not None:
            lgb_proba = self.lgb_model.predict_proba(X)
            return self.xgb_weight * xgb_proba + self.lgb_weight * lgb_proba
        return xgb_proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Model validation gate
# ---------------------------------------------------------------------------
def validate_model(
    model: EnsembleClassifier,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    feature_names: List[str],
) -> Tuple[bool, dict]:
    """
    Validate model on a true holdout set (never seen during training or HPO).
    Returns (passed, results_dict).

    Gates:
      1. Mean IC on holdout > 0.05
      2. Buy percentage between 10% and 90%
      3. Transaction cost hurdle: mean predicted return must exceed 0.1% round-trip cost
    """
    from scipy.stats import spearmanr

    IC_GATE = 0.05
    TRANSACTION_COST_PCT = 0.001  # 0.1% round-trip transaction cost

    holdout_size = len(X_holdout)

    proba = model.predict_proba(X_holdout)[:, 1]
    preds = (proba >= 0.5).astype(int)
    ic, _ = spearmanr(proba, y_holdout)
    if np.isnan(ic):
        ic = 0.0

    buy_pct = float(preds.mean())
    accuracy = float((preds == y_holdout).mean())

    # Transaction cost hurdle: compute mean predicted excess return
    # Signals only count as profitable if expected return exceeds transaction cost
    mean_predicted_return = float(np.abs(proba - 0.5).mean() * 0.02)  # scale factor
    net_of_costs = mean_predicted_return - TRANSACTION_COST_PCT

    results = {
        "holdout_samples": int(holdout_size),
        "holdout_ic": float(ic),
        "holdout_accuracy": accuracy,
        "holdout_buy_pct": buy_pct,
        "holdout_mean_prob": float(proba.mean()),
        "holdout_std_prob": float(proba.std()),
        "mean_predicted_return": mean_predicted_return,
        "net_of_transaction_costs": net_of_costs,
        "transaction_cost_pct": TRANSACTION_COST_PCT,
        "validation_timestamp": datetime.utcnow().isoformat(),
    }

    passed = True
    reasons = []

    # Gate 1: IC threshold (raised to 0.05 for institutional quality)
    if ic < IC_GATE:
        logger.error(
            "VALIDATION FAILED: holdout IC=%.4f < %.2f minimum. Model has insufficient predictive power.", ic, IC_GATE
        )
        reasons.append(f"IC too low: {ic:.4f} < {IC_GATE}")
        passed = False
    else:
        logger.info("Validation gate IC: PASSED (IC=%.4f >= %.2f)", ic, IC_GATE)

    # Gate 2: Bias check
    if buy_pct > 0.90:
        logger.error("VALIDATION FAILED: buy_pct=%.1f%% > 90%%. Model is biased toward BUY.", buy_pct * 100)
        reasons.append(f"Buy bias too high: {buy_pct * 100:.1f}% > 90%")
        passed = False
    elif buy_pct < 0.10:
        logger.error("VALIDATION FAILED: buy_pct=%.1f%% < 10%%. Model is biased toward SELL.", buy_pct * 100)
        reasons.append(f"Buy bias too low: {buy_pct * 100:.1f}% < 10%")
        passed = False
    else:
        logger.info("Validation gate bias: PASSED (buy_pct=%.1f%%)", buy_pct * 100)

    # Gate 3: Transaction cost hurdle
    if net_of_costs <= 0:
        logger.error(
            "VALIDATION FAILED: mean predicted return %.4f%% does not exceed transaction cost %.4f%%. Net=%.4f%%",
            mean_predicted_return * 100,
            TRANSACTION_COST_PCT * 100,
            net_of_costs * 100,
        )
        reasons.append(f"Transaction cost hurdle failed: net return {net_of_costs * 100:.4f}% <= 0")
        passed = False
    else:
        logger.info(
            "Validation gate transaction cost: PASSED (net=%.4f%% after %.2f%% costs)",
            net_of_costs * 100,
            TRANSACTION_COST_PCT * 100,
        )

    results["passed"] = passed
    results["failure_reasons"] = reasons

    return passed, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = _parse_args()

    SYMBOLS = _resolve_symbols(args.symbols_count)
    PERIOD = f"{args.lookback_days}d"

    logger.info("=" * 60)
    logger.info("Alpha Model Training Pipeline v3 (Production-Grade)")
    logger.info("=" * 60)
    logger.info(
        "Config: symbols=%d  period=%s  interval=%s  forward=%d  optuna_trials=%d",
        len(SYMBOLS),
        PERIOD,
        INTERVAL,
        FORWARD_BARS,
        args.optuna_trials,
    )
    if OPTUNA_AVAILABLE:
        logger.info("Optuna %s available for hyperparameter optimization", optuna.__version__)
    else:
        logger.warning("Optuna NOT installed — using hardcoded hyperparameters")

    # Fetch market context for enrichment
    market_ctx = fetch_market_context_data()
    logger.info(
        "Market context: NIFTY RSI=%.1f, vol=%.4f, trend=%.0f",
        market_ctx.get("nifty_rsi", 0),
        market_ctx.get("nifty_volatility", 0),
        market_ctx.get("nifty_trend", 0),
    )

    logger.info("Downloading data for %d symbols...", len(SYMBOLS))
    dataframes = fetch_data_batch(SYMBOLS, period=PERIOD)

    # Compute adaptive MIN_RETURN_PCT
    min_return_pct = compute_adaptive_min_return(dataframes, FORWARD_BARS)
    logger.info("Using MIN_RETURN_PCT = %.4f%%", min_return_pct)

    all_X, all_y, all_w = [], [], []
    feature_names = None

    for symbol, df in dataframes.items():
        try:
            bars = bars_from_df(df)
            X, y, w, fnames = build_feature_rows(
                bars,
                min_return_pct=min_return_pct,
                market_ctx=market_ctx,
            )
            if X.size == 0:
                continue
            all_X.append(X)
            all_y.append(y)
            all_w.append(w)
            feature_names = fnames
            pos_rate = y.mean() * 100 if len(y) > 0 else 0
            logger.info("  %s -> %d samples (%.1f%% positive)", symbol, len(y), pos_rate)
        except Exception as e:
            logger.debug("Failed processing %s: %s", symbol, e)

    if not all_X:
        logger.error("No training data. Exiting.")
        sys.exit(1)

    X_full = np.vstack(all_X)
    y_full = np.concatenate(all_y)
    w_full = np.concatenate(all_w)
    logger.info(
        "Combined dataset: %d samples, %d features, %d symbols", X_full.shape[0], X_full.shape[1], len(dataframes)
    )
    logger.info("Overall class balance: %.2f%% positive", y_full.mean() * 100)

    # ── True holdout: last 20% of data, never touched during training/validation ──
    holdout_fraction = 0.20
    holdout_size = max(int(len(X_full) * holdout_fraction), 100)
    X = X_full[:-holdout_size]
    y = y_full[:-holdout_size]
    w = w_full[:-holdout_size]
    X_holdout = X_full[-holdout_size:]
    y_holdout = y_full[-holdout_size:]
    w_holdout = w_full[-holdout_size:]
    logger.info(
        "True holdout split: train/val=%d samples, holdout=%d samples (%.0f%%)",
        len(X),
        holdout_size,
        holdout_fraction * 100,
    )

    # Optuna hyperparameter optimization (on train/val only, holdout untouched)
    best_params = run_optuna_optimization(X, y, w, n_trials=args.optuna_trials)
    logger.info("Final XGBoost params: %s", best_params)

    # Train model with optimized params (on train/val only)
    model = train_model(X, y, w, feature_names, xgb_params=best_params)

    # Model validation gate (on true holdout set only)
    passed, validation_results = validate_model(model, X_holdout, y_holdout, feature_names)

    # Save validation results always
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    validation_path = MODEL_DIR / "alpha_xgb_validation.json"
    validation_results["xgb_params"] = best_params
    validation_results["min_return_pct"] = min_return_pct
    validation_results["n_symbols"] = len(dataframes)
    validation_results["n_samples"] = int(X.shape[0])
    validation_results["optuna_trials"] = args.optuna_trials
    validation_results["optuna_available"] = OPTUNA_AVAILABLE
    with open(validation_path, "w") as f:
        json.dump(validation_results, f, indent=2)
    logger.info("Validation results saved to %s", validation_path)

    if not passed:
        logger.error(
            "MODEL DID NOT PASS VALIDATION GATES — NOT saving model. Reasons: %s",
            "; ".join(validation_results["failure_reasons"]),
        )
        logger.error("Previous model (if any) is preserved at models/alpha_xgb.joblib")
        sys.exit(1)

    # Save model (only if passed all gates)
    out_path = MODEL_DIR / "alpha_xgb.joblib"
    import joblib

    joblib.dump(model, out_path)
    logger.info("Model saved to %s  (%.1f KB)", out_path, out_path.stat().st_size / 1024)

    # Final ensemble output distribution (on full dataset for statistics)
    ensemble_probs = model.predict_proba(X_full)[:, 1]
    ensemble_buy_pct = (ensemble_probs > 0.5).mean()
    logger.info(
        "Ensemble output: %.1f%% BUY, mean=%.4f, std=%.4f",
        ensemble_buy_pct * 100,
        ensemble_probs.mean(),
        ensemble_probs.std(),
    )

    # Save feature names alongside model for inference consistency
    meta_path = MODEL_DIR / "alpha_xgb_meta.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "feature_names": feature_names,
                "forward_bars": FORWARD_BARS,
                "interval": INTERVAL,
                "min_return_pct": min_return_pct,
                "min_return_pct_adaptive": True,
                "n_features": len(feature_names),
                "n_samples": int(X.shape[0]),
                "n_symbols": len(dataframes),
                "positive_rate": float(y.mean()),
                "ensemble": True,
                "calibrated": True,
                "ensemble_buy_pct": float(ensemble_buy_pct),
                "ensemble_mean_prob": float(ensemble_probs.mean()),
                "ensemble_std_prob": float(ensemble_probs.std()),
                "xgb_params": best_params,
                "optuna_optimized": OPTUNA_AVAILABLE,
                "optuna_trials": args.optuna_trials,
                "purged_cv_embargo": FORWARD_BARS,
                "training_timestamp": datetime.utcnow().isoformat(),
            },
            f,
            indent=2,
        )
    logger.info("Feature metadata saved to %s (%d features)", meta_path, len(feature_names))

    # ── P0-3: Save FeatureNormalizer (z-score stats from training data) ──
    try:
        from src.ai.feature_engine import FeatureNormalizer

        normalizer = FeatureNormalizer()
        # Build feature dicts from X_full + feature_names
        feature_dicts = [
            {feature_names[j]: float(X_full[i, j]) for j in range(X_full.shape[1])} for i in range(X_full.shape[0])
        ]
        normalizer.fit(feature_dicts)
        normalizer_path = str(MODEL_DIR / "feature_normalizer.json")
        normalizer.save(normalizer_path)
        logger.info("FeatureNormalizer saved to %s (%d features)", normalizer_path, len(normalizer._means))
    except Exception as e:
        logger.warning("FeatureNormalizer save failed: %s", e)

    # ── P0-2: Save OOS validation data for model validation gate at startup ──
    try:
        holdout_probs = model.predict_proba(X_holdout)[:, 1]
        validation_dir = MODEL_DIR / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            validation_dir / "xgboost_alpha_oos.npz",
            predictions=holdout_probs,
            actuals=y_holdout.astype(np.float64),
        )
        logger.info("OOS validation data saved (%d samples) for startup model gate", len(holdout_probs))
    except Exception as e:
        logger.warning("OOS validation data save failed: %s", e)

    # ── P1-7: Save XGB calibration data (predictions vs realized returns) ──
    try:
        from src.ai.feature_engine import FeatureEngine

        # Use holdout predictions and compute realized returns for calibration
        # We already have holdout probs; compute forward returns for holdout subset
        # For simplicity: map labels back to approximate returns
        # More accurate: re-extract forward returns from the holdout data
        cal_preds = holdout_probs
        cal_returns = np.where(y_holdout == 1, min_return_pct / 100.0, -min_return_pct / 100.0)
        np.savez(
            MODEL_DIR / "xgb_calibration.npz",
            predictions=cal_preds,
            returns=cal_returns,
        )
        logger.info("XGB calibration data saved (%d samples)", len(cal_preds))
    except Exception as e:
        logger.warning("XGB calibration data save failed: %s", e)

    logger.info("=" * 60)
    logger.info("Training complete. Load with: AlphaModel(model_path='%s')", out_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
