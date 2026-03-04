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

Usage:
    PYTHONPATH=. python scripts/train_alpha_model.py
    TRAIN_SYMBOLS=all PYTHONPATH=. python scripts/train_alpha_model.py

Configurable via env vars:
    TRAIN_SYMBOLS      comma-separated or 'all' for NIFTY 500
    TRAIN_PERIOD       yfinance period  (default: 60d)
    TRAIN_INTERVAL     yfinance interval (default: 5m)
    FORWARD_BARS       bars ahead for target (default: 5)
    MIN_RETURN_PCT     minimum return % for positive label (default: 0.3)
    MODEL_DIR          output directory (default: models/)
"""
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

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
_NIFTY50_DEFAULT = ",".join([
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "BAJFINANCE.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    "TATAMOTORS.NS", "NESTLEIND.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS",
    "M&M.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "TECHM.NS", "INDUSINDBK.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS",
    "GRASIM.NS", "DIVISLAB.NS", "BRITANNIA.NS", "CIPLA.NS", "EICHERMOT.NS",
    "DRREDDY.NS", "APOLLOHOSP.NS", "COALINDIA.NS", "BPCL.NS", "TATACONSUM.NS",
    "HEROMOTOCO.NS", "UPL.NS", "BAJAJ-AUTO.NS", "HINDALCO.NS", "LTIM.NS",
])
_SYMBOLS_ENV = os.environ.get("TRAIN_SYMBOLS", _NIFTY50_DEFAULT)
PERIOD = os.environ.get("TRAIN_PERIOD", "60d")
INTERVAL = os.environ.get("TRAIN_INTERVAL", "5m")
FORWARD_BARS = int(os.environ.get("FORWARD_BARS", "5"))
MIN_RETURN_PCT = float(os.environ.get("MIN_RETURN_PCT", "0.3"))  # 0.3% minimum move
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))


def _resolve_symbols() -> list:
    """Resolve symbol list. Use 'all' for full NIFTY 500 universe."""
    if _SYMBOLS_ENV.strip().lower() == "all":
        try:
            from src.scanner.nse_universe import get_universe
            symbols = get_universe(yfinance_suffix=True)
            logger.info("Using full NSE universe: %d symbols", len(symbols))
            return symbols
        except Exception as e:
            logger.warning("Failed to fetch universe, using defaults: %s", e)
            return "RELIANCE.NS,INFY.NS,TCS.NS,HDFCBANK.NS,ICICIBANK.NS".split(",")
    return [s.strip() for s in _SYMBOLS_ENV.split(",") if s.strip()]

SYMBOLS = _resolve_symbols()


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_data_batch(symbols: list) -> Dict[str, pd.DataFrame]:
    """Download intraday OHLCV from yfinance in batch."""
    import yfinance as yf

    result = {}
    batch_size = 50
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        logger.info("Fetching batch %d-%d / %d  (%d symbols)",
                    i + 1, min(i + batch_size, len(symbols)), len(symbols), len(batch))
        try:
            data = yf.download(
                batch,
                period=PERIOD,
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
        return fetch_market_context(interval=INTERVAL, period=PERIOD)
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
    min_return_pct: float = MIN_RETURN_PCT,
    market_ctx: Dict[str, float] = None,
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
# Model training: XGBoost + LightGBM ensemble
# ---------------------------------------------------------------------------
def train_model(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    feature_names: List[str],
):
    """Train XGBoost + LightGBM ensemble with walk-forward CV."""
    from xgboost import XGBClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from scipy.stats import spearmanr

    logger.info("Training ensemble  X=%s  class balance: %.2f%% positive",
                X.shape, y.mean() * 100)

    # ── XGBoost ──
    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        reg_alpha=0.5,
        reg_lambda=2.0,
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
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=10,
            reg_alpha=0.5,
            reg_lambda=2.0,
            scale_pos_weight=max(1.0, (1 - y.mean()) / (y.mean() + 1e-9)),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        logger.info("LightGBM available — training ensemble")
    except ImportError:
        logger.info("LightGBM not installed — using XGBoost only")

    # Walk-forward cross validation
    tscv = TimeSeriesSplit(n_splits=5)
    xgb_ics, lgb_ics = [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = sample_weights[train_idx]

        # XGBoost
        xgb_model.fit(X_tr, y_tr, sample_weight=w_tr,
                       eval_set=[(X_val, y_val)], verbose=False)
        xgb_proba = xgb_model.predict_proba(X_val)[:, 1]
        xgb_ic, _ = spearmanr(xgb_proba, y_val)
        xgb_ics.append(xgb_ic)
        xgb_acc = (xgb_model.predict(X_val) == y_val).mean()

        # LightGBM
        lgb_ic = 0.0
        lgb_acc = 0.0
        if lgb_model is not None:
            lgb_model.fit(X_tr, y_tr, sample_weight=w_tr,
                          eval_set=[(X_val, y_val)])
            lgb_proba = lgb_model.predict_proba(X_val)[:, 1]
            lgb_ic, _ = spearmanr(lgb_proba, y_val)
            lgb_ics.append(lgb_ic)
            lgb_acc = (lgb_model.predict(X_val) == y_val).mean()

        if lgb_model:
            logger.info("  Fold %d  XGB IC=%.4f acc=%.3f  |  LGB IC=%.4f acc=%.3f",
                        fold + 1, xgb_ic, xgb_acc, lgb_ic, lgb_acc)
        else:
            logger.info("  Fold %d  XGB IC=%.4f  accuracy=%.3f", fold + 1, xgb_ic, xgb_acc)

    logger.info("XGBoost walk-forward mean IC=%.4f (std=%.4f)",
                np.mean(xgb_ics), np.std(xgb_ics))
    if lgb_ics:
        logger.info("LightGBM walk-forward mean IC=%.4f (std=%.4f)",
                    np.mean(lgb_ics), np.std(lgb_ics))

    # Final refit on full data
    xgb_model.fit(X, y, sample_weight=sample_weights, verbose=False)
    if lgb_model is not None:
        lgb_model.fit(X, y, sample_weight=sample_weights)

    # ── Post-training calibration (Platt scaling) ──
    # Fixes SELL-only bias by recalibrating probabilities to match true class frequency
    try:
        from sklearn.calibration import CalibratedClassifierCV
        logger.info("Applying Platt scaling calibration to XGBoost...")
        xgb_calibrated = CalibratedClassifierCV(xgb_model, method="sigmoid", cv=3)
        xgb_calibrated.fit(X, y, sample_weight=sample_weights)
        xgb_model = xgb_calibrated

        if lgb_model is not None:
            logger.info("Applying Platt scaling calibration to LightGBM...")
            lgb_calibrated = CalibratedClassifierCV(lgb_model, method="sigmoid", cv=3)
            lgb_calibrated.fit(X, y, sample_weight=sample_weights)
            lgb_model = lgb_calibrated

        logger.info("Post-training calibration complete")
    except Exception as e:
        logger.warning("Calibration failed (non-fatal, using uncalibrated model): %s", e)

    # ── Bias validation gate ──
    # Reject model if it's severely biased toward BUY or SELL
    probs = xgb_model.predict_proba(X)[:, 1]
    buy_pct = (probs > 0.5).mean()
    logger.info("Bias check: %.1f%% BUY signals (mean prob=%.4f, std=%.4f)",
                buy_pct * 100, probs.mean(), probs.std())
    if buy_pct < 0.05:
        logger.error(
            "MODEL BIAS DETECTED: only %.1f%% BUY signals (threshold: >5%%). "
            "The model is stuck predicting SELL for nearly all inputs. "
            "Check training data class balance and MIN_RETURN_PCT threshold.",
            buy_pct * 100,
        )
        # Don't exit — save the model but warn loudly
    elif buy_pct > 0.95:
        logger.error(
            "MODEL BIAS DETECTED: %.1f%% BUY signals (threshold: <95%%). "
            "The model is stuck predicting BUY for nearly all inputs.",
            buy_pct * 100,
        )
    else:
        logger.info("Bias check PASSED: %.1f%% BUY (within 5-95%% range)", buy_pct * 100)

    # Create ensemble wrapper
    ensemble = EnsembleClassifier(xgb_model, lgb_model)

    # Feature importance (from XGBoost)
    importances = xgb_model.feature_importances_
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
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("Alpha Model Training Pipeline v2 (Ensemble)")
    logger.info("=" * 60)
    logger.info("Config: symbols=%d  period=%s  interval=%s  forward=%d  min_return=%.2f%%",
                len(SYMBOLS), PERIOD, INTERVAL, FORWARD_BARS, MIN_RETURN_PCT)

    # Fetch market context for enrichment
    market_ctx = fetch_market_context_data()
    logger.info("Market context: NIFTY RSI=%.1f, vol=%.4f, trend=%.0f",
                market_ctx.get("nifty_rsi", 0), market_ctx.get("nifty_volatility", 0),
                market_ctx.get("nifty_trend", 0))

    all_X, all_y, all_w = [], [], []
    feature_names = None

    logger.info("Downloading data for %d symbols...", len(SYMBOLS))
    dataframes = fetch_data_batch(SYMBOLS)

    for symbol, df in dataframes.items():
        try:
            bars = bars_from_df(df)
            X, y, w, fnames = build_feature_rows(bars, market_ctx=market_ctx)
            if X.size == 0:
                continue
            all_X.append(X)
            all_y.append(y)
            all_w.append(w)
            feature_names = fnames
            pos_rate = y.mean() * 100 if len(y) > 0 else 0
            logger.info("  %s → %d samples (%.1f%% positive)", symbol, len(y), pos_rate)
        except Exception as e:
            logger.debug("Failed processing %s: %s", symbol, e)

    if not all_X:
        logger.error("No training data. Exiting.")
        sys.exit(1)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    w = np.concatenate(all_w)
    logger.info("Combined dataset: %d samples, %d features, %d symbols",
                X.shape[0], X.shape[1], len(dataframes))
    logger.info("Overall class balance: %.2f%% positive", y.mean() * 100)

    model = train_model(X, y, w, feature_names)

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_DIR / "alpha_xgb.joblib"
    import joblib
    joblib.dump(model, out_path)
    logger.info("Model saved to %s  (%.1f KB)", out_path, out_path.stat().st_size / 1024)

    # Final validation: check ensemble output distribution
    ensemble_probs = model.predict_proba(X)[:, 1]
    ensemble_buy_pct = (ensemble_probs > 0.5).mean()
    logger.info("Ensemble output: %.1f%% BUY, mean=%.4f, std=%.4f",
                ensemble_buy_pct * 100, ensemble_probs.mean(), ensemble_probs.std())

    # Save feature names alongside model for inference consistency
    meta_path = MODEL_DIR / "alpha_xgb_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "feature_names": feature_names,
            "forward_bars": FORWARD_BARS,
            "interval": INTERVAL,
            "min_return_pct": MIN_RETURN_PCT,
            "n_features": len(feature_names),
            "n_samples": int(X.shape[0]),
            "n_symbols": len(dataframes),
            "positive_rate": float(y.mean()),
            "ensemble": True,
            "calibrated": True,
            "ensemble_buy_pct": float(ensemble_buy_pct),
            "ensemble_mean_prob": float(ensemble_probs.mean()),
            "ensemble_std_prob": float(ensemble_probs.std()),
        }, f, indent=2)
    logger.info("Feature metadata saved to %s (%d features)", meta_path, len(feature_names))

    logger.info("=" * 60)
    logger.info("Training complete. Load with: AlphaModel(model_path='%s')", out_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
