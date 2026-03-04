"""
Example model training pipeline: load features from feature store (or CSV),
train XGBoost classifier, return model and metrics for registry.
Used by RetrainPipeline train_fn. Walk-forward validation in backtest layer.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_feature_matrix(
    feature_vectors: List[Any],
    target_col: str = "target",
    feature_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    feature_vectors: list of dict with 'features' and optional 'target'.
    Returns X, y, feature_names.
    """
    if not feature_vectors:
        return np.zeros((0, 0)), np.zeros(0), feature_names or []
    names = feature_names or sorted(feature_vectors[0].get("features", {}).keys())
    X = np.array([[v.get("features", {}).get(n, 0) for n in names] for v in feature_vectors], dtype=np.float32)
    y = np.array([v.get(target_col, 0) for v in feature_vectors], dtype=np.float32)
    return X, y, names


def train_xgb_direction(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, float]]:
    """
    Train XGBoost binary classifier for direction (1=up, 0=down).
    Returns (model, metrics dict with accuracy, logloss, sharpe_proxy).
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("xgboost not installed; returning stub")
        return None, {"accuracy": 0.0, "logloss": 0.0, "sharpe": 0.0}

    params = params or {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 100}
    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)
    bst = xgb.train(
        {**params, "objective": "binary:logistic", "eval_metric": "logloss"},
        dtrain,
        num_boost_round=params.get("n_estimators", 100),
    )
    pred = bst.predict(dtrain)
    acc = float(np.mean((pred >= 0.5) == y))
    logloss = float(-np.mean(y * np.log(pred + 1e-6) + (1 - y) * np.log(1 - pred + 1e-6)))
    # Sharpe proxy from predicted returns
    fake_ret = np.where(pred >= 0.5, 1.0, -1.0) * (2 * y - 1)
    sharpe = float(np.mean(fake_ret) / (np.std(fake_ret) + 1e-12)) if len(fake_ret) > 1 else 0.0
    return bst, {"accuracy": acc, "logloss": logloss, "sharpe": sharpe}


def run_example_training(
    data_path: Optional[Path] = None,
) -> Tuple[Any, Dict[str, float]]:
    """
    Example entrypoint: load data, train, return model and metrics.
    If data_path is None, use synthetic data.
    """
    if data_path and data_path.exists():
        import json
        with open(data_path) as f:
            vectors = json.load(f)
        X, y, names = load_feature_matrix(vectors, target_col="target")
    else:
        np.random.seed(42)
        n = 500
        names = ["returns_1m", "rolling_vol_20", "momentum_5", "zscore_20"]
        X = np.random.randn(n, len(names)).astype(np.float32) * 0.01
        y = (np.random.rand(n) > 0.5).astype(np.float32)
    if X.shape[0] < 50:
        return None, {"accuracy": 0.0, "logloss": 0.0, "sharpe": 0.0}
    return train_xgb_direction(X, y, names)
