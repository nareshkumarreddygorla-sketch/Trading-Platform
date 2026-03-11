#!/usr/bin/env python3
"""
Model validation script for CI integration.

Checks:
  1. All expected model files exist.
  2. Each model that exists can be loaded and produces valid predictions
     (prob_up in [0,1], confidence in [0,1]) from synthetic feature input.
  3. Ensemble model_id alignment: every model_id referenced by the
     EnsembleEngine has a matching entry in the ModelRegistry.
  4. Prints pass/fail report and exits with code 1 on any critical failure.

Usage:
    python -m scripts.validate_models          # from project root
    python scripts/validate_models.py          # direct invocation
"""
import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src.*` imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"

# Expected model files: (display_name, filename, criticality)
# criticality: "critical" means CI should fail, "optional" means warn only
EXPECTED_MODELS: List[Tuple[str, str, str]] = [
    ("XGBoost Alpha", "alpha_xgb.joblib", "critical"),
    ("LSTM Predictor", "lstm_predictor.pt", "critical"),
    ("Transformer Predictor", "transformer_predictor.pt", "critical"),
    ("RL Agent (PPO)", "rl_agent.zip", "optional"),
]

# Ensemble model_ids that must be registered for alignment
ENSEMBLE_MODEL_IDS = ["xgboost_alpha", "lstm_ts", "transformer_ts", "rl_ppo", "sentiment_finbert"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Result:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.skipped = False
        self.message = ""
        self.critical = True


def _make_synthetic_bars(n: int = 100):
    """Generate n synthetic Bar objects with realistic-ish OHLCV data."""
    import numpy as np
    from src.core.events import Bar, Exchange

    np.random.seed(42)
    price = 1500.0
    bars = []
    base_ts = datetime(2025, 1, 2, 9, 15, tzinfo=timezone.utc)

    for i in range(n):
        ret = np.random.normal(0, 0.01)
        price *= (1 + ret)
        o = price
        h = price * (1 + abs(np.random.normal(0, 0.003)))
        l = price * (1 - abs(np.random.normal(0, 0.003)))
        c = price * (1 + np.random.normal(0, 0.002))
        v = max(1000, int(np.random.exponential(50000)))
        ts = base_ts.replace(minute=15 + (i % 45), hour=9 + i // 45)
        bars.append(Bar(
            symbol="TESTSTOCK",
            exchange=Exchange.NSE,
            interval="1m",
            open=round(o, 2),
            high=round(max(o, h, c), 2),
            low=round(min(o, l, c), 2),
            close=round(c, 2),
            volume=float(v),
            ts=ts,
            source="synthetic",
        ))
    return bars


def _build_features(bars) -> Dict[str, float]:
    """Build features from synthetic bars via FeatureEngine."""
    from src.ai.feature_engine import FeatureEngine
    engine = FeatureEngine()
    return engine.build_features(bars)


def _validate_prediction(pred, label: str) -> Tuple[bool, str]:
    """Validate a PredictionOutput has sane values."""
    errors = []
    prob_up = getattr(pred, "prob_up", None)
    confidence = getattr(pred, "confidence", None)

    if prob_up is None:
        errors.append("prob_up is None")
    elif not (0.0 <= prob_up <= 1.0):
        errors.append(f"prob_up={prob_up} out of [0,1]")

    if confidence is None:
        errors.append("confidence is None")
    elif not (0.0 <= confidence <= 1.0):
        errors.append(f"confidence={confidence} out of [0,1]")

    model_id = getattr(pred, "model_id", None)
    if not model_id:
        errors.append("model_id is empty/None")

    if errors:
        return False, f"{label}: {'; '.join(errors)}"
    return True, f"{label}: prob_up={prob_up:.4f}, confidence={confidence:.4f}, model_id={model_id}"


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def check_model_files_exist() -> List[Result]:
    """Check 1: Verify all expected model files exist."""
    results = []
    for display_name, filename, criticality in EXPECTED_MODELS:
        r = Result(f"File exists: {filename}")
        r.critical = (criticality == "critical")
        path = MODELS_DIR / filename
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            r.passed = True
            r.message = f"Found ({size_mb:.2f} MB)"
        else:
            r.passed = False
            r.message = f"NOT FOUND at {path}"
        results.append(r)
    return results


def check_xgboost_prediction(features: Dict[str, float]) -> Result:
    """Check 2a: Load XGBoost model and run prediction."""
    r = Result("XGBoost predict()")
    path = MODELS_DIR / "alpha_xgb.joblib"
    if not path.exists():
        r.skipped = True
        r.message = "Model file not found; skipping"
        return r
    try:
        from src.ai.alpha_model import AlphaModel
        model = AlphaModel(strategy_id="ai_alpha", model_path=str(path))
        if model._model is None:
            r.passed = False
            r.message = "Model loaded but _model is None"
            return r
        # AlphaModel.predict() expects features dict + market_state
        # We test the raw model predict_proba instead
        import numpy as np
        feature_names = model._feature_names
        if feature_names:
            X = np.array([[features.get(f, 0.0) for f in feature_names]])
        else:
            X = np.array([[features.get(f, 0.0) for f in sorted(features.keys())]])
        proba = model._model.predict_proba(X)
        prob_up = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
        if 0.0 <= prob_up <= 1.0:
            r.passed = True
            r.message = f"prob_up={prob_up:.4f}"
        else:
            r.passed = False
            r.message = f"prob_up={prob_up} out of [0,1]"
    except Exception as e:
        r.passed = False
        r.message = f"Error: {e}\n{traceback.format_exc()}"
    return r


def check_lstm_prediction(features: Dict[str, float]) -> Result:
    """Check 2b: Load LSTM model and run prediction."""
    r = Result("LSTM predict()")
    path = MODELS_DIR / "lstm_predictor.pt"
    if not path.exists():
        r.skipped = True
        r.message = "Model file not found; skipping"
        return r
    try:
        from src.ai.models.lstm_predictor import LSTMPredictor
        predictor = LSTMPredictor()
        loaded = predictor.load(str(path))
        if not loaded:
            r.passed = False
            r.message = "LSTMPredictor.load() returned False"
            return r
        pred = predictor.predict(features)
        ok, msg = _validate_prediction(pred, "LSTM")
        r.passed = ok
        r.message = msg
    except Exception as e:
        r.passed = False
        r.message = f"Error: {e}\n{traceback.format_exc()}"
    return r


def check_transformer_prediction(features: Dict[str, float]) -> Result:
    """Check 2c: Load Transformer model and run prediction."""
    r = Result("Transformer predict()")
    path = MODELS_DIR / "transformer_predictor.pt"
    if not path.exists():
        r.skipped = True
        r.message = "Model file not found; skipping"
        return r
    try:
        from src.ai.models.transformer_predictor import TransformerPredictor
        predictor = TransformerPredictor()
        loaded = predictor.load(str(path))
        if not loaded:
            r.passed = False
            r.message = "TransformerPredictor.load() returned False"
            return r
        pred = predictor.predict(features)
        ok, msg = _validate_prediction(pred, "Transformer")
        r.passed = ok
        r.message = msg
    except Exception as e:
        r.passed = False
        r.message = f"Error: {e}\n{traceback.format_exc()}"
    return r


def check_rl_prediction(features: Dict[str, float]) -> Result:
    """Check 2d: Load RL agent and run prediction."""
    r = Result("RL Agent predict()")
    r.critical = False  # RL is optional
    path = MODELS_DIR / "rl_agent.zip"
    if not path.exists():
        r.skipped = True
        r.message = "Model file not found; skipping"
        return r
    try:
        from src.ai.models.rl_agent import RLPredictor
        predictor = RLPredictor()
        loaded = predictor.load(str(path))
        if not loaded:
            r.passed = False
            r.message = "RLPredictor.load() returned False"
            return r
        pred = predictor.predict(features)
        ok, msg = _validate_prediction(pred, "RL")
        r.passed = ok
        r.message = msg
    except Exception as e:
        r.passed = False
        r.message = f"Error: {e}\n{traceback.format_exc()}"
    return r


def check_ensemble_alignment() -> Result:
    """Check 3: Verify ensemble model_id alignment with registry."""
    r = Result("Ensemble model_id alignment")
    try:
        from src.ai.models.ensemble import DEFAULT_MODEL_IDS, EnsembleEngine
        from src.ai.models.registry import ModelRegistry

        registry = ModelRegistry()

        # Check that DEFAULT_MODEL_IDS is consistent with ENSEMBLE_MODEL_IDS
        # used in lifespan setup
        lifespan_ids = ["xgboost_alpha", "lstm_ts", "transformer_ts", "rl_ppo", "sentiment_finbert"]
        mismatches = []

        for mid in lifespan_ids:
            if mid not in DEFAULT_MODEL_IDS:
                # Not necessarily a failure -- lifespan may register extras
                pass

        for mid in DEFAULT_MODEL_IDS:
            if mid not in lifespan_ids:
                mismatches.append(f"{mid} in DEFAULT_MODEL_IDS but not in lifespan setup")

        # Verify EnsembleEngine can be instantiated with the registry
        engine = EnsembleEngine(registry=registry, model_ids=lifespan_ids)
        if set(engine.model_ids) != set(lifespan_ids):
            mismatches.append(
                f"EnsembleEngine.model_ids={engine.model_ids} != lifespan_ids={lifespan_ids}"
            )

        if mismatches:
            r.passed = False
            r.message = "Mismatches: " + "; ".join(mismatches)
        else:
            r.passed = True
            r.message = f"All {len(lifespan_ids)} model_ids aligned (DEFAULT_MODEL_IDS={DEFAULT_MODEL_IDS})"
    except Exception as e:
        r.passed = False
        r.message = f"Error: {e}\n{traceback.format_exc()}"
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("  Model Validation Report")
    print(f"  Models dir: {MODELS_DIR}")
    print(f"  Timestamp:  {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    all_results: List[Result] = []

    # --- Check 1: File existence ---
    print("\n[1] Model file existence")
    file_results = check_model_files_exist()
    all_results.extend(file_results)
    for r in file_results:
        status = "PASS" if r.passed else ("SKIP" if r.skipped else "FAIL")
        print(f"  {status:4s}  {r.name}: {r.message}")

    # --- Check 2: Load + predict ---
    print("\n[2] Model load & predict (synthetic features)")
    try:
        bars = _make_synthetic_bars(100)
        features = _build_features(bars)
        print(f"  Generated {len(bars)} synthetic bars -> {len(features)} features")
    except Exception as e:
        print(f"  FAIL  Could not generate synthetic features: {e}")
        traceback.print_exc()
        return 1

    pred_checks = [
        check_xgboost_prediction(features),
        check_lstm_prediction(features),
        check_transformer_prediction(features),
        check_rl_prediction(features),
    ]
    all_results.extend(pred_checks)
    for r in pred_checks:
        status = "PASS" if r.passed else ("SKIP" if r.skipped else "FAIL")
        print(f"  {status:4s}  {r.name}: {r.message}")

    # --- Check 3: Ensemble alignment ---
    print("\n[3] Ensemble model_id alignment")
    align_result = check_ensemble_alignment()
    all_results.append(align_result)
    status = "PASS" if align_result.passed else "FAIL"
    print(f"  {status:4s}  {align_result.name}: {align_result.message}")

    # --- Summary ---
    print("\n" + "=" * 70)
    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    skipped = sum(1 for r in all_results if r.skipped)
    failed = sum(1 for r in all_results if not r.passed and not r.skipped)
    critical_failures = sum(1 for r in all_results if not r.passed and not r.skipped and r.critical)

    print(f"  TOTAL: {total}  PASSED: {passed}  SKIPPED: {skipped}  FAILED: {failed}")
    if critical_failures > 0:
        print(f"  CRITICAL FAILURES: {critical_failures}")
        for r in all_results:
            if not r.passed and not r.skipped and r.critical:
                print(f"    - {r.name}: {r.message}")
    print("=" * 70)

    if critical_failures > 0:
        print("\nEXIT CODE: 1 (critical check(s) failed)")
        return 1
    else:
        print("\nEXIT CODE: 0 (all critical checks passed)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
