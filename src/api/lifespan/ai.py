"""
Lifespan: Model registry, ensemble engine, drift detector, meta-alpha,
self-learning scheduler.

Note: These are called from trading.py during autonomous loop setup since
the AI components are tightly coupled with strategy registration.
"""
import asyncio
import logging
import os
import sys

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def setup_ensemble_engine(app: FastAPI, alpha_model, feature_engine, models_dir: str):
    """Set up the AI Ensemble Engine: LSTM + Transformer + RL + Sentiment + XGBoost.
    Returns (ensemble_engine, model_registry, rl_predictor) or (None, None, None)."""
    _ensemble_engine = None
    _rl_predictor = None
    model_registry = None
    try:
        from src.ai.models.registry import ModelRegistry
        from src.ai.models.ensemble import EnsembleEngine
        from src.ai.models.lstm_predictor import LSTMPredictor
        from src.ai.models.transformer_predictor import TransformerPredictor
        from src.ai.models.rl_agent import RLPredictor
        from src.ai.models.sentiment_predictor import SentimentPredictor

        model_registry = ModelRegistry()

        # LSTM
        _lstm_path = os.path.join(models_dir, "lstm_predictor.pt")
        lstm_pred = LSTMPredictor(model_path=_lstm_path)
        model_registry.register(lstm_pred)
        logger.info("LSTM predictor registered (loaded=%s)", lstm_pred._loaded)

        # Transformer
        _tf_path = os.path.join(models_dir, "transformer_predictor.pt")
        tf_pred = TransformerPredictor(model_path=_tf_path)
        model_registry.register(tf_pred)
        logger.info("Transformer predictor registered (loaded=%s)", tf_pred._loaded)

        # RL Agent
        _rl_path = os.path.join(models_dir, "rl_agent.zip")
        _rl_predictor = RLPredictor(model_path=_rl_path)
        model_registry.register(_rl_predictor)
        logger.info("RL predictor registered (loaded=%s)", _rl_predictor._loaded)

        # Sentiment
        sentiment_pred = SentimentPredictor()
        model_registry.register(sentiment_pred)
        logger.info("Sentiment predictor registered")

        # Volatility Predictor (vol regime detection + confidence scaling)
        try:
            from src.ai.models.vol_predictor import VolPredictor
            vol_pred = VolPredictor()
            model_registry.register(vol_pred)
            logger.info("Volatility predictor registered (vol_pred)")
        except Exception as e:
            logger.debug("VolPredictor not registered: %s", e)

        # NOTE: XGBoost predictor is wired AFTER AlphaModel.load() in trading.py
        # to avoid joblib/torch deadlock. See trading.py init_trading().

        # ── Model Validation Gate (P0-2): disable models that fail validation ──
        try:
            from src.ai.model_validation import ModelValidator
            import numpy as np
            _models_dir_abs = os.path.abspath(models_dir)
            _validation_dir = os.path.join(_models_dir_abs, "validation")
            _validator = ModelValidator()
            _disabled_models = []
            for _mid in list(model_registry.list_models()):
                _oos_path = os.path.join(_validation_dir, f"{_mid}_oos.npz")
                if os.path.exists(_oos_path):
                    try:
                        _oos_data = np.load(_oos_path)
                        _val_result = _validator.validate(
                            _oos_data.get("predictions", np.array([])),
                            _oos_data.get("actuals", np.array([])),
                        )
                        if not _val_result.passed:
                            _pred_obj = model_registry.get(_mid)
                            if _pred_obj and hasattr(_pred_obj, "_model"):
                                _pred_obj._model = None
                            elif _pred_obj and hasattr(_pred_obj, "_loaded"):
                                _pred_obj._loaded = False
                            _disabled_models.append(_mid)
                            logger.critical(
                                "MODEL VALIDATION FAILED for %s — DISABLED. %s",
                                _mid, getattr(_val_result, "summary", ""),
                            )
                        else:
                            logger.info("Model %s validated OK: %s", _mid, getattr(_val_result, "summary", "passed"))
                    except Exception as _ve:
                        logger.warning("Model validation error for %s: %s (model allowed)", _mid, _ve)
                else:
                    logger.debug("No OOS validation data for %s — model allowed without validation", _mid)
            if _disabled_models:
                logger.warning("Models disabled by validation gate: %s", _disabled_models)
        except ImportError:
            logger.debug("ModelValidator not available — skipping validation gate")
        except Exception as _gate_err:
            logger.warning("Model validation gate error: %s — all models allowed", _gate_err)

        # RL agent: set weight=0 if model not loaded (Sprint 8.10)
        _rl_weight = 0.15
        if _rl_predictor and not _rl_predictor._loaded:
            _rl_weight = 0.0
            logger.warning("RL model not loaded — setting ensemble weight to 0")

        _ensemble_weights = {
            "xgboost_alpha": 0.25,
            "lstm_ts": 0.22,
            "transformer_ts": 0.18,
            "rl_ppo": _rl_weight,
            "sentiment_finbert": 0.20 if _rl_weight == 0 else 0.08,
            "vol_pred": 0.05 if _rl_weight == 0 else 0.05,
        }

        # Normalize weights to sum to 1.0
        _total_w = sum(_ensemble_weights.values())
        if _total_w > 0:
            _ensemble_weights = {k: v / _total_w for k, v in _ensemble_weights.items()}

        _ensemble_engine = EnsembleEngine(
            registry=model_registry,
            model_ids=["xgboost_alpha", "lstm_ts", "transformer_ts", "rl_ppo", "sentiment_finbert", "vol_pred"],
            weights=_ensemble_weights,
        )
        app.state.ensemble_engine = _ensemble_engine
        app.state.model_registry = model_registry
        logger.info("AI Ensemble Engine configured (6 models incl. vol_pred, rl_weight=%.2f)", _rl_weight)
    except Exception as e:
        logger.warning("AI Ensemble Engine not configured: %s", e)

    # ── Feature Distribution Shift Detector (Sprint 11.1) ──
    try:
        from src.ai.feature_shift_detector import FeatureShiftDetector
        _shift_detector = FeatureShiftDetector(
            min_samples=100,
            alert_callback=lambda report: logger.critical(
                "FEATURE SHIFT ALERT: %s — recommendation=%s, shifted=%d/%d features",
                report.max_psi_feature, report.recommendation,
                report.features_shifted, report.total_features,
            ),
        )
        app.state.feature_shift_detector = _shift_detector
        logger.info("Feature shift detector initialized")
    except Exception as e:
        logger.debug("Feature shift detector not initialized: %s", e)

    return _ensemble_engine, model_registry, _rl_predictor


def setup_drift_detector(app: FastAPI):
    """Set up MultiLayerDriftDetector and daily drift check task.
    Returns the drift_detector instance or None."""
    try:
        from src.ai.drift.multi_drift import MultiLayerDriftDetector
        _drift_detector = MultiLayerDriftDetector()
        app.state.drift_detector = _drift_detector
        logger.info("MultiLayerDriftDetector initialized")

        async def _daily_drift_check():
            """Run drift detection daily at ~15:45 IST (after market close)."""
            from datetime import datetime as _dt, timezone as _tz, timedelta as _td
            _IST = _tz(_td(hours=5, minutes=30))
            while True:
                await asyncio.sleep(300)  # check every 5 min
                try:
                    now_ist = _dt.now(_IST)
                    # Only run at 15:45
                    if now_ist.hour != 15 or now_ist.minute < 45 or now_ist.minute > 50:
                        continue
                    if now_ist.weekday() >= 5:
                        continue

                    dd = getattr(app.state, "drift_detector", None)
                    if dd is None:
                        continue

                    signals = dd.check_all()
                    drifted_layers = [s for s in signals if s.drifted]
                    if len(drifted_layers) >= 2:
                        logger.warning("Model drift detected: %d layers flagged — %s",
                                       len(drifted_layers), [(s.drift_type.value, s.value) for s in drifted_layers])
                        _an = getattr(app.state, "alert_notifier", None)
                        if _an:
                            from src.alerts.notifier import AlertSeverity
                            await _an.send(AlertSeverity.CRITICAL, "Model Drift Detected",
                                           f"{len(drifted_layers)} drift layers flagged: {[s.drift_type.value for s in drifted_layers]}",
                                           source="drift_detector")
                        from src.api.ws_manager import get_ws_manager
                        mgr = get_ws_manager()
                        if mgr:
                            await mgr.broadcast({
                                "type": "model_drift_detected",
                                "layers": {s.drift_type.value: {"drifted": s.drifted, "value": s.value, "threshold": s.threshold} for s in signals},
                            })
                    elif drifted_layers:
                        logger.info("Drift check: 1 layer flagged (warning only) — %s", drifted_layers[0].drift_type.value)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.debug("Drift check error: %s", e)

        _drift_task = asyncio.get_running_loop().create_task(_daily_drift_check())
        app.state._drift_task = _drift_task
        logger.info("Drift detection scheduler active (daily check at 15:45 IST)")
        return _drift_detector
    except Exception as e:
        logger.debug("Drift detection not configured: %s", e)
        return None


def setup_self_learning(app: FastAPI, ensemble_engine, alpha_model, model_path, models_dir, feature_engine, bar_cache):
    """Set up SelfLearningScheduler with drift->retrain->promote pipeline.

    Pipeline (daily at 15:45 IST after NSE close):
      1. Multi-layer drift detection (feature, PSI, IC degradation).
      2. If >=2 layers drifted -> trigger retrain for all models.
      3. If not drifted -> IC weight update for ensemble.
      4. Weekly (Friday): forced walk-forward revalidation regardless of drift.
      5. Auto-promote models that pass stability checks.

    Graceful: if self-learning modules fail to import, log warning and skip.
    """
    # ── Lazy imports with graceful fallback ──
    try:
        from src.ai.self_learning.scheduler import SelfLearningScheduler
    except ImportError:
        logger.warning("SelfLearningScheduler not available (import failed) — skipping self-learning setup")
        app.state.self_learning_scheduler = None
        return

    try:
        from src.ai.self_learning.drift import ConceptDriftDetector, DataDistributionMonitor
    except ImportError:
        logger.warning("self_learning.drift not available — skipping self-learning setup")
        app.state.self_learning_scheduler = None
        return

    # Optional: orchestrator for structured retrain-all
    _SelfLearningOrchestrator = None
    try:
        from src.ai.self_learning.orchestrator import SelfLearningOrchestrator
        _SelfLearningOrchestrator = SelfLearningOrchestrator
    except ImportError:
        logger.debug("SelfLearningOrchestrator not available — using subprocess retrain fallback")

    # Optional: retrain pipeline for walk-forward promotion
    _RetrainPipeline = None
    _RetrainConfig = None
    try:
        from src.ai.self_learning.retrain import RetrainPipeline, RetrainConfig
        _RetrainPipeline = RetrainPipeline
        _RetrainConfig = RetrainConfig
    except ImportError:
        logger.debug("RetrainPipeline not available — walk-forward promotion disabled")

    try:
        _sl_drift = ConceptDriftDetector(threshold=0.3)
        _sl_dist_monitor = DataDistributionMonitor(window=100)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        # ── Build retrain function: orchestrator-based or subprocess fallback ──
        def _hot_reload_all_models():
            """Hot-reload all model files after retrain."""
            reloaded = []
            if alpha_model and os.path.exists(model_path):
                try:
                    alpha_model.load(model_path)
                    reloaded.append("xgboost_alpha")
                except Exception as e:
                    logger.warning("XGBoost hot-reload failed: %s", e)
            _mr = getattr(app.state, "model_registry", None)
            if _mr:
                for _mid, _suffix in [
                    ("lstm_ts", "lstm_predictor.pt"),
                    ("transformer_ts", "transformer_predictor.pt"),
                    ("rl_ppo", "rl_agent.zip"),
                ]:
                    _mp = os.path.join(models_dir, _suffix)
                    if os.path.exists(_mp):
                        try:
                            _pred = _mr.get(_mid)
                            if _pred and hasattr(_pred, 'load'):
                                _pred.load(_mp)
                                reloaded.append(_mid)
                                logger.info("SL hot-reloaded model: %s", _mid)
                        except Exception:
                            pass
            return reloaded

        def _on_model_promote(model_id: str, version: str, metrics: dict):
            """Callback when a model is auto-promoted: hot-load + broadcast."""
            logger.warning("MODEL PROMOTED: %s -> %s (metrics=%s)", model_id, version, metrics)
            _hot_reload_all_models()
            # Broadcast promotion event via WebSocket
            try:
                import asyncio as _aio
                from src.api.ws_manager import get_ws_manager
                mgr = get_ws_manager()
                if mgr:
                    _loop = _aio.get_event_loop()
                    if _loop.is_running():
                        _aio.ensure_future(mgr.broadcast({
                            "type": "model_promoted",
                            "model_id": model_id,
                            "version": version,
                            "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
                        }))
            except Exception:
                pass

        def _sl_retrain_fn():
            """Retrain all models via subprocess, then hot-reload."""
            import subprocess, sys as _sys
            env = os.environ.copy()
            env["PYTHONPATH"] = project_root
            train_script = os.path.join(project_root, "scripts", "auto_train_all.py")
            if not os.path.exists(train_script):
                train_script = os.path.join(project_root, "scripts", "train_alpha_model.py")
            if not os.path.exists(train_script):
                logger.warning("No training script found — retrain skipped")
                return {"all_models": False, "reason": "no_train_script"}
            result = subprocess.run(
                [_sys.executable, train_script, "--quick"],
                env=env, capture_output=True, text=True, timeout=1800,
                cwd=project_root,
            )
            if result.returncode == 0:
                reloaded = _hot_reload_all_models()
                logger.info("SL retrain complete, hot-reloaded: %s", reloaded)
                return {"all_models": True, "reloaded": reloaded}
            logger.warning("SL retrain subprocess failed (rc=%d): %s",
                           result.returncode, (result.stderr or "")[-300:])
            return {"all_models": False, "returncode": result.returncode}

        # ── IC update function ──
        _sl_ic_fn = None
        if ensemble_engine and hasattr(ensemble_engine, 'update_weights_from_ic'):
            _sl_ic_fn = ensemble_engine.update_weights_from_ic

        # ── Alert function ──
        _sl_alert_fn = None
        _an_sl = getattr(app.state, "alert_notifier", None)
        if _an_sl:
            _sl_alert_fn = _an_sl.send

        # ── Recent features gatherer ──
        def _get_recent_features():
            """Gather recent feature snapshots from bar cache for drift detection."""
            feats = []
            try:
                from src.core.events import Exchange
                syms = bar_cache.symbols_with_bars(Exchange.NSE, "1m", min_bars=30)
                for sym in syms[:5]:
                    bars_list = bar_cache.get_bars(sym, Exchange.NSE, "1m", 50)
                    if bars_list and len(bars_list) >= 20:
                        f = feature_engine.build_features(bars_list)
                        if f:
                            feats.append(f)
            except Exception:
                pass
            return feats

        # ── Create and start the scheduler ──
        _self_learning_scheduler = SelfLearningScheduler(
            drift_detector=_sl_drift,
            distribution_monitor=_sl_dist_monitor,
            retrain_fn=_sl_retrain_fn,
            ic_update_fn=_sl_ic_fn,
            alert_fn=_sl_alert_fn,
            get_recent_features_fn=_get_recent_features,
            post_market_hour=15,
            post_market_minute=45,
            min_drift_layers_for_retrain=2,
            weekly_revalidation_day=4,  # Friday = 4 (Monday=0)
        )

        # Wire ensemble for calibrator fitting (Sprint 8.4)
        _self_learning_scheduler._ensemble = getattr(app.state, "ensemble_engine", None)
        _self_learning_scheduler.start()
        app.state.self_learning_scheduler = _self_learning_scheduler

        # ── Store drift/distribution monitors on app.state for API access ──
        app.state.sl_drift_detector = _sl_drift
        app.state.sl_distribution_monitor = _sl_dist_monitor

        # ── Wire SelfLearningOrchestrator if available (structured retrain-all) ──
        if _SelfLearningOrchestrator is not None:
            try:
                _orchestrator = _SelfLearningOrchestrator(
                    drift_detector=_sl_drift,
                    distribution_monitor=_sl_dist_monitor,
                    retrain_pipelines=[],  # Pipelines added below if RetrainPipeline available
                    on_retrain_complete=lambda mid, replaced: logger.info(
                        "Orchestrator retrain %s: replaced=%s", mid, replaced
                    ),
                )
                app.state.self_learning_orchestrator = _orchestrator
                logger.info("SelfLearningOrchestrator wired (structured drift→retrain)")
            except Exception as e:
                logger.debug("SelfLearningOrchestrator init failed: %s", e)

        # ── Walk-forward revalidation task (Friday 17:00 IST) ──
        async def _weekly_walk_forward_loop():
            """Forced walk-forward revalidation every Friday at 17:00 IST."""
            from datetime import datetime as _dt, timezone as _tz, timedelta as _td
            _IST = _tz(_td(hours=5, minutes=30))
            _last_wf_date = None
            while True:
                await asyncio.sleep(300)  # check every 5 min
                try:
                    now_ist = _dt.now(_IST)
                    today = now_ist.strftime("%Y-%m-%d")
                    # Friday = 4, run at 17:00-17:05 IST
                    if now_ist.weekday() != 4:
                        continue
                    if now_ist.hour != 17 or now_ist.minute > 5:
                        continue
                    if _last_wf_date == today:
                        continue
                    _last_wf_date = today
                    logger.info("Weekly walk-forward revalidation START (Friday %s)", today)

                    sls = getattr(app.state, "self_learning_scheduler", None)
                    if sls is not None:
                        try:
                            result = await sls.run_now()
                            logger.info(
                                "Weekly walk-forward revalidation DONE: action=%s, elapsed=%.1fs",
                                result.get("action", "unknown"),
                                result.get("elapsed_seconds", 0),
                            )
                            # Broadcast to WebSocket
                            from src.api.ws_manager import get_ws_manager
                            mgr = get_ws_manager()
                            if mgr:
                                await mgr.broadcast({
                                    "type": "self_learning_weekly_revalidation",
                                    "date": today,
                                    "action": result.get("action", "unknown"),
                                    "drift_layers_fired": result.get("drift_layers_fired", 0),
                                    "elapsed_seconds": result.get("elapsed_seconds", 0),
                                })
                        except Exception as e:
                            logger.warning("Weekly walk-forward run_now failed: %s", e)
                    else:
                        logger.debug("Weekly walk-forward: no scheduler available")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning("Weekly walk-forward error: %s", e)

        _wf_task = asyncio.get_running_loop().create_task(_weekly_walk_forward_loop())
        app.state._walk_forward_task = _wf_task

        logger.info(
            "SelfLearningScheduler started: "
            "post-market 15:45 IST daily (drift->retrain/IC-update), "
            "walk-forward Friday 17:00 IST, "
            "orchestrator=%s",
            "enabled" if _SelfLearningOrchestrator else "disabled",
        )
    except Exception as e:
        logger.warning("SelfLearningScheduler not started: %s", e)
        app.state.self_learning_scheduler = None


def setup_auto_retrain(app: FastAPI, alpha_model, model_path, models_dir):
    """Set up auto-retrain scheduler (checks every 6 hours, retrains if stale >7 days)."""
    async def _auto_retrain_loop():
        """Check model age every 6 hours; retrain all models if >7 days old."""
        import json, subprocess, time as _time
        retrain_interval = 6 * 3600  # check every 6 hours
        model_max_age = 7 * 86400     # 7 days
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        while True:
            await asyncio.sleep(retrain_interval)
            try:
                # Check training_meta.json (from auto_train_all.py) or XGBoost meta
                meta_path = os.path.join(project_root, "models", "training_meta.json")
                xgb_meta_path = os.path.join(project_root, "models", "alpha_xgb_meta.json")
                should_retrain = False

                if os.path.exists(meta_path):
                    age = _time.time() - os.path.getmtime(meta_path)
                    if age > model_max_age:
                        should_retrain = True
                        logger.info("Auto-retrain: models are %.1f days old (limit=7), triggering full retrain", age / 86400)
                elif os.path.exists(xgb_meta_path):
                    age = _time.time() - os.path.getmtime(xgb_meta_path)
                    if age > model_max_age:
                        should_retrain = True
                        logger.info("Auto-retrain: XGBoost model is %.1f days old, triggering retrain", age / 86400)
                else:
                    should_retrain = True
                    logger.info("Auto-retrain: no model metadata found, triggering training")

                if should_retrain:
                    logger.info("Auto-retrain starting (all AI models)...")
                    env = os.environ.copy()
                    env["PYTHONPATH"] = project_root
                    # Use auto_train_all.py with --quick for scheduled retrains
                    train_script = os.path.join(project_root, "scripts", "auto_train_all.py")
                    if not os.path.exists(train_script):
                        train_script = os.path.join(project_root, "scripts", "train_alpha_model.py")
                    result = await asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: subprocess.run(
                            [sys.executable, train_script, "--quick"],
                            env=env, capture_output=True, text=True, timeout=1800,
                            cwd=project_root,
                        )
                    )
                    if result.returncode == 0:
                        logger.info("Auto-retrain completed successfully")
                        # Hot-reload ALL models (Sprint 8.9)
                        if alpha_model.load(model_path):
                            logger.info("XGBoost model hot-reloaded")
                        _mr = getattr(app.state, "model_registry", None)
                        if _mr:
                            for _mid, _suffix in [("lstm_ts", "lstm_predictor.pt"), ("transformer_ts", "transformer_predictor.pt"), ("rl_ppo", "rl_agent.zip")]:
                                _mp = os.path.join(models_dir, _suffix)
                                if os.path.exists(_mp):
                                    try:
                                        _pred = _mr.get(_mid)
                                        if _pred and hasattr(_pred, 'load') and hasattr(_pred, '_loaded'):
                                            if _pred.load(_mp):
                                                _pred._loaded = True
                                                logger.info("Hot-reloaded model: %s from %s", _mid, _suffix)
                                    except Exception as _re:
                                        logger.warning("Hot-reload failed for %s: %s", _mid, _re)
                        # Broadcast retrain event
                        try:
                            ws_mgr = getattr(app.state, "ws_manager", None)
                            if ws_mgr:
                                await ws_mgr.broadcast({"type": "models_retrained", "status": "success"})
                        except Exception:
                            pass
                    else:
                        logger.warning("Auto-retrain failed: %s", result.stderr[-500:] if result.stderr else "unknown error")
            except Exception as e:
                logger.exception("Auto-retrain error: %s", e)

    _retrain_task = asyncio.get_running_loop().create_task(_auto_retrain_loop())
    app.state._retrain_task = _retrain_task
    logger.info("Auto-retrain scheduler active (checks every 6h, retrains all AI models if >7 days old)")


async def init_ai(app: FastAPI) -> None:
    """Placeholder — AI initialization is done inside trading.py's init_trading
    because AI components are tightly coupled with strategy registration and
    the autonomous loop setup. This function exists for interface consistency."""
    pass
