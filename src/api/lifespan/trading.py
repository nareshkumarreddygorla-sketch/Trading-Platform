"""
Lifespan: Autonomous loop, strategies, agents, background tasks
(auto-retrain, drift detection, self-learning, daily report).
"""

import asyncio
import logging
import os
import time
from datetime import UTC

from fastapi import FastAPI

from .ai import (
    setup_auto_retrain,
    setup_drift_detector,
    setup_ensemble_engine,
    setup_self_learning,
)

logger = logging.getLogger(__name__)


async def init_trading(app: FastAPI) -> None:
    """Initialize autonomous trading loop: strategies, AI models, agents, schedulers."""

    _autonomous_loop = None
    if getattr(app.state, "order_entry_service", None) is not None:
        try:
            from src.ai.alpha_model import AlphaModel, AlphaStrategy
            from src.ai.feature_engine import FeatureEngine
            from src.ai.regime.classifier import RegimeClassifier
            from src.api.routers.strategies import get_registry
            from src.core.events import Exchange
            from src.execution.autonomous_loop import AutonomousLoop
            from src.strategy_engine.allocator import AllocatorConfig, PortfolioAllocator
            from src.strategy_engine.runner import StrategyRunner

            # ── AI Ensemble Engine: create BEFORE joblib.load to avoid torch/joblib deadlock ──
            _models_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models"
            )
            _ensemble_engine, model_registry, _rl_predictor = setup_ensemble_engine(app, None, None, _models_dir)

            # Load trained XGBoost model if available (uses joblib.load — must happen AFTER
            # setup_ensemble_engine to avoid torch load_state_dict deadlock on macOS/Py3.9)
            _model_path = os.path.join(_models_dir, "alpha_xgb.joblib")
            _alpha_model = AlphaModel(strategy_id="ai_alpha")
            if os.path.exists(_model_path):
                if _alpha_model.load(_model_path):
                    logger.info("Loaded trained AI model from %s", _model_path)
                else:
                    logger.warning("Failed to load AI model from %s — using fallback heuristic", _model_path)
            else:
                logger.info(
                    "No trained AI model found at %s — using fallback heuristic. Run: PYTHONPATH=. python scripts/train_alpha_model.py",
                    _model_path,
                )

            def _get_safe_mode() -> bool:
                return bool(getattr(app.state, "safe_mode", True))

            bar_cache = app.state.bar_cache
            registry = get_registry()

            # ── Register ALL strategies (multi-strategy ensemble) ──
            # 1. AI Alpha (XGBoost + market context)
            try:
                registry.register(AlphaStrategy(alpha_model=_alpha_model))
                logger.info("Strategy registered: ai_alpha (XGBoost)")
            except Exception:
                pass

            # 2. Classical technical strategies
            try:
                from src.strategy_engine.classical import (
                    EMACrossoverStrategy,
                    MACDStrategy,
                    RSIStrategy,
                )

                registry.register(EMACrossoverStrategy(fast=9, slow=21))
                registry.register(MACDStrategy(fast=12, slow=26, signal=9))
                registry.register(RSIStrategy(period=14, oversold=30.0, overbought=70.0))
                logger.info("Strategies registered: ema_crossover, macd, rsi")
            except Exception as e:
                logger.warning("Classical strategies not registered: %s", e)

            # 3. Momentum Breakout (for trending markets)
            try:
                from src.strategy_engine.momentum_breakout import MomentumBreakoutStrategy

                registry.register(MomentumBreakoutStrategy())
                logger.info("Strategy registered: momentum_breakout")
            except Exception as e:
                logger.debug("MomentumBreakout not available: %s", e)

            # 4. Mean Reversion (for sideways markets)
            try:
                from src.strategy_engine.mean_reversion import MeanReversionStrategy

                registry.register(MeanReversionStrategy())
                logger.info("Strategy registered: mean_reversion")
            except Exception as e:
                logger.debug("MeanReversion not available: %s", e)

            logger.info("Total strategies registered: %d → %s", len(registry.list_all()), registry.list_all())

            # ── Wire strategy registry disable into existing PerformanceTracker (Sprint 10.8: single instance) ──
            _performance_tracker = getattr(app.state, "performance_tracker", None)
            if _performance_tracker is not None:
                try:
                    # Wire registry.disable into the existing tracker's strategy disabled callback
                    _orig_disabled_cb = getattr(_performance_tracker, "_on_strategy_disabled", None)

                    def _enhanced_strategy_disabled(strategy_id: str, reason: str):
                        logger.warning(
                            "PerformanceTracker AUTO-DISABLED strategy '%s' (reason=%s)", strategy_id, reason
                        )
                        registry.disable(strategy_id)
                        # Also fire the original WS broadcast callback
                        if _orig_disabled_cb:
                            try:
                                _orig_disabled_cb(strategy_id, reason)
                            except Exception:
                                pass

                    _performance_tracker._on_strategy_disabled = _enhanced_strategy_disabled
                    # Update config thresholds
                    _performance_tracker.max_consecutive_losses_disable = 5
                    _performance_tracker.min_win_rate_disable = 0.35
                    _performance_tracker.max_drawdown_pct_disable = 15.0
                    logger.info("PerformanceTracker enhanced with strategy registry disable (single instance)")
                except Exception as e:
                    logger.warning("PerformanceTracker registry wiring failed: %s", e)

            strategy_runner = StrategyRunner(registry)
            feature_engine = FeatureEngine()
            regime_classifier = RegimeClassifier()
            app.state.regime_classifier = regime_classifier

            # P0-3: Load feature normalizer (z-score normalization for model safety)
            from src.ai.feature_engine import FeatureNormalizer

            _normalizer_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "models",
                "feature_normalizer.json",
            )
            _feature_normalizer = FeatureNormalizer.load(_normalizer_path)
            app.state.feature_normalizer = _feature_normalizer

            # Enhanced allocator: more concurrent signals, strategy-level caps
            allocator = PortfolioAllocator(
                AllocatorConfig(
                    max_active_signals=10,  # allow 10 concurrent positions
                    max_capital_pct_per_signal=5.0,  # 5% per position (aligned with risk limit)
                    min_confidence=0.2,
                    strategy_cap_pct={
                        "ai_alpha": 5.0,  # Aligned with max_position_pct
                        "ema_crossover": 5.0,
                        "macd": 5.0,
                        "rsi": 5.0,
                        "momentum_breakout": 5.0,
                        "mean_reversion": 5.0,
                        "ml_predictor": 5.0,  # ML ensemble (LSTM+Transformer+XGB)
                        "rl_agent": 5.0,  # Reinforcement learning agent
                    },
                )
            )

            def _get_market_feed_healthy() -> bool:
                svc = getattr(app.state, "market_data_service", None)
                if svc is not None:
                    return svc.is_healthy()
                # No market data service — check bar cache freshness instead
                bc = getattr(app.state, "bar_cache", None)
                if bc is not None:
                    last_ts = getattr(bc, "last_bar_timestamp", lambda: None)()
                    if last_ts and (time.time() - last_ts) > 120:
                        return False  # No bars received in last 2 minutes
                return True  # No cache = paper mode, OK

            def _get_bar_ts() -> str:
                ts = bar_cache.get_current_bar_ts()
                if ts:
                    return ts
                from datetime import datetime

                return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

            def _get_bars(symbol: str, exchange: Exchange, interval: str, n: int):
                return bar_cache.get_bars(symbol, exchange, interval, n)

            def _get_symbols():
                syms = bar_cache.symbols_with_bars(Exchange.NSE, "1m", min_bars=20)
                return [(s, Exchange.NSE) for s in syms]

            def _get_risk_state():
                rm = getattr(app.state, "risk_manager", None)
                if rm is None:
                    return {"equity": 0, "exposure_multiplier": 1.0, "max_position_pct": 5.0}
                cb = getattr(app.state, "circuit_breaker", None)
                peak = getattr(cb, "_peak_equity", rm.equity) if cb else rm.equity
                dd = (peak - rm.equity) / peak * 100 if peak > 0 else 0
                drawdown_scale = 0.5 if dd > 5 else (0.75 if dd > 3 else 1.0)

                # Regime-aware scaling: reduce exposure in crisis/high-vol
                regime_scale = 1.0
                try:
                    import numpy as np

                    recent_returns = np.array([0.0])
                    vol = getattr(rm, "_current_vol", 0.01)
                    regime_result = regime_classifier.classify(recent_returns, vol, 0.0)
                    regime_label = regime_result.label.value
                    if regime_label == "crisis":
                        regime_scale = 0.0
                    elif regime_label == "high_volatility":
                        regime_scale = 0.5
                except Exception:
                    pass

                # Vol targeting: scale positions by target_vol / realized_vol
                vol_scale = 1.0
                _vt = getattr(app.state, "vol_targeter", None)
                if _vt:
                    vol_scale = _vt.get_scale_factor()

                # Tail risk: reduce exposure based on VIX level
                tail_risk_scale = 1.0
                _trp = getattr(app.state, "tail_risk_protector", None)
                if _trp and hasattr(_trp, "state"):
                    tail_risk_scale = getattr(_trp.state, "exposure_scale", 1.0)

                # Composite exposure multiplier
                base_mult = getattr(rm, "_exposure_multiplier", 1.0)
                composite_mult = base_mult * vol_scale * tail_risk_scale

                return {
                    "equity": rm.equity,
                    "daily_pnl": rm.daily_pnl,
                    "exposure_multiplier": composite_mult,
                    "max_position_pct": rm.limits.max_position_pct,
                    "drawdown_scale": drawdown_scale,
                    "regime_scale": regime_scale,
                    "vol_scale": vol_scale,
                    "tail_risk_scale": tail_risk_scale,
                }

            def _get_positions():
                rm = getattr(app.state, "risk_manager", None)
                return list(rm.positions) if rm else []

            async def _submit_order_async(req):
                return await app.state.order_entry_service.submit_order(req)

            # Full-market scanner: scores all NSE stocks via AlphaModel
            _market_scanner = None
            try:
                from src.scanner.market_scanner import MarketScanner

                _market_scanner = MarketScanner(
                    alpha_model=_alpha_model,
                    feature_engine=feature_engine,
                    top_n=20,
                    min_confidence=0.55,
                )
                logger.info("MarketScanner configured (full NSE universe)")
            except Exception as e:
                logger.warning("MarketScanner not configured: %s", e)

            # ── Wire XGBoost into ensemble (after AlphaModel loaded via joblib) ──
            if _alpha_model and model_registry:
                try:
                    from src.ai.models.base import BasePredictor, PredictionOutput

                    class XGBoostPredictor(BasePredictor):
                        model_id = "xgboost_alpha"
                        version = "v1"

                        def __init__(self_xgb, alpha_model):
                            self_xgb._alpha = alpha_model
                            self_xgb.path = ""

                        def predict(self_xgb, features, context=None):
                            try:
                                score = self_xgb._alpha.score(features)
                                prob_up = 0.5 + score * 0.3
                                return PredictionOutput(
                                    prob_up=min(0.95, max(0.05, prob_up)),
                                    expected_return=score * 0.01,
                                    confidence=min(1.0, abs(score)),
                                    model_id="xgboost_alpha",
                                    version="v1",
                                    metadata={"raw_score": score},
                                )
                            except Exception:
                                return PredictionOutput(0.5, 0.0, 0.0, "xgboost_alpha", "v1", {})

                    xgb_pred = XGBoostPredictor(_alpha_model)
                    model_registry.register(xgb_pred)
                    logger.info("XGBoost predictor wired into ensemble (post-joblib)")
                except Exception as e:
                    logger.debug("XGBoost predictor wiring failed: %s", e)

            # ── Register ML strategies (connect to real models) ──
            try:
                from src.strategy_engine.ml_strategies import MLPredictorStrategy, RLAgentStrategy

                ml_strategy = MLPredictorStrategy(
                    ensemble_engine=_ensemble_engine,
                    feature_engine=feature_engine,
                    confidence_threshold=0.55,
                    prob_threshold=0.58,
                )
                registry.register(ml_strategy)
                logger.info("Strategy registered: ml_predictor (ensemble AI)")

                rl_strategy = RLAgentStrategy(
                    rl_predictor=_rl_predictor,
                    feature_engine=feature_engine,
                    confidence_threshold=0.5,
                )
                registry.register(rl_strategy)
                logger.info("Strategy registered: rl_agent (PPO)")
            except Exception as e:
                logger.warning("ML strategies not registered: %s", e)

            async def _ws_broadcast(message):
                from src.api.ws_manager import get_ws_manager

                mgr = get_ws_manager()
                if mgr:
                    await mgr.broadcast(message)

            def _on_daily_reset():
                rm = getattr(app.state, "risk_manager", None)
                if rm and hasattr(rm, "reset_daily_pnl"):
                    rm.reset_daily_pnl()

            # GAP-5 fix: wire drift_gate and regime_gate into autonomous loop
            _drift_detector = getattr(app.state, "drift_detector", None)

            def _drift_gate():
                if _drift_detector is None:
                    return True
                try:
                    return not getattr(_drift_detector, "is_drifted", lambda: False)()
                except Exception:
                    return True  # Fail open on error

            _regime_cls = regime_classifier

            def _regime_gate():
                if _regime_cls is None:
                    return True
                try:
                    regime = getattr(_regime_cls, "current_regime", None)
                    if regime is not None and hasattr(regime, "value"):
                        return regime.value != "CRISIS"
                    return True
                except Exception:
                    return True

            gateway = getattr(app.state, "gateway", None)

            _autonomous_loop = AutonomousLoop(
                _submit_order_async,
                get_safe_mode=_get_safe_mode,
                get_bar_ts=_get_bar_ts,
                get_bars=_get_bars,
                get_symbols=_get_symbols,
                strategy_runner=strategy_runner,
                allocator=allocator,
                get_risk_state=_get_risk_state,
                get_positions=_get_positions,
                get_market_feed_healthy=_get_market_feed_healthy,
                feature_engine=feature_engine,
                regime_classifier=regime_classifier,
                market_scanner=_market_scanner,
                performance_tracker=_performance_tracker,
                ws_broadcast=_ws_broadcast,
                on_daily_reset=_on_daily_reset,
                drift_gate=_drift_gate,
                regime_gate=_regime_gate,
                poll_interval_seconds=60.0,
                paper_mode=getattr(gateway, "paper", True),
            )
            # P0-3: Wire feature normalizer into autonomous loop
            if _feature_normalizer and _feature_normalizer._fitted:
                _autonomous_loop.set_feature_normalizer(_feature_normalizer)
                logger.info(
                    "Feature normalizer wired to autonomous loop (%d features)", len(_feature_normalizer._means)
                )

            # Wire open trade persistence (write-ahead for SL/TP tracking)
            # Try PostgreSQL first (if DATABASE_URL set), fall back to SQLite TradeStore
            _trade_repo_wired = False
            if os.environ.get("DATABASE_URL"):
                try:
                    from src.persistence.open_trade_repo import OpenTradeRepository

                    _open_trade_repo = OpenTradeRepository()
                    _autonomous_loop.set_open_trade_repo(_open_trade_repo)
                    recovered = _autonomous_loop.load_open_trades_from_db()
                    logger.info("Open trade persistence wired via PostgreSQL (recovered %d trades from DB)", recovered)
                    _trade_repo_wired = True
                except Exception as e:
                    logger.warning("PostgreSQL open trade persistence not available: %s", e)

            if not _trade_repo_wired:
                try:
                    from src.persistence.trade_store import TradeStore

                    _db_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                        "data",
                        "trades.db",
                    )
                    _trade_store = TradeStore(db_path=_db_path)
                    _autonomous_loop.set_open_trade_repo(_trade_store)
                    recovered = _autonomous_loop.load_open_trades_from_db()
                    app.state.trade_store = _trade_store
                    logger.info("Open trade persistence wired via SQLite (recovered %d trades from DB)", recovered)
                except Exception as e:
                    logger.warning("SQLite trade persistence not available: %s", e)

            # ── Wire Trade Outcome Repository (self-learning feedback loop) ──
            try:
                from src.persistence.trade_outcome_repo import TradeOutcomeRepository

                _trade_outcome_repo = TradeOutcomeRepository()
                _autonomous_loop.set_trade_outcome_repo(_trade_outcome_repo)
                app.state.trade_outcome_repo = _trade_outcome_repo
                logger.info("Trade outcome repository wired to autonomous loop (self-learning feedback)")
            except Exception as e:
                logger.warning("Trade outcome repository not wired: %s", e)

            _autonomous_loop.start()
            app.state.autonomous_loop = _autonomous_loop
            logger.info("Autonomous loop started (multi-strategy→regime→scanner→allocator→risk→execution)")

            # ── Wire kill switch into autonomous loop (Sprint 7.1) ──
            _ks_for_loop = getattr(app.state, "kill_switch", None)
            if _ks_for_loop is not None:

                async def _kill_switch_check():
                    ks = getattr(app.state, "kill_switch", None)
                    if ks is None:
                        return False
                    return await ks.is_armed()

                _autonomous_loop.set_kill_switch(_kill_switch_check)
                logger.info("Kill switch wired to autonomous loop (auto-close on arm)")

            # ── Wire risk_manager for forced close symbols (Sprint 7.9) ──
            _autonomous_loop._risk_manager = getattr(app.state, "risk_manager", None)

            # ── Wire ensemble_engine for prediction metadata (Sprint 8.3) ──
            _autonomous_loop._ensemble_engine = getattr(app.state, "ensemble_engine", None)

            # ── Wire news sentiment into autonomous loop (Phase 1: news-aware trading) ──
            # Two-tier: LLM-based (if API key) + FinBERT fallback (always, no API key needed)
            _llm_sentiment_wired = False
            try:
                from src.ai.llm.client import LLMClient, LLMConfig
                from src.ai.llm.sentiment import NewsSentimentService

                _openai_key = os.environ.get("OPENAI_API_KEY", "")
                _anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
                if _openai_key:
                    _llm_config = LLMConfig(provider="openai", api_key=_openai_key, model="gpt-4o-mini")
                elif _anthropic_key:
                    _llm_config = LLMConfig(
                        provider="anthropic", api_key=_anthropic_key, model="claude-3-haiku-20240307"
                    )
                else:
                    _llm_config = None
                if _llm_config:
                    _llm_client = LLMClient(_llm_config)
                    _sentiment_service = NewsSentimentService(_llm_client)
                    _autonomous_loop.set_sentiment_service(_sentiment_service)
                    _llm_sentiment_wired = True
                    logger.info("News sentiment wired to autonomous loop (provider=%s)", _llm_config.provider)
                else:
                    logger.info("No LLM API key set -- LLM sentiment skipped, FinBERT fallback will be used")
            except Exception as e:
                logger.warning("LLM sentiment service not wired: %s", e)

            # Always wire FinBERT sentiment predictor as fallback (works without API keys)
            try:
                from src.ai.models.sentiment_predictor import SentimentPredictor

                _finbert_predictor = SentimentPredictor()
                _autonomous_loop.set_sentiment_predictor(_finbert_predictor)
                if _llm_sentiment_wired:
                    logger.info("FinBERT sentiment predictor wired as fallback (LLM is primary)")
                else:
                    logger.info("FinBERT sentiment predictor wired as primary sentiment source (no LLM API key)")
            except Exception as e:
                logger.debug("FinBERT sentiment predictor not wired: %s", e)

            # ── Multi-Agent Orchestrator ──
            try:
                from src.agents.base import AgentOrchestrator
                from src.agents.execution_agent import ExecutionAgent
                from src.agents.research_agent import ResearchAgent
                from src.agents.risk_agent import RiskMonitorAgent
                from src.agents.strategy_selector import StrategySelectorAgent

                orchestrator = AgentOrchestrator()

                # Pass MarketScanner for full-market discovery beyond BarCache
                _market_scanner_for_agent = None
                try:
                    from src.scanner.market_scanner import MarketScanner

                    _market_scanner_for_agent = MarketScanner(
                        alpha_model=_alpha_model,
                        feature_engine=feature_engine,
                        top_n=20,
                        min_confidence=0.55,
                    )
                except Exception:
                    pass

                risk_manager = getattr(app.state, "risk_manager", None)
                circuit_breaker = getattr(app.state, "circuit_breaker", None)
                kill_switch = getattr(app.state, "kill_switch", None)

                research_agent = ResearchAgent(
                    get_symbols=_get_symbols,
                    get_bars=_get_bars,
                    feature_engine=feature_engine,
                    ensemble_engine=_ensemble_engine,
                    market_scanner=_market_scanner_for_agent,
                    top_n=10,
                    min_confidence=0.55,
                    scan_interval=300.0,
                )
                orchestrator.register(research_agent)

                risk_agent = RiskMonitorAgent(
                    risk_manager=risk_manager,
                    circuit_breaker=circuit_breaker,
                    kill_switch=kill_switch,
                    get_positions=_get_positions,
                    get_bars=_get_bars,
                    max_portfolio_drawdown_pct=5.0,
                    monitor_interval=30.0,
                )
                orchestrator.register(risk_agent)

                execution_agent = ExecutionAgent(
                    submit_order_fn=_submit_order_async,
                    get_bars=_get_bars,
                    get_positions=_get_positions,
                    max_concurrent_orders=5,
                    execution_interval=15.0,
                )
                orchestrator.register(execution_agent)

                strategy_selector = StrategySelectorAgent(
                    strategy_registry=registry,
                    regime_classifier=regime_classifier,
                    get_bars=_get_bars,
                    selection_interval=120.0,
                )
                orchestrator.register(strategy_selector)

                # Broadcast callback for WebSocket
                async def _agent_broadcast(msg):
                    from src.api.ws_manager import get_ws_manager

                    mgr = get_ws_manager()
                    if mgr:
                        await mgr.broadcast(
                            {
                                "type": f"agent_{msg.msg_type}",
                                "source": msg.source,
                                "payload": msg.payload,
                                "timestamp": msg.timestamp,
                            }
                        )

                orchestrator.set_broadcast_callback(_agent_broadcast)

                orchestrator.start_all()
                app.state.agent_orchestrator = orchestrator
                logger.info("Agent Orchestrator started (research, risk, execution, strategy_selector)")
            except Exception as e:
                logger.warning("Agent Orchestrator not started: %s", e)

            # ── Auto-Retrain Scheduler ──
            setup_auto_retrain(app, _alpha_model, _model_path, _models_dir)

            # ── Drift Detection Scheduler (daily after market close) ──
            setup_drift_detector(app)

            # ── Self-Learning Scheduler (Sprint 4.1 + 4.5) ──
            setup_self_learning(
                app, _ensemble_engine, _alpha_model, _model_path, _models_dir, feature_engine, bar_cache
            )

            # ── Daily Report Scheduler (16:00 IST after market close) ──
            try:
                _drg = getattr(app.state, "daily_report_generator", None)
                if _drg:

                    async def _daily_report_loop():
                        """Generate daily performance report at 16:00 IST."""
                        from datetime import datetime as _dt
                        from datetime import timedelta as _td
                        from datetime import timezone as _tz

                        _IST = _tz(_td(hours=5, minutes=30))
                        _last_report_date = None
                        while True:
                            await asyncio.sleep(300)  # check every 5 min
                            try:
                                now_ist = _dt.now(_IST)
                                today = now_ist.strftime("%Y-%m-%d")
                                if now_ist.hour != 16 or now_ist.minute > 5:
                                    continue
                                if now_ist.weekday() >= 5:
                                    continue
                                if _last_report_date == today:
                                    continue

                                _last_report_date = today
                                drg = getattr(app.state, "daily_report_generator", None)
                                if drg:
                                    report = drg.generate(date=today)
                                    try:
                                        drg.save_to_db(report)
                                    except Exception:
                                        pass
                                    logger.info(
                                        "Daily report generated: %s trades, net_pnl=%.2f, sharpe=%.2f",
                                        report.total_trades,
                                        report.net_pnl,
                                        report.sharpe_ratio_20d,
                                    )
                                    from src.api.ws_manager import get_ws_manager

                                    mgr = get_ws_manager()
                                    if mgr:
                                        from src.reporting.daily_report import DailyReportGenerator

                                        await mgr.broadcast(
                                            {"type": "daily_report", "report": DailyReportGenerator.to_dict(report)}
                                        )
                            except asyncio.CancelledError:
                                break
                            except Exception as e:
                                logger.debug("Daily report error: %s", e)

                    _report_task = asyncio.get_running_loop().create_task(_daily_report_loop())
                    app.state._report_task = _report_task
                    logger.info("Daily report scheduler active (generates at 16:00 IST)")
            except Exception as e:
                logger.debug("Daily report scheduler not configured: %s", e)

            # ── Nightly Simulation Scheduler (16:30 IST after market close) ──
            try:

                async def _nightly_simulation_loop():
                    """Run nightly strategy simulation at 16:30 IST."""
                    from datetime import datetime as _dt
                    from datetime import timedelta as _td
                    from datetime import timezone as _tz

                    _IST = _tz(_td(hours=5, minutes=30))
                    _last_sim_date = None
                    while True:
                        await asyncio.sleep(300)
                        try:
                            now_ist = _dt.now(_IST)
                            today = now_ist.strftime("%Y-%m-%d")
                            if now_ist.hour != 16 or now_ist.minute < 25 or now_ist.minute > 35:
                                continue
                            if now_ist.weekday() >= 5:
                                continue
                            if _last_sim_date == today:
                                continue
                            _last_sim_date = today
                            logger.info("Starting nightly simulation pipeline...")
                            from src.simulation.orchestrator import SimulationOrchestrator

                            sim_orch = SimulationOrchestrator()
                            results = await sim_orch.run_nightly_pipeline(intervals=["15m", "1h"])
                            logger.info("Nightly simulation complete: %d results", len(results) if results else 0)
                            from src.api.ws_manager import get_ws_manager

                            mgr = get_ws_manager()
                            if mgr:
                                await mgr.broadcast(
                                    {"type": "simulation_complete", "result_count": len(results) if results else 0}
                                )
                        except asyncio.CancelledError:
                            break
                        except Exception as e:
                            logger.warning("Nightly simulation error: %s", e)

                _sim_task = asyncio.get_running_loop().create_task(_nightly_simulation_loop())
                app.state._sim_task = _sim_task
                logger.info("Nightly simulation scheduler active (runs at 16:30 IST)")
            except Exception as e:
                logger.debug("Nightly simulation scheduler not configured: %s", e)

            # ── Pre-Market Briefing Scheduler (9:00 IST before market open) ──
            try:

                async def _pre_market_briefing_loop():
                    """Generate pre-market briefing at 9:00 IST."""
                    from datetime import datetime as _dt
                    from datetime import timedelta as _td
                    from datetime import timezone as _tz

                    _IST = _tz(_td(hours=5, minutes=30))
                    _last_brief_date = None
                    while True:
                        await asyncio.sleep(300)
                        try:
                            now_ist = _dt.now(_IST)
                            today = now_ist.strftime("%Y-%m-%d")
                            if now_ist.hour != 9 or now_ist.minute > 5:
                                continue
                            if now_ist.weekday() >= 5:
                                continue
                            if _last_brief_date == today:
                                continue
                            _last_brief_date = today
                            logger.info("Generating pre-market briefing...")
                            from src.ai.llm.pre_market_brief import PreMarketBriefing

                            briefing_engine = PreMarketBriefing()
                            briefing = await briefing_engine.generate_briefing()
                            # Apply exposure multiplier to risk manager
                            rm = getattr(app.state, "risk_manager", None)
                            if rm and briefing and hasattr(briefing, "exposure_multiplier"):
                                rm._exposure_multiplier = briefing.exposure_multiplier
                                logger.info(
                                    "Pre-market briefing: outlook=%s, exposure_mult=%.2f",
                                    getattr(briefing, "outlook", "unknown"),
                                    briefing.exposure_multiplier,
                                )
                            from src.api.ws_manager import get_ws_manager

                            mgr = get_ws_manager()
                            if mgr and briefing:
                                await mgr.broadcast(
                                    {
                                        "type": "pre_market_briefing",
                                        "outlook": getattr(briefing, "outlook", "neutral"),
                                        "confidence": getattr(briefing, "confidence", 0.5),
                                        "exposure_multiplier": getattr(briefing, "exposure_multiplier", 1.0),
                                        "key_risks": getattr(briefing, "key_risks", []),
                                    }
                                )
                        except asyncio.CancelledError:
                            break
                        except Exception as e:
                            logger.warning("Pre-market briefing error: %s", e)

                _brief_task = asyncio.get_running_loop().create_task(_pre_market_briefing_loop())
                app.state._brief_task = _brief_task
                logger.info("Pre-market briefing scheduler active (runs at 9:00 IST)")
            except Exception as e:
                logger.debug("Pre-market briefing scheduler not configured: %s", e)

        except Exception as e:
            logger.warning("Autonomous loop not started: %s", e)


async def shutdown_trading(app: FastAPI) -> None:
    """Shutdown: stop self-learning, agent orchestrator, autonomous loop, snapshot open trades."""

    # ── Stop self-learning scheduler gracefully ──
    _sls = getattr(app.state, "self_learning_scheduler", None)
    if _sls is not None:
        try:
            _sls.stop()
            logger.info("SelfLearningScheduler stopped")
        except Exception as ex:
            logger.warning("SelfLearningScheduler stop: %s", ex)

    # Cancel walk-forward background task
    _wf_task = getattr(app.state, "_walk_forward_task", None)
    if _wf_task is not None and not _wf_task.done():
        _wf_task.cancel()
        logger.debug("Walk-forward revalidation task cancelled")

    _orch = getattr(app.state, "agent_orchestrator", None)
    if _orch is not None:
        try:
            await _orch.stop_all()
        except Exception as ex:
            logger.warning("Agent orchestrator stop: %s", ex)
    _al = getattr(app.state, "autonomous_loop", None)
    if _al is not None:
        try:
            await _al.stop()
        except Exception as ex:
            logger.warning("Autonomous loop stop: %s", ex)

    # ── Snapshot open trades to TradeStore before exit ──
    if hasattr(app.state, "trade_store") and hasattr(app.state, "autonomous_loop"):
        _loop_ref = app.state.autonomous_loop
        for trade_key, trade in _loop_ref._open_trades.items():
            try:
                app.state.trade_store.upsert_trade(
                    trade_key=trade_key,
                    symbol=trade.get("symbol", ""),
                    exchange=trade.get("exchange", "NSE"),
                    side=trade.get("side", ""),
                    quantity=trade.get("quantity", 0),
                    entry_price=trade.get("entry_price", 0),
                    stop_loss=trade.get("stop_loss"),
                    take_profit=trade.get("take_profit"),
                )
            except Exception as e:
                logger.warning("Failed to snapshot trade %s: %s", trade_key, e)
        logger.info("Persisted %d open trades to TradeStore on shutdown", len(_loop_ref._open_trades))
