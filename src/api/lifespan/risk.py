"""
Lifespan: RiskManager, circuit breaker, institutional risk modules (VaR, correlation guard,
sector classifier, gap risk, tail risk, vol targeting), freeze qty, market impact,
alert notifier, SEBI audit, daily report, ADV cache.
"""
import logging
import os

from fastapi import FastAPI

logger = logging.getLogger(__name__)


async def init_risk(app: FastAPI) -> None:
    """Initialize institutional risk modules and wire them into RiskManager."""

    # ── Institutional Risk Modules ──
    try:
        from src.risk_engine.var import PortfolioVaR
        app.state.portfolio_var = PortfolioVaR(horizon_days=1)
        logger.info("PortfolioVaR initialized (95%%/99%% parametric VaR)")
    except Exception as e:
        logger.warning("PortfolioVaR not initialized: %s", e)

    try:
        from src.risk_engine.correlation import CorrelationGuard
        app.state.correlation_guard = CorrelationGuard(max_pairwise_correlation=0.70, max_portfolio_vol_pct=15.0)
        logger.info("CorrelationGuard initialized (max_corr=0.70, max_port_vol=15%%)")
    except Exception as e:
        logger.warning("CorrelationGuard not initialized: %s", e)

    try:
        from src.risk_engine.sector_map import SectorClassifier
        app.state.sector_classifier = SectorClassifier()
        logger.info("SectorClassifier initialized (%d sectors)", len(app.state.sector_classifier.list_sectors()))
    except Exception as e:
        logger.warning("SectorClassifier not initialized: %s", e)

    # NOTE: GapRiskManager is initialized later at lines 131-137 (Sprint 7.3).
    # Do not duplicate initialization here.

    try:
        from src.risk_engine.tail_risk import TailRiskProtector
        app.state.tail_risk_protector = TailRiskProtector()
        logger.info("TailRiskProtector initialized (VIX + rapid drawdown defense)")
    except Exception as e:
        logger.warning("TailRiskProtector not initialized: %s", e)

    try:
        from src.risk_engine.vol_targeting import VolatilityTargeter
        app.state.vol_targeter = VolatilityTargeter(target_vol_annual=12.0)
        logger.info("VolatilityTargeter initialized (target=12%% annualized)")
    except Exception as e:
        logger.warning("VolatilityTargeter not initialized: %s", e)

    try:
        from src.execution.freeze_qty import FreezeQuantityManager
        app.state.freeze_qty_manager = FreezeQuantityManager()
        logger.info("FreezeQuantityManager initialized")
    except Exception as e:
        logger.warning("FreezeQuantityManager not initialized: %s", e)

    try:
        from src.execution.market_impact import MarketImpactModel
        app.state.market_impact_model = MarketImpactModel(gamma=0.25)
        logger.info("MarketImpactModel initialized (Almgren-Chriss)")
    except Exception as e:
        logger.warning("MarketImpactModel not initialized: %s", e)

    try:
        from src.alerts.notifier import AlertNotifier
        app.state.alert_notifier = AlertNotifier()
        logger.info("AlertNotifier initialized (channels: %s)", app.state.alert_notifier.config.enabled_channels)
    except Exception as e:
        logger.warning("AlertNotifier not initialized: %s", e)

    # ── Wire institutional risk modules into RiskManager ──
    rm = getattr(app.state, "risk_manager", None)
    if rm is not None:
        rm._portfolio_var = getattr(app.state, "portfolio_var", None)
        rm._correlation_guard = getattr(app.state, "correlation_guard", None)
        rm._tail_risk_protector = getattr(app.state, "tail_risk_protector", None)
        wired = [n for n in ("portfolio_var", "correlation_guard", "tail_risk_protector") if getattr(rm, f"_{n}") is not None]
        if wired:
            logger.info("RiskManager: institutional modules wired: %s", wired)

    # ── Load equity from DB snapshot (GAP-6 fix) ──
    if rm is not None:
        try:
            snap_repo = getattr(app.state, "risk_snapshot_repo", None)
            if snap_repo is not None:
                snap = snap_repo.get_latest_sync()
                if snap and getattr(snap, "equity", 0) > 0:
                    rm.equity = snap.equity
                    rm.daily_pnl = getattr(snap, "daily_pnl", 0.0)
                    logger.info("RiskManager equity loaded from DB snapshot: %.2f", rm.equity)
        except Exception as e:
            logger.warning("Could not load equity from DB snapshot: %s", e)

    try:
        from src.compliance.audit_trail import SEBIAuditTrail
        app.state.sebi_audit = SEBIAuditTrail()
        app.state.audit_trail = app.state.sebi_audit  # Alias for audit router
        logger.info("SEBI audit trail initialized (append-only)")
    except Exception as e:
        logger.debug("SEBI audit trail not initialized: %s", e)

    try:
        from src.reporting.daily_report import DailyReportGenerator
        app.state.daily_report_generator = DailyReportGenerator(
            risk_manager=getattr(app.state, "risk_manager", None),
            persistence_service=getattr(app.state, "persistence_service", None),
        )
        logger.info("DailyReportGenerator initialized")
    except Exception as e:
        logger.debug("DailyReportGenerator not initialized: %s", e)

    # ── ADV Cache (Sprint 7.7) ──
    try:
        from src.market_data.adv_cache import ADVCache
        _adv_cache = ADVCache()
        app.state.adv_cache = _adv_cache
        logger.info("ADVCache initialized")
    except Exception as e:
        logger.debug("ADVCache not initialized: %s", e)

    # ── Stress Testing Engine (Sprint 10.12) ──
    try:
        from src.risk_engine.stress_testing import StressTestEngine
        _sector_clf = getattr(app.state, "sector_classifier", None)
        app.state.stress_test_engine = StressTestEngine(sector_classifier=_sector_clf)
        logger.info("StressTestEngine initialized (10 scenarios: 4 historical + 6 hypothetical)")
    except Exception as e:
        logger.debug("StressTestEngine not initialized: %s", e)

    # ── Data Archival Manager (Sprint 10.13) ──
    try:
        from src.persistence.archival import DataArchivalManager
        app.state.archival_manager = DataArchivalManager()
        logger.info("DataArchivalManager initialized (SEBI 7yr audit retention)")
    except Exception as e:
        logger.debug("DataArchivalManager not initialized: %s", e)

    # ── Gap Risk Manager (Sprint 7.3) ──
    try:
        from src.risk_engine.gap_risk import GapRiskManager
        _gap_risk_mgr = GapRiskManager()
        app.state.gap_risk_manager = _gap_risk_mgr
        logger.info("GapRiskManager initialized")
    except Exception as e:
        logger.debug("GapRiskManager not initialized: %s", e)
