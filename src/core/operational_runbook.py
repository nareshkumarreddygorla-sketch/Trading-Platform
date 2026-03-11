"""
Operational runbook: system health checks, pre-market readiness gate,
post-market reconciliation, incident classification, alert escalation,
and recovery procedures.

Production-grade: provides a systematic approach to monitoring and incident
response for autonomous trading operations.
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Severity(str, Enum):
    """Incident severity classification."""
    P1 = "P1"  # Critical: trading halted, data loss, broker disconnect
    P2 = "P2"  # High: degraded performance, partial outage
    P3 = "P3"  # Medium: non-critical alerts, performance degradation
    P4 = "P4"  # Low: informational, cosmetic issues


class SystemComponent(str, Enum):
    """Monitored system components."""
    DATABASE = "database"
    REDIS = "redis"
    BROKER_API = "broker_api"
    MARKET_DATA = "market_data"
    AI_MODELS = "ai_models"
    RISK_ENGINE = "risk_engine"
    ORDER_ENTRY = "order_entry"
    AUTONOMOUS_LOOP = "autonomous_loop"
    WEBSOCKET = "websocket"
    KILL_SWITCH = "kill_switch"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    component: str
    healthy: bool
    latency_ms: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "healthy": self.healthy,
            "latency_ms": round(self.latency_ms, 1),
            "message": self.message,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class IncidentReport:
    """Structured incident report."""
    severity: Severity
    component: str
    title: str
    description: str
    detected_at: datetime = field(default_factory=_utc_now)
    resolved_at: Optional[datetime] = None
    actions_taken: List[str] = field(default_factory=list)
    recovery_procedure: str = ""
    escalation_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "component": self.component,
            "title": self.title,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "actions_taken": self.actions_taken,
            "recovery_procedure": self.recovery_procedure,
            "escalation_path": self.escalation_path,
        }


# Recovery procedures per failure type
RECOVERY_PROCEDURES = {
    SystemComponent.DATABASE: {
        "description": "Database connectivity failure recovery",
        "steps": [
            "1. Check PostgreSQL/SQLite service status",
            "2. Verify DATABASE_URL environment variable",
            "3. Check connection pool exhaustion (pool_checked_out vs pool_size)",
            "4. If PostgreSQL: check pg_stat_activity for blocked queries",
            "5. If disk full: run archival to free space",
            "6. Restart database service if unresponsive",
            "7. Verify tables exist with alembic upgrade head",
        ],
        "auto_recovery": "Connection pool auto-reconnects with pool_pre_ping=True",
    },
    SystemComponent.REDIS: {
        "description": "Redis connectivity failure recovery",
        "steps": [
            "1. Check Redis service: redis-cli ping",
            "2. Verify REDIS_URL environment variable",
            "3. Check Redis memory usage: redis-cli info memory",
            "4. If maxmemory reached: flush expired keys or increase limit",
            "5. Restart Redis service if unresponsive",
            "6. Idempotency store falls back to in-memory on Redis failure",
        ],
        "auto_recovery": "In-memory fallback for idempotency; distributed locks disabled",
    },
    SystemComponent.BROKER_API: {
        "description": "Broker API failure recovery",
        "steps": [
            "1. Check broker API status page (Angel One / Zerodha)",
            "2. Verify API credentials (api_key, api_secret, access_token)",
            "3. Check if token has expired — re-authenticate",
            "4. Verify TOTP secret is correct",
            "5. Check rate limits (Max 10 req/sec for Angel One)",
            "6. If persistent failure: enter safe_mode to prevent new orders",
            "7. Existing positions remain; no new orders until reconnected",
        ],
        "auto_recovery": "3 consecutive heartbeat failures trigger safe_mode automatically",
    },
    SystemComponent.MARKET_DATA: {
        "description": "Market data feed failure recovery",
        "steps": [
            "1. Check WebSocket connection status to Angel One",
            "2. Verify market hours (NSE: 9:15-15:30 IST, Mon-Fri)",
            "3. If Angel One WS down: yfinance fallback activates automatically",
            "4. Check bar_cache freshness (should update every 1 minute)",
            "5. If all feeds down: signal generation pauses automatically",
            "6. Reconnection attempts happen automatically every 30 seconds",
        ],
        "auto_recovery": "Automatic reconnect + yfinance fallback feeder",
    },
    SystemComponent.AI_MODELS: {
        "description": "AI model failure recovery",
        "steps": [
            "1. Check model files exist in models/ directory",
            "2. Verify model versions: alpha_xgb.joblib, lstm, transformer",
            "3. If model file corrupted: retrain with scripts/train_alpha_model.py",
            "4. If ensemble engine fails: individual model fallbacks activate",
            "5. Check model IC score — if < 0.01, trigger retraining",
            "6. Sentiment model (FinBERT) runs locally — no API key needed",
        ],
        "auto_recovery": "Strategy runner works with any subset of models loaded",
    },
    SystemComponent.RISK_ENGINE: {
        "description": "Risk engine failure recovery",
        "steps": [
            "1. Check RiskManager state: equity, daily_pnl, positions",
            "2. If circuit breaker tripped: review daily_pnl and drawdown",
            "3. If kill switch armed: check arm reason (loss limit, VIX, manual)",
            "4. To reset circuit breaker: POST /api/v1/risk/circuit/reset",
            "5. To disarm kill switch: POST /api/v1/risk/kill-switch/disarm",
            "6. Verify risk limits in config (max_position_pct, max_daily_loss_pct)",
        ],
        "auto_recovery": "Kill switch auto-disarms after cooldown if conditions improve",
    },
    SystemComponent.KILL_SWITCH: {
        "description": "Kill switch state recovery",
        "steps": [
            "1. Check kill switch state: GET /api/v1/risk/kill-switch/status",
            "2. If armed by VIX spike: wait for VIX to normalize (<25)",
            "3. If armed by loss limit: review positions and P&L",
            "4. Manual disarm: POST /api/v1/risk/kill-switch/disarm",
            "5. Auto-disarm requires: broker healthy + VIX normal + cooldown elapsed",
            "6. All open positions were closed when kill switch armed",
        ],
        "auto_recovery": "Auto-disarms when VIX normalizes and broker is healthy",
    },
}

# Alert escalation paths by severity
ESCALATION_PATHS = {
    Severity.P1: "Immediate: PagerDuty/SMS -> On-call engineer -> Engineering lead -> CTO (within 5 min)",
    Severity.P2: "Urgent: Slack #trading-alerts -> On-call engineer (within 15 min)",
    Severity.P3: "Standard: Slack #trading-monitoring -> Review next business day",
    Severity.P4: "Low: Log only -> Weekly review",
}


class OperationalRunbook:
    """
    Provides production-grade operational monitoring and incident response:
    - System health checklist (DB, Redis, broker, models)
    - Pre-market readiness gate
    - Post-market reconciliation trigger
    - Incident classification (P1-P4)
    - Alert escalation paths
    - Recovery procedures per failure type
    """

    def __init__(
        self,
        app_state=None,
        audit_repo=None,
        alert_callback: Optional[Callable] = None,
    ):
        self._app_state = app_state
        self._audit_repo = audit_repo
        self._alert_callback = alert_callback
        self._incidents: List[IncidentReport] = []
        self._last_health_check: Optional[Dict[str, HealthCheckResult]] = None
        self._last_health_check_ts: float = 0
        self._pre_market_ready: bool = False

    # ── System Health Checklist ──

    async def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """
        Run comprehensive system health checks.
        Returns dict of component -> HealthCheckResult.
        """
        results: Dict[str, HealthCheckResult] = {}

        # 1. Database connectivity
        results["database"] = await self._check_database()

        # 2. Redis connectivity
        results["redis"] = await self._check_redis()

        # 3. Broker API
        results["broker_api"] = await self._check_broker()

        # 4. AI Models
        results["ai_models"] = self._check_models()

        # 5. Risk Engine
        results["risk_engine"] = self._check_risk_engine()

        # 6. Autonomous Loop
        results["autonomous_loop"] = self._check_autonomous_loop()

        # 7. Market Data
        results["market_data"] = self._check_market_data()

        # 8. Kill Switch
        results["kill_switch"] = self._check_kill_switch()

        self._last_health_check = results
        self._last_health_check_ts = time.time()

        # Classify any failures as incidents
        for name, result in results.items():
            if not result.healthy:
                self._create_incident_from_health_check(name, result)

        healthy_count = sum(1 for r in results.values() if r.healthy)
        total = len(results)
        logger.info(
            "Health check complete: %d/%d healthy",
            healthy_count, total,
        )

        return results

    async def _check_database(self) -> HealthCheckResult:
        """Check database connectivity and health."""
        start = time.time()
        try:
            from src.persistence.database import check_database_health
            health = check_database_health()
            latency = (time.time() - start) * 1000
            return HealthCheckResult(
                component="database",
                healthy=health["healthy"],
                latency_ms=latency,
                message="OK" if health["healthy"] else health.get("error", "unhealthy"),
                details=health,
            )
        except Exception as e:
            return HealthCheckResult(
                component="database",
                healthy=False,
                latency_ms=(time.time() - start) * 1000,
                message=f"Check failed: {e}",
            )

    async def _check_redis(self) -> HealthCheckResult:
        """Check Redis connectivity."""
        start = time.time()
        try:
            import redis
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            r = redis.from_url(redis_url, socket_timeout=5)
            r.ping()
            info = r.info("memory")
            used_mb = info.get("used_memory", 0) / (1024 * 1024)
            r.close()
            latency = (time.time() - start) * 1000
            return HealthCheckResult(
                component="redis",
                healthy=True,
                latency_ms=latency,
                message="OK",
                details={"used_memory_mb": round(used_mb, 1)},
            )
        except Exception as e:
            return HealthCheckResult(
                component="redis",
                healthy=False,
                latency_ms=(time.time() - start) * 1000,
                message=f"Redis unavailable: {e}",
                details={"fallback": "in-memory idempotency"},
            )

    async def _check_broker(self) -> HealthCheckResult:
        """Check broker API connectivity."""
        start = time.time()
        try:
            if self._app_state is None:
                return HealthCheckResult(
                    component="broker_api",
                    healthy=True,
                    message="No app state (standalone mode)",
                )
            gateway = getattr(self._app_state, "gateway", None)
            if gateway is None:
                return HealthCheckResult(
                    component="broker_api",
                    healthy=False,
                    message="No broker gateway configured",
                )
            paper = getattr(gateway, "paper", True)
            if paper:
                return HealthCheckResult(
                    component="broker_api",
                    healthy=True,
                    latency_ms=(time.time() - start) * 1000,
                    message="Paper mode (no live broker)",
                    details={"mode": "paper"},
                )
            # Live mode: check last heartbeat
            last_ok = getattr(self._app_state, "_last_broker_ok_ts", 0)
            age = time.time() - last_ok if last_ok else 999
            healthy = age < 120  # Healthy if heartbeat within 2 min
            return HealthCheckResult(
                component="broker_api",
                healthy=healthy,
                latency_ms=(time.time() - start) * 1000,
                message="OK" if healthy else f"Last heartbeat {age:.0f}s ago",
                details={"last_heartbeat_age_s": round(age, 1), "mode": "live"},
            )
        except Exception as e:
            return HealthCheckResult(
                component="broker_api",
                healthy=False,
                latency_ms=(time.time() - start) * 1000,
                message=f"Broker check failed: {e}",
            )

    def _check_models(self) -> HealthCheckResult:
        """Check AI model status."""
        try:
            if self._app_state is None:
                return HealthCheckResult(component="ai_models", healthy=True, message="No app state")
            ensemble = getattr(self._app_state, "ensemble_engine", None)
            if ensemble is None:
                return HealthCheckResult(
                    component="ai_models",
                    healthy=True,
                    message="No ensemble engine (using heuristic strategies)",
                )
            model_count = 0
            ready_count = 0
            model_registry = getattr(self._app_state, "model_registry", None)
            if model_registry:
                models = model_registry.list_models() if hasattr(model_registry, "list_models") else []
                model_count = len(models)
                for mid in models:
                    m = model_registry.get(mid)
                    if m and getattr(m, "is_ready", True):
                        ready_count += 1
            return HealthCheckResult(
                component="ai_models",
                healthy=True,
                message=f"{ready_count}/{model_count} models ready",
                details={"total": model_count, "ready": ready_count},
            )
        except Exception as e:
            return HealthCheckResult(
                component="ai_models",
                healthy=False,
                message=f"Model check failed: {e}",
            )

    def _check_risk_engine(self) -> HealthCheckResult:
        """Check risk engine state."""
        try:
            if self._app_state is None:
                return HealthCheckResult(component="risk_engine", healthy=True, message="No app state")
            rm = getattr(self._app_state, "risk_manager", None)
            if rm is None:
                return HealthCheckResult(
                    component="risk_engine",
                    healthy=False,
                    message="RiskManager not initialized",
                )
            circuit_open = rm.is_circuit_open()
            safe_mode = getattr(self._app_state, "safe_mode", False)
            healthy = not circuit_open and not safe_mode
            return HealthCheckResult(
                component="risk_engine",
                healthy=healthy,
                message="OK" if healthy else (
                    "CIRCUIT OPEN" if circuit_open else "SAFE MODE"
                ),
                details={
                    "equity": rm.equity,
                    "daily_pnl": rm.daily_pnl,
                    "positions": len(rm.positions),
                    "circuit_open": circuit_open,
                    "safe_mode": safe_mode,
                },
            )
        except Exception as e:
            return HealthCheckResult(
                component="risk_engine",
                healthy=False,
                message=f"Risk engine check failed: {e}",
            )

    def _check_autonomous_loop(self) -> HealthCheckResult:
        """Check autonomous loop status."""
        try:
            if self._app_state is None:
                return HealthCheckResult(component="autonomous_loop", healthy=True, message="No app state")
            al = getattr(self._app_state, "autonomous_loop", None)
            if al is None:
                return HealthCheckResult(
                    component="autonomous_loop",
                    healthy=False,
                    message="Autonomous loop not configured",
                )
            running = getattr(al, "_running", False)
            tick_count = getattr(al, "_tick_count", 0)
            open_trades = len(getattr(al, "_open_trades", {}))
            circuit_open = getattr(al, "_loop_circuit_open", False)
            healthy = running and not circuit_open
            return HealthCheckResult(
                component="autonomous_loop",
                healthy=healthy,
                message="OK" if healthy else (
                    "CIRCUIT OPEN" if circuit_open else "NOT RUNNING"
                ),
                details={
                    "running": running,
                    "tick_count": tick_count,
                    "open_trades": open_trades,
                    "circuit_open": circuit_open,
                },
            )
        except Exception as e:
            return HealthCheckResult(
                component="autonomous_loop",
                healthy=False,
                message=f"Loop check failed: {e}",
            )

    def _check_market_data(self) -> HealthCheckResult:
        """Check market data feed status."""
        try:
            if self._app_state is None:
                return HealthCheckResult(component="market_data", healthy=True, message="No app state")
            mds = getattr(self._app_state, "market_data_service", None)
            if mds is not None:
                healthy = mds.is_healthy()
                status = mds.get_status() if hasattr(mds, "get_status") else {}
                return HealthCheckResult(
                    component="market_data",
                    healthy=healthy,
                    message="OK" if healthy else "Feed unhealthy",
                    details=status,
                )
            # Check yfinance feeder as fallback
            yf = getattr(self._app_state, "yf_feeder", None)
            if yf is not None:
                running = getattr(yf, "_running", False)
                return HealthCheckResult(
                    component="market_data",
                    healthy=running,
                    message="YFinance feeder " + ("running" if running else "stopped"),
                    details={"source": "yfinance"},
                )
            return HealthCheckResult(
                component="market_data",
                healthy=True,
                message="No market data service (paper mode)",
            )
        except Exception as e:
            return HealthCheckResult(
                component="market_data",
                healthy=False,
                message=f"Market data check failed: {e}",
            )

    def _check_kill_switch(self) -> HealthCheckResult:
        """Check kill switch state."""
        try:
            if self._app_state is None:
                return HealthCheckResult(component="kill_switch", healthy=True, message="No app state")
            ks = getattr(self._app_state, "kill_switch", None)
            if ks is None:
                return HealthCheckResult(
                    component="kill_switch",
                    healthy=True,
                    message="No kill switch configured",
                )
            armed = ks.is_armed()
            return HealthCheckResult(
                component="kill_switch",
                healthy=not armed,
                message="ARMED" if armed else "OK (disarmed)",
                details={"armed": armed},
            )
        except Exception as e:
            return HealthCheckResult(
                component="kill_switch",
                healthy=False,
                message=f"Kill switch check failed: {e}",
            )

    # ── Pre-Market Readiness Gate ──

    async def pre_market_readiness_check(self) -> Dict[str, Any]:
        """
        Run all pre-market checks. ALL systems must report ready before
        trading starts.

        Returns:
            ready: bool - True if all checks pass
            checks: dict - individual check results
            blockers: list - list of failed checks preventing trading
        """
        result: Dict[str, Any] = {
            "ready": True,
            "checks": {},
            "blockers": [],
            "timestamp": _utc_now().isoformat(),
        }

        health = await self.run_health_checks()

        # Critical checks that must pass
        critical_components = ["database", "risk_engine"]
        # Important but non-blocking in paper mode
        live_required = ["broker_api", "market_data"]

        is_live = False
        if self._app_state:
            gateway = getattr(self._app_state, "gateway", None)
            is_live = gateway is not None and not getattr(gateway, "paper", True)

        for component, check in health.items():
            result["checks"][component] = check.to_dict()
            if not check.healthy:
                if component in critical_components:
                    result["ready"] = False
                    result["blockers"].append(f"{component}: {check.message}")
                elif is_live and component in live_required:
                    result["ready"] = False
                    result["blockers"].append(f"{component}: {check.message}")

        # Additional pre-market checks
        # 1. Data freshness
        try:
            if self._app_state:
                bar_cache = getattr(self._app_state, "bar_cache", None)
                if bar_cache:
                    from src.core.events import Exchange
                    symbols = bar_cache.symbols_with_bars(Exchange.NSE, "1m", min_bars=5)
                    result["checks"]["data_freshness"] = {
                        "healthy": len(symbols) > 0,
                        "symbols_with_data": len(symbols),
                    }
                    if len(symbols) == 0 and is_live:
                        result["blockers"].append("data_freshness: no symbols have bar data")
        except Exception as e:
            logger.debug("Data freshness check: %s", e)

        # 2. Model readiness
        try:
            models_check = health.get("ai_models")
            if models_check:
                ready_count = models_check.details.get("ready", 0)
                result["checks"]["model_readiness"] = {
                    "healthy": ready_count > 0,
                    "models_ready": ready_count,
                }
        except Exception:
            pass

        # 3. Risk limits configured
        try:
            if self._app_state:
                rm = getattr(self._app_state, "risk_manager", None)
                if rm and hasattr(rm, "limits"):
                    result["checks"]["risk_limits"] = {
                        "healthy": True,
                        "max_position_pct": rm.limits.max_position_pct,
                        "max_daily_loss_pct": rm.limits.max_daily_loss_pct,
                    }
        except Exception:
            pass

        self._pre_market_ready = result["ready"]

        if result["ready"]:
            logger.info("Pre-market readiness: ALL SYSTEMS READY")
        else:
            logger.warning(
                "Pre-market readiness: NOT READY (%d blockers: %s)",
                len(result["blockers"]),
                ", ".join(result["blockers"]),
            )

        return result

    @property
    def is_pre_market_ready(self) -> bool:
        """Check if pre-market readiness gate has passed."""
        return self._pre_market_ready

    # ── Post-Market Reconciliation ──

    async def trigger_post_market_reconciliation(self) -> Dict[str, Any]:
        """
        Trigger post-market reconciliation checks:
        - Compare local positions with broker
        - Verify all orders are in terminal state
        - Calculate day P&L vs broker P&L
        - Generate end-of-day summary
        """
        report: Dict[str, Any] = {
            "timestamp": _utc_now().isoformat(),
            "checks": {},
        }

        try:
            if self._app_state:
                # 1. Check for any non-terminal orders
                oes = getattr(self._app_state, "order_entry_service", None)
                if oes and hasattr(oes, "lifecycle"):
                    pending = oes.lifecycle.get_pending_orders() if hasattr(oes.lifecycle, "get_pending_orders") else []
                    report["checks"]["pending_orders"] = {
                        "count": len(pending) if pending else 0,
                        "healthy": not pending,
                    }
                    if pending:
                        logger.warning(
                            "Post-market: %d orders still pending — may need manual review",
                            len(pending),
                        )

                # 2. Position summary
                rm = getattr(self._app_state, "risk_manager", None)
                if rm:
                    report["checks"]["positions"] = {
                        "count": len(rm.positions),
                        "equity": rm.equity,
                        "daily_pnl": rm.daily_pnl,
                    }

                # 3. Autonomous loop stats
                al = getattr(self._app_state, "autonomous_loop", None)
                if al:
                    report["checks"]["loop_stats"] = {
                        "tick_count": getattr(al, "_tick_count", 0),
                        "open_trades": len(getattr(al, "_open_trades", {})),
                        "daily_pnl": getattr(al, "_daily_pnl", 0.0),
                    }

                # 4. Trigger broker reconciliation if available
                recon_job = getattr(self._app_state, "reconciliation_job", None)
                if recon_job:
                    try:
                        recon_result = await recon_job.run()
                        report["checks"]["broker_reconciliation"] = {
                            "in_sync": recon_result.in_sync if recon_result else False,
                            "mismatches": len(recon_result.mismatches) if recon_result and hasattr(recon_result, "mismatches") else 0,
                        }
                    except Exception as e:
                        report["checks"]["broker_reconciliation"] = {
                            "error": str(e),
                        }
        except Exception as e:
            report["error"] = str(e)
            logger.error("Post-market reconciliation failed: %s", e)

        logger.info("Post-market reconciliation complete: %s", json.dumps(report.get("checks", {}), default=str))
        return report

    # ── Incident Classification ──

    def classify_incident(
        self,
        component: str,
        error_type: str,
        details: str = "",
    ) -> IncidentReport:
        """
        Classify an incident by severity based on component and error type.
        Returns a structured IncidentReport with recovery procedures.
        """
        # Severity classification rules
        severity = Severity.P4  # Default: low

        p1_conditions = {
            "database": ["connection_failed", "disk_full", "corruption"],
            "broker_api": ["auth_failure", "disconnect", "timeout_all_retries"],
            "risk_engine": ["circuit_open", "equity_zero"],
            "kill_switch": ["armed_loss_limit"],
        }
        p2_conditions = {
            "broker_api": ["heartbeat_failure", "rate_limited"],
            "market_data": ["feed_disconnected", "all_stale"],
            "ai_models": ["ensemble_failure", "all_models_down"],
            "autonomous_loop": ["circuit_open", "not_running"],
        }
        p3_conditions = {
            "market_data": ["single_symbol_stale", "reconnecting"],
            "ai_models": ["single_model_failure", "ic_degradation"],
            "redis": ["connection_failed"],
        }

        for comp, error_types in p1_conditions.items():
            if component == comp and error_type in error_types:
                severity = Severity.P1
                break
        else:
            for comp, error_types in p2_conditions.items():
                if component == comp and error_type in error_types:
                    severity = Severity.P2
                    break
            else:
                for comp, error_types in p3_conditions.items():
                    if component == comp and error_type in error_types:
                        severity = Severity.P3
                        break

        # Get recovery procedure
        comp_enum = None
        for sc in SystemComponent:
            if sc.value == component:
                comp_enum = sc
                break
        recovery = RECOVERY_PROCEDURES.get(comp_enum, {})

        incident = IncidentReport(
            severity=severity,
            component=component,
            title=f"[{severity.value}] {component}: {error_type}",
            description=details or f"{error_type} detected on {component}",
            recovery_procedure=recovery.get("description", "") + "\n" + "\n".join(recovery.get("steps", [])),
            escalation_path=ESCALATION_PATHS.get(severity, ""),
        )

        self._incidents.append(incident)

        # Keep at most 200 incidents
        if len(self._incidents) > 200:
            self._incidents = self._incidents[-200:]

        # Audit log
        if self._audit_repo:
            try:
                self._audit_repo.append_sync(
                    event_type=f"incident_{severity.value}",
                    actor="operational_runbook",
                    payload=incident.to_dict(),
                )
            except Exception as e:
                logger.debug("Failed to audit incident: %s", e)

        # Alert callback for P1/P2
        if severity in (Severity.P1, Severity.P2) and self._alert_callback:
            try:
                self._alert_callback(incident)
            except Exception as e:
                logger.warning("Alert callback failed: %s", e)

        logger.warning(
            "Incident classified: %s — escalation: %s",
            incident.title,
            incident.escalation_path[:80],
        )
        return incident

    def _create_incident_from_health_check(
        self, component: str, check: HealthCheckResult
    ) -> None:
        """Create an incident from a failed health check."""
        error_type = "health_check_failed"
        if "circuit" in check.message.lower():
            error_type = "circuit_open"
        elif "armed" in check.message.lower():
            error_type = "armed_loss_limit"
        elif "timeout" in check.message.lower():
            error_type = "timeout_all_retries"
        elif "unavailable" in check.message.lower() or "failed" in check.message.lower():
            error_type = "connection_failed"

        self.classify_incident(component, error_type, check.message)

    # ── Recovery Procedures ──

    def get_recovery_procedure(self, component: str) -> Dict[str, Any]:
        """Get the recovery procedure for a specific component."""
        comp_enum = None
        for sc in SystemComponent:
            if sc.value == component:
                comp_enum = sc
                break
        return RECOVERY_PROCEDURES.get(comp_enum, {
            "description": f"No recovery procedure defined for {component}",
            "steps": ["Contact engineering team"],
            "auto_recovery": "None",
        })

    # ── Reporting ──

    def get_recent_incidents(
        self,
        severity: Optional[Severity] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get recent incidents, optionally filtered by severity."""
        incidents = self._incidents
        if severity:
            incidents = [i for i in incidents if i.severity == severity]
        return [i.to_dict() for i in incidents[-limit:]]

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status summary."""
        health = self._last_health_check or {}
        healthy_count = sum(1 for r in health.values() if r.healthy) if health else 0
        total = len(health) if health else 0

        active_p1 = sum(1 for i in self._incidents if i.severity == Severity.P1 and i.resolved_at is None)
        active_p2 = sum(1 for i in self._incidents if i.severity == Severity.P2 and i.resolved_at is None)

        overall = "healthy"
        if active_p1 > 0:
            overall = "critical"
        elif active_p2 > 0:
            overall = "degraded"
        elif healthy_count < total:
            overall = "warning"

        return {
            "overall": overall,
            "healthy_components": healthy_count,
            "total_components": total,
            "active_p1_incidents": active_p1,
            "active_p2_incidents": active_p2,
            "pre_market_ready": self._pre_market_ready,
            "last_health_check_age_s": round(time.time() - self._last_health_check_ts, 1) if self._last_health_check_ts else None,
            "timestamp": _utc_now().isoformat(),
        }
