"""
Self-learning scheduler: runs post-market at 15:45 IST daily.

Pipeline:
  1. Run multi-layer drift detection (feature, target, distribution).
  2. If >=2 layers drifted -> trigger immediate retrain for all models.
  3. If not drifted -> run IC weight update for ensemble.
  4. Log results, fire alerts on drift/retrain.
  5. Weekly (Friday): trigger walk-forward revalidation regardless of drift.

Replaces the crude 6-hour subprocess retrain with a principled,
drift-aware, market-close-aligned scheduler.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# IST offset: UTC + 5:30
IST_OFFSET = timedelta(hours=5, minutes=30)


def _now_ist() -> datetime:
    return datetime.now(UTC) + IST_OFFSET


class SelfLearningScheduler:
    """
    Post-market self-learning loop.

    Runs daily at ~15:45 IST (after NSE close at 15:30).
    On each run:
      - Check drift across multiple layers (feature distribution,
        prediction accuracy, target distribution).
      - If significant drift detected: trigger model retrain.
      - Otherwise: update IC-based ensemble weights.
      - Friday: force walk-forward revalidation.

    Usage:
        scheduler = SelfLearningScheduler(
            drift_detector=drift_detector,
            distribution_monitor=dist_monitor,
            retrain_fn=retrain_all,
            ic_update_fn=ensemble.update_weights_from_ic,
            alert_fn=notifier.send,
        )
        scheduler.start()  # fire-and-forget async task
    """

    def __init__(
        self,
        drift_detector=None,
        distribution_monitor=None,
        retrain_fn: Callable[[], dict[str, bool]] | None = None,
        ic_update_fn: Callable[[], dict[str, float]] | None = None,
        alert_fn: Callable | None = None,
        get_recent_features_fn: Callable[[], list[dict[str, float]]] | None = None,
        post_market_hour: int = 15,
        post_market_minute: int = 45,
        min_drift_layers_for_retrain: int = 2,
        weekly_revalidation_day: int = 4,  # Friday = 4 (Monday=0)
    ):
        self.drift_detector = drift_detector
        self.distribution_monitor = distribution_monitor
        self.retrain_fn = retrain_fn
        self.ic_update_fn = ic_update_fn
        self.alert_fn = alert_fn
        self.get_recent_features_fn = get_recent_features_fn
        self.post_market_hour = post_market_hour
        self.post_market_minute = post_market_minute
        self.min_drift_layers = min_drift_layers_for_retrain
        self.weekly_revalidation_day = weekly_revalidation_day

        self._task: asyncio.Task | None = None
        self._running = False
        self._last_run_date: str | None = None
        self._history: list[dict[str, Any]] = []

    def start(self) -> None:
        """Start the scheduler as an asyncio background task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.ensure_future(self._run_loop())
        logger.info(
            "SelfLearningScheduler started (post-market %02d:%02d IST, drift_layers=%d, weekly_day=%d)",
            self.post_market_hour,
            self.post_market_minute,
            self.min_drift_layers,
            self.weekly_revalidation_day,
        )

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("SelfLearningScheduler stopped")

    def get_history(self) -> list[dict[str, Any]]:
        """Return execution history for audit."""
        return list(self._history)

    async def _run_loop(self) -> None:
        """Main scheduler loop: sleep until post-market, then run cycle."""
        while self._running:
            try:
                # Calculate sleep until next post-market window
                sleep_seconds = self._seconds_until_next_run()
                if sleep_seconds > 0:
                    logger.debug(
                        "SelfLearningScheduler sleeping %.0f seconds until next run",
                        sleep_seconds,
                    )
                    await asyncio.sleep(sleep_seconds)

                if not self._running:
                    break

                # Prevent double-run on same calendar day
                today_str = _now_ist().strftime("%Y-%m-%d")
                if self._last_run_date == today_str:
                    await asyncio.sleep(300)  # 5 min guard
                    continue

                self._last_run_date = today_str
                await self._run_cycle(today_str)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("SelfLearningScheduler error: %s", e)
                await asyncio.sleep(60)

    def _seconds_until_next_run(self) -> float:
        """Calculate seconds until next post-market window."""
        now = _now_ist()
        target = now.replace(
            hour=self.post_market_hour,
            minute=self.post_market_minute,
            second=0,
            microsecond=0,
        )
        if now >= target:
            # Already past today's window; schedule for tomorrow
            target += timedelta(days=1)
        delta = (target - now).total_seconds()
        return max(0, delta)

    async def _run_cycle(self, date_str: str) -> None:
        """Execute one post-market self-learning cycle."""
        t0 = time.monotonic()
        now_ist = _now_ist()
        is_weekly = now_ist.weekday() == self.weekly_revalidation_day
        entry: dict[str, Any] = {
            "date": date_str,
            "timestamp": datetime.now(UTC).isoformat(),
            "is_weekly": is_weekly,
        }

        logger.info(
            "SelfLearningScheduler cycle START (date=%s, weekly=%s)",
            date_str,
            is_weekly,
        )

        # Step 1: Gather recent features for drift detection
        recent_features = []
        if self.get_recent_features_fn:
            try:
                recent_features = self.get_recent_features_fn()
            except Exception as e:
                logger.warning("Failed to get recent features: %s", e)

        # Step 2: Multi-layer drift detection
        drift_layers_fired = 0
        drift_reasons: list[str] = []

        # Layer 1: Feature distribution drift (ConceptDriftDetector)
        if self.drift_detector and recent_features:
            try:
                for feat_dict in recent_features[-5:]:  # Check last 5 snapshots
                    drifted, reason = self.drift_detector.detect(feat_dict)
                    if drifted:
                        drift_layers_fired += 1
                        drift_reasons.append(f"feature_drift: {reason}")
                        break
            except Exception as e:
                logger.warning("Feature drift check failed: %s", e)

        # Layer 2: Distribution PSI check
        if self.distribution_monitor:
            try:
                high_psi_features = []
                for feat_name in list(getattr(self.distribution_monitor, "_samples", {}).keys())[:20]:
                    psi_val = self.distribution_monitor.psi(feat_name)
                    if psi_val > 0.2:  # PSI > 0.2 = significant shift
                        high_psi_features.append((feat_name, psi_val))
                if high_psi_features:
                    drift_layers_fired += 1
                    drift_reasons.append(
                        f"psi_drift: {len(high_psi_features)} features (max={max(p for _, p in high_psi_features):.3f})"
                    )
            except Exception as e:
                logger.warning("PSI drift check failed: %s", e)

        # Layer 3: Prediction accuracy drift (check if IC scores have degraded)
        if self.ic_update_fn:
            try:
                ic_result = self.ic_update_fn(recent_predictions=None, actual_returns=None)
                if ic_result and isinstance(ic_result, dict):
                    negative_ic_count = sum(1 for v in ic_result.values() if isinstance(v, (int, float)) and v < 0)
                    total_ic_count = sum(1 for v in ic_result.values() if isinstance(v, (int, float)))
                    if total_ic_count > 0 and negative_ic_count >= total_ic_count * 0.8:
                        drift_layers_fired += 1
                        drift_reasons.append(f"ic_degradation: {negative_ic_count}/{total_ic_count} models negative IC")
                    elif total_ic_count > 0:
                        logger.debug("IC check: %d/%d models have negative IC", negative_ic_count, total_ic_count)
            except Exception as e:
                logger.warning("IC degradation check failed: %s", e)

        entry["drift_layers_fired"] = drift_layers_fired
        entry["drift_reasons"] = drift_reasons

        # Step 3: Decision — retrain or just update weights
        _retrain_triggered = False
        if drift_layers_fired >= self.min_drift_layers or is_weekly:
            # Trigger retrain
            trigger_reason = (
                "weekly_revalidation"
                if is_weekly and drift_layers_fired < self.min_drift_layers
                else f"drift ({drift_layers_fired} layers)"
            )
            logger.warning(
                "SelfLearningScheduler RETRAIN triggered: %s (reasons: %s)",
                trigger_reason,
                drift_reasons,
            )
            _retrain_triggered = True
            entry["action"] = "retrain"
            entry["trigger_reason"] = trigger_reason

            if self.retrain_fn:
                try:
                    loop = asyncio.get_running_loop()
                    retrain_results = await loop.run_in_executor(None, self.retrain_fn)
                    entry["retrain_results"] = retrain_results
                    logger.info("Retrain results: %s", retrain_results)

                    # Fire alert
                    if self.alert_fn:
                        try:
                            replaced = [m for m, r in retrain_results.items() if r]
                            msg = f"Models retrained: {replaced or 'none replaced'} (trigger: {trigger_reason})"
                            await self._fire_alert("INFO", "Model Retrain Complete", msg)
                        except Exception:
                            pass
                except Exception as e:
                    logger.exception("Retrain failed: %s", e)
                    entry["retrain_error"] = str(e)
                    if self.alert_fn:
                        await self._fire_alert("WARNING", "Model Retrain Failed", str(e))
        else:
            # No retrain needed — just update IC weights
            entry["action"] = "ic_update"
            logger.info("SelfLearningScheduler: no drift, updating IC weights")

            if self.ic_update_fn:
                try:
                    new_weights = self.ic_update_fn()
                    entry["ic_weights"] = new_weights
                    logger.info("IC weights updated: %s", new_weights)
                except Exception as e:
                    logger.warning("IC weight update failed: %s", e)
                    entry["ic_error"] = str(e)

        # Step 4: Fit calibrator from recent prediction history (Sprint 8.4)
        _ensemble = getattr(self, "_ensemble", None)
        if _ensemble and hasattr(_ensemble, "fit_calibrator") and hasattr(_ensemble, "_prediction_history"):
            try:
                # Collect raw probs and actual outcomes from prediction history
                raw_probs = []
                actual_outcomes = []
                for model_id, symbol_data in _ensemble._prediction_history.items():
                    for _symbol, pairs in symbol_data.items():
                        for pred_dir, actual_ret in pairs:
                            # Convert predicted direction to probability-like value
                            prob = max(0.0, min(1.0, 0.5 + pred_dir * 0.5))
                            outcome = 1 if actual_ret > 0 else 0
                            raw_probs.append(prob)
                            actual_outcomes.append(outcome)
                if len(raw_probs) >= 100:
                    fitted = _ensemble.fit_calibrator(raw_probs, actual_outcomes, method="isotonic")
                    if fitted:
                        logger.info("Calibrator refitted on %d samples", len(raw_probs))
                        entry["calibrator_fitted"] = True
                        entry["calibrator_samples"] = len(raw_probs)
                    else:
                        entry["calibrator_fitted"] = False
                else:
                    logger.debug("Calibrator: insufficient data (%d < 100)", len(raw_probs))
            except Exception as e:
                logger.warning("Calibrator fitting failed: %s", e)

        elapsed = time.monotonic() - t0
        entry["elapsed_seconds"] = round(elapsed, 2)
        self._history.append(entry)

        # Keep only last 90 days of history
        if len(self._history) > 90:
            self._history = self._history[-90:]

        logger.info(
            "SelfLearningScheduler cycle END (date=%s, action=%s, elapsed=%.1fs)",
            date_str,
            entry.get("action", "unknown"),
            elapsed,
        )

    async def _fire_alert(self, severity: str, title: str, message: str) -> None:
        """Send alert via configured alert function."""
        if not self.alert_fn:
            return
        try:
            # Adapt to AlertNotifier.send() signature
            from src.alerts.notifier import AlertSeverity

            sev_map = {
                "INFO": AlertSeverity.INFO,
                "WARNING": AlertSeverity.WARNING,
                "CRITICAL": AlertSeverity.CRITICAL,
            }
            await self.alert_fn(
                sev_map.get(severity, AlertSeverity.INFO),
                title,
                message,
                source="self_learning_scheduler",
            )
        except Exception as e:
            logger.debug("Alert send failed: %s", e)

    async def run_now(self) -> dict[str, Any]:
        """Force an immediate run (for API/testing). Returns cycle result."""
        date_str = _now_ist().strftime("%Y-%m-%d") + "_manual"
        await self._run_cycle(date_str)
        return self._history[-1] if self._history else {}
