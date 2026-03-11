"""
Phase 5: Shadow model governance.
Lifecycle: production runs live; candidate trained -> deploy_shadow -> compare -> promote or keep;
rollback if live Sharpe drops below threshold after promotion.
Registry: model_id, version, training_window, metrics, stability_score, promotion_date, status.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    PRODUCTION = "production"
    SHADOW = "shadow"
    ARCHIVED = "archived"


@dataclass
class ModelMetadata:
    model_id: str
    version: str
    training_window_start: str | None = None
    training_window_end: str | None = None
    performance_metrics: dict[str, float] = field(default_factory=dict)
    stability_score: float = 0.0
    promotion_date: str | None = None
    status: ModelStatus = ModelStatus.ARCHIVED


@dataclass
class ShadowResult:
    promoted: bool
    reason: str
    production_metrics: dict[str, float]
    shadow_metrics: dict[str, float]
    replacement_rule_passed: bool


class ShadowModelGovernance:
    """
    Deploy candidate as shadow (live predictions, no execution);
    compare shadow vs production; promote only if replacement_rule passes;
    archive previous; rollback if live Sharpe drops below threshold.
    """

    def __init__(
        self,
        replacement_rule_fn,
        live_sharpe_rollback_threshold: float = 0.3,
        live_sharpe_lookback_days: int = 5,
    ):
        self.replacement_rule_fn = replacement_rule_fn
        self.live_sharpe_rollback_threshold = live_sharpe_rollback_threshold
        self.live_sharpe_lookback_days = live_sharpe_lookback_days
        self._production_model: Any | None = None
        self._shadow_model: Any | None = None
        self._production_metadata: ModelMetadata | None = None
        self._shadow_metadata: ModelMetadata | None = None
        self._registry: list[ModelMetadata] = []
        self._live_sharpes: list[float] = []

    def get_production_model(self) -> Any | None:
        return self._production_model

    def get_production_metadata(self) -> ModelMetadata | None:
        return self._production_metadata

    def deploy_production(self, model: Any, metadata: ModelMetadata) -> None:
        """Set initial production model."""
        if self._production_metadata:
            self._registry.append(self._production_metadata)
            self._production_metadata.status = ModelStatus.ARCHIVED
        self._production_model = model
        self._production_metadata = metadata
        self._production_metadata.status = ModelStatus.PRODUCTION
        self._production_metadata.promotion_date = datetime.now(UTC).isoformat()

    def deploy_shadow(self, candidate_model: Any, candidate_metadata: ModelMetadata) -> None:
        """Deploy candidate as shadow; shadow generates predictions but no execution."""
        self._shadow_model = candidate_model
        self._shadow_metadata = candidate_metadata
        self._shadow_metadata.status = ModelStatus.SHADOW

    def compare_and_promote(
        self,
        shadow_metrics: dict[str, float],
        production_metrics: dict[str, float],
    ) -> ShadowResult:
        """
        Run replacement_rule(shadow vs production); if pass, promote shadow to production,
        archive current production; else keep production.
        """
        passed = self.replacement_rule_fn(
            current_sharpe=production_metrics.get("sharpe", 0.0),
            current_dd_pct=production_metrics.get("max_drawdown_pct", 100.0),
            candidate_sharpe=shadow_metrics.get("sharpe", 0.0),
            candidate_dd_pct=shadow_metrics.get("max_drawdown_pct", 100.0),
            candidate_stability=shadow_metrics.get("stability_score", 0.0),
            candidate_consecutive_positive=shadow_metrics.get("consecutive_positive_windows", 0),
        )
        if passed and self._shadow_model is not None and self._shadow_metadata is not None:
            if self._production_metadata:
                self._production_metadata.status = ModelStatus.ARCHIVED
                self._registry.append(self._production_metadata)
            self._production_model = self._shadow_model
            self._production_metadata = self._shadow_metadata
            self._production_metadata.status = ModelStatus.PRODUCTION
            self._production_metadata.promotion_date = datetime.now(UTC).isoformat()
            self._production_metadata.performance_metrics = shadow_metrics
            self._shadow_model = None
            self._shadow_metadata = None
            logger.info("Shadow promoted to production")
            return ShadowResult(
                promoted=True,
                reason="replacement_rule_passed",
                production_metrics=production_metrics,
                shadow_metrics=shadow_metrics,
                replacement_rule_passed=True,
            )
        return ShadowResult(
            promoted=False,
            reason="replacement_rule_failed" if not passed else "no_shadow",
            production_metrics=production_metrics,
            shadow_metrics=shadow_metrics,
            replacement_rule_passed=passed,
        )

    def record_live_sharpe(self, sharpe: float) -> None:
        """Record live rolling Sharpe for rollback check."""
        self._live_sharpes.append(sharpe)
        if len(self._live_sharpes) > self.live_sharpe_lookback_days * 78:  # e.g. 78 bars/day
            self._live_sharpes = self._live_sharpes[-(self.live_sharpe_lookback_days * 78) :]

    def should_rollback(self) -> bool:
        """True if live Sharpe (e.g. mean over lookback) dropped below threshold."""
        if not self._live_sharpes or len(self._live_sharpes) < 10:
            return False
        mean_sharpe = sum(self._live_sharpes) / len(self._live_sharpes)
        return mean_sharpe < self.live_sharpe_rollback_threshold

    def rollback(self, previous_model: Any, previous_metadata: ModelMetadata) -> bool:
        """Restore previous production model; return True if rollback performed."""
        if self._production_model is None:
            return False
        self._shadow_model = self._production_model
        self._shadow_metadata = self._production_metadata
        self._production_model = previous_model
        self._production_metadata = previous_metadata
        self._production_metadata.status = ModelStatus.PRODUCTION
        logger.warning("Rollback to previous production model")
        return True
