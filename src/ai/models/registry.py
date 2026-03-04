"""
Model registry: versioning, performance tracking, auto-replace when new model beats current.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import BasePredictor

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    version: str
    created_at: datetime
    path: str  # artifact path or key
    metrics: Dict[str, float]  # e.g. sharpe, accuracy, logloss


@dataclass
class ModelMetadata:
    model_id: str
    current_version: str
    versions: List[ModelVersion] = field(default_factory=list)
    performance_log: List[Dict[str, Any]] = field(default_factory=list)


class ModelRegistry:
    """
    Register models by id; track versions and performance; promote new version
    if it beats current on validation metrics.
    """

    def __init__(self):
        self._models: Dict[str, BasePredictor] = {}
        self._metadata: Dict[str, ModelMetadata] = {}

    def register(self, model: BasePredictor, metrics: Optional[Dict[str, float]] = None) -> None:
        self._models[model.model_id] = model
        if model.model_id not in self._metadata:
            self._metadata[model.model_id] = ModelMetadata(
                model_id=model.model_id,
                current_version=model.version,
                versions=[],
            )
        meta = self._metadata[model.model_id]
        meta.current_version = model.version
        meta.versions.append(
            ModelVersion(
                version=model.version,
                created_at=datetime.now(timezone.utc),
                path=getattr(model, "path", ""),
                metrics=metrics or {},
            )
        )

    def get(self, model_id: str) -> Optional[BasePredictor]:
        return self._models.get(model_id)

    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        return self._metadata.get(model_id)

    def log_performance(self, model_id: str, metrics: Dict[str, Any]) -> None:
        if model_id in self._metadata:
            self._metadata[model_id].performance_log.append(
                {"ts": datetime.now(timezone.utc).isoformat(), **metrics}
            )

    def replace_if_better(
        self,
        model_id: str,
        candidate: BasePredictor,
        candidate_metrics: Dict[str, float],
        compare_metric: str = "sharpe",
        higher_is_better: bool = True,
    ) -> bool:
        """
        If candidate's compare_metric is better than current, register candidate
        as new current and return True.
        """
        current = self.get(model_id)
        if current is None:
            self.register(candidate, candidate_metrics)
            return True
        meta = self.get_metadata(model_id)
        if not meta or not meta.versions:
            self.register(candidate, candidate_metrics)
            return True
        current_metric = meta.versions[-1].metrics.get(compare_metric)
        candidate_metric = candidate_metrics.get(compare_metric)
        if current_metric is None:
            self.register(candidate, candidate_metrics)
            return True
        if candidate_metric is None:
            return False
        better = candidate_metric > current_metric if higher_is_better else candidate_metric < current_metric
        if better:
            self.register(candidate, candidate_metrics)
            logger.info("Model %s replaced: %s %s -> %s", model_id, compare_metric, current_metric, candidate_metric)
            return True
        return False
