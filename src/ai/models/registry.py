"""
Model registry: versioning, performance tracking, auto-replace when new model beats current.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .base import BasePredictor

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    version: str
    created_at: datetime
    path: str  # artifact path or key
    metrics: dict[str, float]  # e.g. sharpe, accuracy, logloss


@dataclass
class ModelMetadata:
    model_id: str
    current_version: str
    versions: list[ModelVersion] = field(default_factory=list)
    performance_log: list[dict[str, Any]] = field(default_factory=list)


class ModelRegistry:
    """
    Register models by id; track versions and performance; promote new version
    if it beats current on validation metrics.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._models: dict[str, BasePredictor] = {}
        self._metadata: dict[str, ModelMetadata] = {}

    def __repr__(self) -> str:
        return f"<ModelRegistry models={list(self._models.keys())}>"

    def register(self, model: BasePredictor, metrics: dict[str, float] | None = None) -> None:
        with self._lock:
            self._register_unlocked(model, metrics)

    def get(self, model_id: str) -> BasePredictor | None:
        with self._lock:
            return self._models.get(model_id)

    def get_metadata(self, model_id: str) -> ModelMetadata | None:
        with self._lock:
            return self._metadata.get(model_id)

    def log_performance(self, model_id: str, metrics: dict[str, Any]) -> None:
        with self._lock:
            if model_id in self._metadata:
                self._metadata[model_id].performance_log.append({"ts": datetime.now(UTC).isoformat(), **metrics})
                if len(self._metadata[model_id].performance_log) > 200:
                    self._metadata[model_id].performance_log = self._metadata[model_id].performance_log[-200:]

    def replace_if_better(
        self,
        model_id: str,
        candidate: BasePredictor,
        candidate_metrics: dict[str, float],
        compare_metric: str = "sharpe",
        higher_is_better: bool = True,
    ) -> bool:
        """
        If candidate's compare_metric is better than current, register candidate
        as new current and return True.
        """
        with self._lock:
            current = self._models.get(model_id)
            if current is None:
                self._register_unlocked(candidate, candidate_metrics)
                return True
            meta = self._metadata.get(model_id)
            if not meta or not meta.versions:
                self._register_unlocked(candidate, candidate_metrics)
                return True
            current_metric = meta.versions[-1].metrics.get(compare_metric)
            candidate_metric = candidate_metrics.get(compare_metric)
            if current_metric is None:
                self._register_unlocked(candidate, candidate_metrics)
                return True
            if candidate_metric is None:
                return False
            better = candidate_metric > current_metric if higher_is_better else candidate_metric < current_metric
            if better:
                self._register_unlocked(candidate, candidate_metrics)
                logger.info(
                    "Model %s replaced: %s %s -> %s", model_id, compare_metric, current_metric, candidate_metric
                )
                return True
            return False

    def _register_unlocked(self, model: BasePredictor, metrics: dict[str, float] | None = None) -> None:
        """Internal register without acquiring lock (caller must hold lock)."""
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
                created_at=datetime.now(UTC),
                path=getattr(model, "path", ""),
                metrics=metrics or {},
            )
        )
        if len(meta.versions) > 50:
            meta.versions = meta.versions[-50:]

    def deregister(self, model_id: str) -> bool:
        """Remove a model from the registry."""
        with self._lock:
            removed = self._models.pop(model_id, None)
            self._metadata.pop(model_id, None)
            if removed:
                logger.info("Model %s deregistered", model_id)
            return removed is not None

    def list_models(self) -> list[str]:
        """Return list of all registered model IDs."""
        with self._lock:
            return list(self._models.keys())
