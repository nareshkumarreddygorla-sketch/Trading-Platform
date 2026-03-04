"""
Self-learning orchestrator: schedule drift check, trigger retrain, run backtest, replace.
"""
import logging
from typing import Callable, List, Optional

from .drift import ConceptDriftDetector, DataDistributionMonitor
from .retrain import RetrainPipeline, RetrainConfig

logger = logging.getLogger(__name__)


class SelfLearningOrchestrator:
    """
    Weekly (or on-event) flow: check drift -> if drift, run retrain pipeline.
    Can be triggered by scheduler or API.
    """

    def __init__(
        self,
        drift_detector: ConceptDriftDetector,
        distribution_monitor: DataDistributionMonitor,
        retrain_pipelines: List[RetrainPipeline],
        on_retrain_complete: Optional[Callable[[str, bool], None]] = None,
    ):
        self.drift_detector = drift_detector
        self.distribution_monitor = distribution_monitor
        self.retrain_pipelines = retrain_pipelines
        self.on_retrain_complete = on_retrain_complete

    def check_drift(self, features: dict) -> bool:
        """Return True if drift detected."""
        drifted, reason = self.drift_detector.detect(features)
        if drifted:
            logger.warning("Concept drift detected: %s", reason)
        return drifted

    def run_retrain_all(self) -> dict:
        """Run all retrain pipelines; return model_id -> replaced."""
        results = {}
        for pipeline in self.retrain_pipelines:
            model_id = pipeline.config.model_id
            try:
                replaced = pipeline.run()
                results[model_id] = replaced
                if self.on_retrain_complete:
                    self.on_retrain_complete(model_id, replaced)
            except Exception as e:
                logger.exception("Retrain pipeline %s failed: %s", model_id, e)
                results[model_id] = False
        return results

    async def run_cycle(self, current_features: dict) -> dict:
        """
        One self-learning cycle: monitor distribution, check drift, optionally retrain.
        """
        self.distribution_monitor.add(current_features)
        if self.check_drift(current_features):
            import asyncio
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.run_retrain_all)
        return {}
