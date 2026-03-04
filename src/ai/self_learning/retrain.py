"""
Weekly retraining pipeline and auto-replace.
Supports walk-forward: replace only if stability score and replacement rule pass.

Auto-promotion: if new model passes stability check AND Sharpe improvement > threshold,
auto-swap model file (rename current -> backup, new -> current), hot-load in ensemble,
fire alert.
"""
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.ai.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

try:
    from src.ai.walk_forward import stability_score, replacement_rule, WalkForwardConfig
except ImportError:
    stability_score = replacement_rule = WalkForwardConfig = None


@dataclass
class RetrainConfig:
    model_id: str
    compare_metric: str = "sharpe"
    higher_is_better: bool = True
    backtest_min_trades: int = 30
    use_walk_forward: bool = False
    stability_min: float = 0.3
    min_consecutive_positive_windows: int = 3
    dd_limit_pct: float = 10.0
    min_frac_positive_sharpe: float = 0.7
    auto_promote: bool = True  # Auto-promote if Sharpe improvement > threshold
    auto_promote_sharpe_improvement_pct: float = 10.0  # Min 10% Sharpe improvement
    model_dir: str = ""  # Directory where model files live (for file-level swap)


class RetrainPipeline:
    """
    Orchestrates: train new model -> backtest -> compare to current -> replace if better.
    Train and backtest are injectable (callables).
    """

    def __init__(
        self,
        registry: ModelRegistry,
        config: RetrainConfig,
        train_fn: Callable[[], Any],  # returns (model, metrics)
        backtest_fn: Callable[[Any], Dict[str, float]],  # model -> metrics
        walk_forward_backtest_fn: Optional[
            Callable[[Any], Tuple[float, float, List[float], List[float]]]
        ] = None,
        # walk_forward_backtest_fn(model) -> (mean_sharpe, mean_dd, sharpes_per_window, max_dds_per_window)
        on_promote: Optional[Callable[[str, str, Dict[str, float]], None]] = None,
        # on_promote(model_id, version, metrics) — callback when model auto-promoted
    ):
        self.registry = registry
        self.config = config
        self.train_fn = train_fn
        self.backtest_fn = backtest_fn
        self.walk_forward_backtest_fn = walk_forward_backtest_fn
        self.on_promote = on_promote
        self._log: list = []

    def run(self) -> bool:
        """
        Train, backtest, optionally replace. Returns True if model was replaced.
        """
        try:
            model, train_metrics = self.train_fn()
        except Exception as e:
            logger.exception("Retrain train_fn failed: %s", e)
            self._log.append({"ts": datetime.now(timezone.utc).isoformat(), "stage": "train", "error": str(e)})
            return False

        try:
            backtest_metrics = self.backtest_fn(model)
        except Exception as e:
            logger.exception("Retrain backtest_fn failed: %s", e)
            self._log.append({"ts": datetime.now(timezone.utc).isoformat(), "stage": "backtest", "error": str(e)})
            return False

        if backtest_metrics.get("num_trades", 0) < self.config.backtest_min_trades:
            self._log.append({"ts": datetime.now(timezone.utc).isoformat(), "stage": "backtest", "reason": "insufficient_trades"})
            return False

        replaced = False
        if self.config.use_walk_forward and self.walk_forward_backtest_fn is not None and stability_score is not None and replacement_rule is not None:
            try:
                cand_sharpe, cand_dd, sharpes_w, max_dds_w = self.walk_forward_backtest_fn(model)
                stability = stability_score(
                    sharpes_w,
                    max_dds_w,
                    self.config.dd_limit_pct,
                    self.config.min_frac_positive_sharpe,
                )
                consecutive = 0
                for s in reversed(sharpes_w):
                    if s > 0:
                        consecutive += 1
                    else:
                        break
                meta = self.registry.get_metadata(self.config.model_id)
                cur_sharpe = meta.versions[-1].metrics.get("sharpe", 0.0) if meta and meta.versions else 0.0
                cur_dd = meta.versions[-1].metrics.get("max_drawdown_pct", 100.0) if meta and meta.versions else 100.0
                if replacement_rule(
                    current_sharpe=cur_sharpe,
                    current_dd_pct=cur_dd,
                    candidate_sharpe=cand_sharpe,
                    candidate_dd_pct=cand_dd,
                    candidate_stability=stability,
                    candidate_consecutive_positive=consecutive,
                    config=WalkForwardConfig(stability_min=self.config.stability_min, min_consecutive_positive_windows=self.config.min_consecutive_positive_windows) if WalkForwardConfig else None,
                ):
                    backtest_metrics["sharpe"] = cand_sharpe
                    backtest_metrics["max_drawdown_pct"] = cand_dd
                    replaced = self.registry.replace_if_better(
                        self.config.model_id, model, backtest_metrics,
                        compare_metric=self.config.compare_metric,
                        higher_is_better=self.config.higher_is_better,
                    )
                self._log.append({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "stage": "walk_forward_replace",
                    "replaced": replaced,
                    "stability": stability,
                    "candidate_sharpe": cand_sharpe,
                    "candidate_dd": cand_dd,
                })
            except Exception as e:
                logger.exception("Walk-forward replace failed: %s", e)
                self._log.append({"ts": datetime.now(timezone.utc).isoformat(), "stage": "walk_forward", "error": str(e)})
        else:
            replaced = self.registry.replace_if_better(
                self.config.model_id,
                model,
                backtest_metrics,
                compare_metric=self.config.compare_metric,
                higher_is_better=self.config.higher_is_better,
            )
        self._log.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "stage": "replace",
            "replaced": replaced,
            "metrics": backtest_metrics,
        })

        # Auto-promotion: if replaced AND Sharpe improvement > threshold
        if replaced and self.config.auto_promote:
            try:
                self._auto_promote(model, backtest_metrics)
            except Exception as e:
                logger.exception("Auto-promote failed for %s: %s", self.config.model_id, e)
                self._log.append({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "stage": "auto_promote",
                    "error": str(e),
                })

        return replaced

    def _auto_promote(self, model: Any, metrics: Dict[str, float]) -> None:
        """
        Auto-promote model if Sharpe improvement exceeds threshold.

        Steps:
          1. Compare new Sharpe to current model's Sharpe.
          2. If improvement > config.auto_promote_sharpe_improvement_pct:
             a. Backup current model file (rename current -> .backup)
             b. Save new model as current
             c. Fire on_promote callback (for hot-load + alerts)
        """
        # Get current model's Sharpe from registry metadata
        meta = self.registry.get_metadata(self.config.model_id)
        current_sharpe = 0.0
        if meta and meta.versions:
            current_sharpe = meta.versions[-1].metrics.get("sharpe", 0.0)

        new_sharpe = metrics.get("sharpe", 0.0)

        # Calculate improvement
        if current_sharpe > 0:
            improvement_pct = ((new_sharpe - current_sharpe) / abs(current_sharpe)) * 100.0
        elif new_sharpe > 0:
            improvement_pct = 100.0  # Any positive Sharpe from zero is 100% improvement
        else:
            improvement_pct = 0.0

        if improvement_pct < self.config.auto_promote_sharpe_improvement_pct:
            logger.info(
                "Auto-promote skipped for %s: Sharpe improvement %.1f%% < threshold %.1f%%",
                self.config.model_id,
                improvement_pct,
                self.config.auto_promote_sharpe_improvement_pct,
            )
            self._log.append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "stage": "auto_promote",
                "action": "skipped",
                "sharpe_improvement_pct": improvement_pct,
            })
            return

        # File-level swap if model_dir configured
        if self.config.model_dir:
            model_file = os.path.join(self.config.model_dir, f"{self.config.model_id}.joblib")
            backup_file = os.path.join(self.config.model_dir, f"{self.config.model_id}.backup.joblib")

            if os.path.exists(model_file):
                try:
                    shutil.copy2(model_file, backup_file)
                    logger.info("Model backed up: %s -> %s", model_file, backup_file)
                except Exception as e:
                    logger.warning("Model backup failed: %s", e)

            # Save new model
            try:
                import joblib
                joblib.dump(model, model_file)
                logger.info("New model saved: %s", model_file)
            except ImportError:
                logger.debug("joblib not available for model file save")
            except Exception as e:
                logger.warning("Model save failed: %s", e)

        # Compute new version string
        version_num = 1
        if meta and meta.versions:
            try:
                prev_v = meta.versions[-1].version_str
                version_num = int(prev_v.replace("v", "")) + 1
            except (ValueError, AttributeError):
                pass
        new_version = f"v{version_num}"

        logger.warning(
            "AUTO-PROMOTED %s: %s -> %s (Sharpe %.4f -> %.4f, +%.1f%%)",
            self.config.model_id,
            meta.versions[-1].version_str if meta and meta.versions else "v0",
            new_version,
            current_sharpe,
            new_sharpe,
            improvement_pct,
        )

        self._log.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "stage": "auto_promote",
            "action": "promoted",
            "model_id": self.config.model_id,
            "old_sharpe": current_sharpe,
            "new_sharpe": new_sharpe,
            "improvement_pct": improvement_pct,
            "new_version": new_version,
        })

        # Fire callback for hot-load + alerts
        if self.on_promote:
            try:
                self.on_promote(self.config.model_id, new_version, metrics)
            except Exception as e:
                logger.exception("on_promote callback failed: %s", e)
