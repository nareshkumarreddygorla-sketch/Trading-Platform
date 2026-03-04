"""
A/B model testing framework (shadow mode).

Run a shadow ensemble alongside the production ensemble.  The shadow model
receives real features and generates predictions, but those predictions are
*never* routed to execution -- they are logged for offline comparison only.

After a configurable minimum observation window, the framework computes
per-day Sharpe ratio, Information Coefficient (IC), and maximum drawdown for
both production and shadow.  If the shadow outperforms production on *all three*
metrics for at least ``min_comparison_days`` consecutive days, automatic
promotion is recommended.

Thread-safe: all signal-recording methods use a reentrant lock so that
concurrent strategies or data feeds can safely push results.
"""

import logging
import math
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..models.base import PredictionOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Daily snapshot kept for audit / visualisation
# ---------------------------------------------------------------------------

@dataclass
class _DailyRecord:
    """Internal per-day metrics for one model (production or shadow)."""
    date: str
    signals: List[Tuple[float, float]] = field(default_factory=list)  # (predicted, actual)
    cumulative_return: float = 0.0


# ---------------------------------------------------------------------------
# Public comparison result
# ---------------------------------------------------------------------------

@dataclass
class ShadowComparison:
    """
    Side-by-side performance snapshot of production vs. shadow ensembles.

    Attributes:
        production_sharpe: Annualised Sharpe ratio of production signals.
        shadow_sharpe: Annualised Sharpe ratio of shadow signals.
        production_ic: Mean Information Coefficient (Pearson correlation of
            predicted direction vs. actual return) for production.
        shadow_ic: Mean IC for shadow.
        production_drawdown: Maximum drawdown (as a positive fraction, e.g.
            0.05 = 5%) observed in the production cumulative-return curve.
        shadow_drawdown: Maximum drawdown for shadow.
        days_compared: Number of calendar days with data for both models.
        min_days_required: Minimum days that must pass before promotion is
            considered.
    """

    production_sharpe: float
    shadow_sharpe: float
    production_ic: float
    shadow_ic: float
    production_drawdown: float
    shadow_drawdown: float
    days_compared: int
    min_days_required: int = 10

    @property
    def should_promote(self) -> bool:
        """
        Return ``True`` when the shadow model outperforms production on
        **all three** core metrics *and* the minimum observation window has
        been met.

        Metrics (shadow must be strictly better):
            1. Higher Sharpe ratio.
            2. Higher Information Coefficient.
            3. Lower maximum drawdown.
        """
        if self.days_compared < self.min_days_required:
            return False
        sharpe_better = self.shadow_sharpe > self.production_sharpe
        ic_better = self.shadow_ic > self.production_ic
        drawdown_better = self.shadow_drawdown < self.production_drawdown
        return sharpe_better and ic_better and drawdown_better


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class ShadowModelRunner:
    """
    Orchestrates A/B (shadow-mode) testing between two
    :class:`~src.ai.models.ensemble.EnsembleEngine` instances.

    The *production* ensemble drives live trading.  The *shadow* ensemble
    receives identical features and produces predictions that are recorded but
    never executed.  After enough data has accumulated the two can be compared
    and, if the shadow is consistently superior, it can be promoted to
    production in a single atomic swap.

    Parameters:
        production_ensemble: The ensemble currently driving live execution.
        shadow_ensemble: The candidate ensemble under evaluation.
        min_comparison_days: Minimum calendar days of parallel observation
            before promotion is considered.
    """

    def __init__(
        self,
        production_ensemble: Any,
        shadow_ensemble: Any,
        min_comparison_days: int = 10,
    ) -> None:
        self._production = production_ensemble
        self._shadow = shadow_ensemble
        self._min_days = min_comparison_days

        # Per-date signal stores: date_str -> list[(predicted_direction, actual_return)]
        self._prod_signals: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self._shadow_signals: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

        # Thread safety
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Signal recording
    # ------------------------------------------------------------------

    def record_production_signal(
        self,
        symbol: str,
        predicted_direction: float,
        actual_return: float,
    ) -> None:
        """
        Log a production prediction/outcome pair for today's date.

        Args:
            symbol: Ticker (used for logging; signals are pooled per day).
            predicted_direction: Production model's directional signal
                (positive = predicted up).
            actual_return: Realised return for the same horizon.
        """
        today = self._today_str()
        with self._lock:
            self._prod_signals[today].append(
                (float(predicted_direction), float(actual_return))
            )
        logger.debug(
            "Prod signal recorded | %s | pred=%.4f actual=%.4f",
            symbol, predicted_direction, actual_return,
        )

    def record_shadow_signal(
        self,
        symbol: str,
        predicted_direction: float,
        actual_return: float,
    ) -> None:
        """
        Log a shadow prediction/outcome pair for today's date.

        Args:
            symbol: Ticker (used for logging; signals are pooled per day).
            predicted_direction: Shadow model's directional signal.
            actual_return: Realised return for the same horizon.
        """
        today = self._today_str()
        with self._lock:
            self._shadow_signals[today].append(
                (float(predicted_direction), float(actual_return))
            )
        logger.debug(
            "Shadow signal recorded | %s | pred=%.4f actual=%.4f",
            symbol, predicted_direction, actual_return,
        )

    # ------------------------------------------------------------------
    # Shadow prediction (log-only, never traded)
    # ------------------------------------------------------------------

    def run_shadow_prediction(
        self,
        features: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> PredictionOutput:
        """
        Run the shadow ensemble on *features* and return its prediction.

        The result is logged internally but is **never forwarded** to the
        execution layer.  Callers can inspect the returned
        :class:`PredictionOutput` for monitoring dashboards, but it must not
        influence position sizing or order placement.
        """
        prediction = self._shadow.predict(features, context)
        logger.info(
            "Shadow prediction | prob_up=%.4f exp_ret=%.6f conf=%.4f",
            prediction.prob_up,
            prediction.expected_return,
            prediction.confidence,
        )
        return prediction

    # ------------------------------------------------------------------
    # Comparison / metrics
    # ------------------------------------------------------------------

    def get_comparison(self) -> ShadowComparison:
        """
        Build a :class:`ShadowComparison` from all recorded signals.

        Only dates present in **both** production and shadow signal stores are
        included so that the comparison is apples-to-apples.
        """
        with self._lock:
            common_dates = sorted(
                set(self._prod_signals.keys()) & set(self._shadow_signals.keys())
            )

            prod_daily_ret: List[float] = []
            shadow_daily_ret: List[float] = []
            prod_daily_ic: List[float] = []
            shadow_daily_ic: List[float] = []

            for d in common_dates:
                p_signals = self._prod_signals[d]
                s_signals = self._shadow_signals[d]

                prod_daily_ret.append(self._daily_return(p_signals))
                shadow_daily_ret.append(self._daily_return(s_signals))

                prod_daily_ic.append(self._daily_ic(p_signals))
                shadow_daily_ic.append(self._daily_ic(s_signals))

        n = len(common_dates)

        prod_sharpe = self._sharpe(prod_daily_ret)
        shadow_sharpe = self._sharpe(shadow_daily_ret)
        prod_ic = float(np.mean(prod_daily_ic)) if prod_daily_ic else 0.0
        shadow_ic = float(np.mean(shadow_daily_ic)) if shadow_daily_ic else 0.0
        prod_dd = self._max_drawdown(prod_daily_ret)
        shadow_dd = self._max_drawdown(shadow_daily_ret)

        return ShadowComparison(
            production_sharpe=prod_sharpe,
            shadow_sharpe=shadow_sharpe,
            production_ic=prod_ic,
            shadow_ic=shadow_ic,
            production_drawdown=prod_dd,
            shadow_drawdown=shadow_dd,
            days_compared=n,
            min_days_required=self._min_days,
        )

    def should_auto_promote(self) -> bool:
        """
        Convenience wrapper: ``True`` when the shadow beats production on
        Sharpe, IC, **and** drawdown for at least ``min_comparison_days``.
        """
        return self.get_comparison().should_promote

    # ------------------------------------------------------------------
    # Promotion / rejection
    # ------------------------------------------------------------------

    def promote_shadow(self) -> bool:
        """
        Atomically swap the shadow ensemble into the production slot.

        Returns:
            ``True`` if promotion occurred, ``False`` if there is no shadow
            ensemble to promote.
        """
        with self._lock:
            if self._shadow is None:
                logger.warning("promote_shadow called but no shadow ensemble is set")
                return False

            old_production = self._production
            self._production = self._shadow
            self._shadow = None

            # Clear signal history -- the promoted model starts fresh as
            # production, old data is no longer meaningful.
            self._prod_signals.clear()
            self._shadow_signals.clear()

        logger.info(
            "Shadow promoted to production (old production discarded: %s)",
            getattr(old_production, "model_ids", "unknown"),
        )
        return True

    def reject_shadow(self) -> None:
        """
        Discard the current shadow ensemble.  Production is unaffected.
        """
        with self._lock:
            self._shadow = None
            self._shadow_signals.clear()
        logger.info("Shadow ensemble rejected and discarded")

    # ------------------------------------------------------------------
    # Daily metrics for dashboards
    # ------------------------------------------------------------------

    def get_daily_metrics(self) -> List[Dict[str, Any]]:
        """
        Return a list of per-day comparison dicts for charting / audit.

        Each entry contains:
            - ``date``: ISO date string.
            - ``production_return``: Mean directional return for production.
            - ``shadow_return``: Mean directional return for shadow.
            - ``production_ic``: Daily IC for production.
            - ``shadow_ic``: Daily IC for shadow.
            - ``production_cumulative``: Cumulative return for production.
            - ``shadow_cumulative``: Cumulative return for shadow.
        """
        with self._lock:
            common_dates = sorted(
                set(self._prod_signals.keys()) & set(self._shadow_signals.keys())
            )

            results: List[Dict[str, Any]] = []
            prod_cum = 0.0
            shadow_cum = 0.0

            for d in common_dates:
                p_ret = self._daily_return(self._prod_signals[d])
                s_ret = self._daily_return(self._shadow_signals[d])
                prod_cum += p_ret
                shadow_cum += s_ret

                results.append({
                    "date": d,
                    "production_return": p_ret,
                    "shadow_return": s_ret,
                    "production_ic": self._daily_ic(self._prod_signals[d]),
                    "shadow_ic": self._daily_ic(self._shadow_signals[d]),
                    "production_cumulative": prod_cum,
                    "shadow_cumulative": shadow_cum,
                })

        return results

    # ------------------------------------------------------------------
    # Accessor helpers
    # ------------------------------------------------------------------

    @property
    def production_ensemble(self) -> Any:
        """The currently active production ensemble."""
        return self._production

    @property
    def shadow_ensemble(self) -> Optional[Any]:
        """The shadow ensemble under evaluation (or ``None``)."""
        return self._shadow

    # ------------------------------------------------------------------
    # Internal calculations
    # ------------------------------------------------------------------

    @staticmethod
    def _today_str() -> str:
        """ISO date string for the current UTC day."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _daily_return(signals: List[Tuple[float, float]]) -> float:
        """
        Compute the mean *directional return* for a day's signals.

        Directional return = sign(predicted) * actual_return.  This measures
        how much money the model would have made if you followed its
        direction signal with equal dollar allocation per trade.
        """
        if not signals:
            return 0.0
        directional = [
            np.sign(pred) * actual for pred, actual in signals if pred != 0.0
        ]
        return float(np.mean(directional)) if directional else 0.0

    @staticmethod
    def _daily_ic(signals: List[Tuple[float, float]]) -> float:
        """
        Pearson correlation between predicted direction and actual return
        for a single day's signals.  Returns 0.0 when data is insufficient
        or variance is zero.
        """
        if len(signals) < 2:
            return 0.0
        preds = np.array([p for p, _a in signals], dtype=np.float64)
        actuals = np.array([a for _p, a in signals], dtype=np.float64)
        if np.std(preds) < 1e-12 or np.std(actuals) < 1e-12:
            return 0.0
        corr = np.corrcoef(preds, actuals)[0, 1]
        return float(corr) if math.isfinite(corr) else 0.0

    @staticmethod
    def _sharpe(daily_returns: List[float], annualisation_factor: float = 252.0) -> float:
        """
        Annualised Sharpe ratio from a list of daily returns.

        Uses the standard ``mean / std * sqrt(N)`` formulation with zero
        risk-free rate.  Returns 0.0 for insufficient or zero-variance data.
        """
        if len(daily_returns) < 2:
            return 0.0
        arr = np.array(daily_returns, dtype=np.float64)
        std = float(np.std(arr, ddof=1))
        if std < 1e-12:
            return 0.0
        mean = float(np.mean(arr))
        return (mean / std) * math.sqrt(annualisation_factor)

    @staticmethod
    def _max_drawdown(daily_returns: List[float]) -> float:
        """
        Maximum drawdown as a positive fraction from peak cumulative return.

        Example: if the cumulative return curve goes from +10% to +4%, the
        drawdown is 0.06 (6%).  Returns 0.0 when there are no returns or
        the curve never declines.
        """
        if not daily_returns:
            return 0.0
        cum = np.cumsum(daily_returns)
        running_max = np.maximum.accumulate(cum)
        drawdowns = running_max - cum  # always >= 0
        max_dd = float(np.max(drawdowns))
        return max(max_dd, 0.0)
