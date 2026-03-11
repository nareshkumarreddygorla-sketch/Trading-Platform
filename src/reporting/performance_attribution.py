"""
Performance Attribution Engine:
Break down PnL by model, strategy, symbol, sector, regime, time-of-day.
Answers the question: "What drove performance and why?"
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Attribution for a single dimension value."""

    dimension: str
    value: str
    total_pnl: float
    total_trades: int
    win_rate: float
    avg_return_pct: float
    sharpe: float
    contribution_pct: float  # % of total portfolio PnL
    avg_holding_bars: float
    avg_confidence: float


class PerformanceAttributionEngine:
    """
    Multi-dimensional performance attribution.

    Dimensions:
    - model_id: Which AI model generated the signal
    - strategy_id: Which strategy was used
    - symbol: Which stock
    - sector: Which sector (NIFTY IT, NIFTY Bank, etc.)
    - regime: Market regime at entry (trending_up, sideways, etc.)
    - time_of_day: Entry time bucket (open, mid-morning, lunch, afternoon, close)
    - exit_reason: Why the trade was closed (stop_loss, take_profit, signal, eod)
    """

    # NSE sector mapping (top stocks)
    SECTOR_MAP = {
        "RELIANCE": "Energy",
        "ONGC": "Energy",
        "IOC": "Energy",
        "BPCL": "Energy",
        "TCS": "IT",
        "INFY": "IT",
        "WIPRO": "IT",
        "HCLTECH": "IT",
        "TECHM": "IT",
        "LTI": "IT",
        "HDFCBANK": "Banking",
        "ICICIBANK": "Banking",
        "SBIN": "Banking",
        "AXISBANK": "Banking",
        "KOTAKBANK": "Banking",
        "INDUSINDBK": "Banking",
        "BANDHANBNK": "Banking",
        "HINDUNILVR": "FMCG",
        "ITC": "FMCG",
        "NESTLEIND": "FMCG",
        "BRITANNIA": "FMCG",
        "BAJFINANCE": "NBFC",
        "BAJAJFINSV": "NBFC",
        "SBILIFE": "NBFC",
        "SUNPHARMA": "Pharma",
        "DRREDDY": "Pharma",
        "CIPLA": "Pharma",
        "DIVISLAB": "Pharma",
        "MARUTI": "Auto",
        "TATAMOTORS": "Auto",
        "M&M": "Auto",
        "BAJAJ-AUTO": "Auto",
        "LT": "Infrastructure",
        "ULTRACEMCO": "Infrastructure",
        "GRASIM": "Infrastructure",
        "TITAN": "Consumer",
        "ASIANPAINT": "Consumer",
        "PIDILITIND": "Consumer",
        "BHARTIARTL": "Telecom",
        "JIOFINANCE": "Telecom",
        "ADANIGREEN": "Adani Group",
        "ADANIPORTS": "Adani Group",
        "ADANIENT": "Adani Group",
    }

    TIME_BUCKETS = {
        (9, 15, 10, 0): "open",
        (10, 0, 11, 30): "mid_morning",
        (11, 30, 13, 0): "lunch",
        (13, 0, 14, 30): "afternoon",
        (14, 30, 15, 30): "close",
    }

    def __init__(self, trade_outcome_repo=None):
        self._repo = trade_outcome_repo

    def compute_attribution(
        self,
        trades: list[dict[str, Any]],
        dimension: str = "model_id",
    ) -> list[AttributionResult]:
        """
        Compute attribution for a single dimension.

        Args:
            trades: List of trade outcome dicts
            dimension: One of model_id, strategy_id, symbol, sector, regime, time_bucket, exit_reason

        Returns:
            Sorted list of AttributionResult (highest PnL first)
        """
        if not trades:
            return []

        # Group trades by dimension
        groups: dict[str, list[dict]] = defaultdict(list)
        for trade in trades:
            key = self._get_dimension_value(trade, dimension)
            if key:
                groups[key].append(trade)

        # Total portfolio PnL for contribution %
        total_portfolio_pnl = sum(t.get("realized_pnl", 0) for t in trades)

        results = []
        for value, group_trades in groups.items():
            pnls = [t.get("realized_pnl", 0) for t in group_trades]
            returns = [t.get("pnl_pct", 0) for t in group_trades]
            wins = sum(1 for p in pnls if p > 0)
            total = len(group_trades)

            # Sharpe approximation
            if len(returns) > 1:
                ret_arr = np.array(returns)
                sharpe = float(np.mean(ret_arr) / (np.std(ret_arr) + 1e-12) * np.sqrt(252))
            else:
                sharpe = 0.0

            total_pnl = sum(pnls)
            contribution = (total_pnl / total_portfolio_pnl * 100) if total_portfolio_pnl != 0 else 0

            results.append(
                AttributionResult(
                    dimension=dimension,
                    value=value,
                    total_pnl=round(total_pnl, 2),
                    total_trades=total,
                    win_rate=round(wins / total * 100, 1) if total > 0 else 0,
                    avg_return_pct=round(float(np.mean(returns)) if returns else 0, 2),
                    sharpe=round(sharpe, 2),
                    contribution_pct=round(contribution, 1),
                    avg_holding_bars=round(float(np.mean([t.get("holding_bars", 0) for t in group_trades])), 1),
                    avg_confidence=round(float(np.mean([t.get("signal_confidence", 0) for t in group_trades])), 3),
                )
            )

        results.sort(key=lambda r: r.total_pnl, reverse=True)
        return results

    def compute_full_attribution(
        self,
        trades: list[dict[str, Any]],
    ) -> dict[str, list[AttributionResult]]:
        """
        Compute attribution across ALL dimensions.

        Returns:
            Dict of dimension -> list of AttributionResult
        """
        dimensions = ["model_id", "strategy_id", "symbol", "sector", "regime", "time_bucket", "exit_reason", "side"]
        return {dim: self.compute_attribution(trades, dim) for dim in dimensions}

    def compute_factor_importance(
        self,
        trades: list[dict[str, Any]],
        top_n: int = 15,
    ) -> list[dict[str, Any]]:
        """
        Analyze which features at entry best predict profitable trades.
        Computes feature importance based on correlation with trade PnL.
        """
        import json

        features_list = []
        pnl_list = []

        for trade in trades:
            feat_json = trade.get("features_at_entry")
            if not feat_json:
                continue
            try:
                feats = json.loads(feat_json) if isinstance(feat_json, str) else feat_json
                features_list.append(feats)
                pnl_list.append(trade.get("pnl_pct", 0))
            except (json.JSONDecodeError, TypeError):
                continue

        if len(features_list) < 10:
            return []

        # Get all feature names
        all_names = set()
        for f in features_list:
            all_names.update(f.keys())

        # Compute correlation of each feature with PnL
        results = []
        pnl_arr = np.array(pnl_list)

        for name in sorted(all_names):
            values = [f.get(name, 0) for f in features_list]
            val_arr = np.array(values, dtype=float)

            if np.std(val_arr) < 1e-12 or np.std(pnl_arr) < 1e-12:
                continue

            corr = float(np.corrcoef(val_arr, pnl_arr)[0, 1])
            if not np.isfinite(corr):
                continue

            results.append(
                {
                    "feature": name,
                    "correlation": round(corr, 4),
                    "abs_correlation": round(abs(corr), 4),
                    "mean_value": round(float(np.mean(val_arr)), 4),
                    "std_value": round(float(np.std(val_arr)), 4),
                }
            )

        results.sort(key=lambda x: x["abs_correlation"], reverse=True)
        return results[:top_n]

    def _get_dimension_value(self, trade: dict, dimension: str) -> str | None:
        """Extract dimension value from a trade dict."""
        if dimension == "sector":
            symbol = trade.get("symbol", "")
            return self.SECTOR_MAP.get(symbol, "Other")

        if dimension == "regime":
            return trade.get("regime_at_entry", "unknown") or "unknown"

        if dimension == "time_bucket":
            entry_time = trade.get("entry_time")
            if entry_time:
                return self._get_time_bucket(entry_time)
            return "unknown"

        return str(trade.get(dimension, "unknown") or "unknown")

    def _get_time_bucket(self, entry_time: Any) -> str:
        """Map entry time to NSE session bucket."""
        try:
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)
            hour = entry_time.hour
            minute = entry_time.minute
            total_minutes = hour * 60 + minute

            if total_minutes < 600:  # Before 10:00
                return "open"
            elif total_minutes < 690:  # Before 11:30
                return "mid_morning"
            elif total_minutes < 780:  # Before 13:00
                return "lunch"
            elif total_minutes < 870:  # Before 14:30
                return "afternoon"
            else:
                return "close"
        except Exception:
            return "unknown"
