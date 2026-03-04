"""
Risk Monitor Agent: continuously monitors portfolio risk,
auto-adjusts exposure, triggers circuit breaker on anomalies.
"""
import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .base import BaseAgent

logger = logging.getLogger(__name__)


class RiskMonitorAgent(BaseAgent):
    """
    Continuously monitors portfolio risk and auto-adjusts exposure.
    - Drawdown monitoring with auto circuit breaker
    - Position correlation analysis
    - Dynamic exposure multiplier adjustment
    - Volatility-based risk scaling
    """

    name = "risk_agent"
    description = "Autonomous portfolio risk monitoring and auto-adjustment"

    def __init__(
        self,
        risk_manager=None,
        circuit_breaker=None,
        kill_switch=None,
        get_positions: Optional[Callable] = None,
        get_bars: Optional[Callable] = None,
        max_portfolio_drawdown_pct: float = 5.0,
        max_position_correlation: float = 0.85,
        volatility_scale_threshold: float = 0.03,
        monitor_interval: float = 30.0,
    ):
        super().__init__()
        self._risk_manager = risk_manager
        self._circuit_breaker = circuit_breaker
        self._kill_switch = kill_switch
        self._get_positions = get_positions
        self._get_bars = get_bars
        self._max_drawdown_pct = max_portfolio_drawdown_pct
        self._max_correlation = max_position_correlation
        self._vol_threshold = volatility_scale_threshold
        self._monitor_interval = monitor_interval
        self._peak_equity: float = 0.0
        self._alerts: List[Dict[str, Any]] = []

    @property
    def interval_seconds(self) -> float:
        return self._monitor_interval

    def _check_drawdown(self, equity: float) -> Optional[Dict]:
        """Check if portfolio drawdown exceeds threshold."""
        if equity > self._peak_equity:
            self._peak_equity = equity

        if self._peak_equity <= 0:
            return None

        drawdown_pct = (self._peak_equity - equity) / self._peak_equity * 100.0

        if drawdown_pct >= self._max_drawdown_pct:
            return {
                "type": "drawdown_breach",
                "severity": "critical" if drawdown_pct >= self._max_drawdown_pct * 1.5 else "warning",
                "drawdown_pct": round(drawdown_pct, 2),
                "peak_equity": self._peak_equity,
                "current_equity": equity,
            }
        return None

    def _check_volatility(self, positions: list) -> Optional[float]:
        """Calculate portfolio volatility and return exposure scale factor."""
        if not positions or not self._get_bars:
            return None

        returns_list = []
        for pos in positions:
            try:
                from src.core.events import Exchange
                bars = self._get_bars(pos.symbol, Exchange.NSE, "1m", 30)
                if bars and len(bars) > 5:
                    closes = np.array([b.close for b in bars])
                    rets = np.diff(closes) / (closes[:-1] + 1e-12)
                    returns_list.append(float(np.std(rets)))
            except Exception:
                continue

        if not returns_list:
            return None

        avg_vol = np.mean(returns_list)

        if avg_vol > self._vol_threshold * 2:
            return 0.3  # Very high vol: scale to 30%
        elif avg_vol > self._vol_threshold:
            return 0.6  # High vol: scale to 60%
        return None  # Normal vol: no change

    def _calculate_position_correlation(self, positions: list) -> Optional[Dict]:
        """Check for dangerous position correlations."""
        if not positions or len(positions) < 2 or not self._get_bars:
            return None

        symbols = [p.symbol for p in positions[:10]]
        returns_matrix = []

        for symbol in symbols:
            try:
                from src.core.events import Exchange
                bars = self._get_bars(symbol, Exchange.NSE, "1m", 60)
                if bars and len(bars) > 20:
                    closes = np.array([b.close for b in bars])
                    rets = np.diff(closes) / (closes[:-1] + 1e-12)
                    returns_matrix.append(rets[-20:])
            except Exception:
                continue

        if len(returns_matrix) < 2:
            return None

        # Ensure equal length
        min_len = min(len(r) for r in returns_matrix)
        returns_matrix = [r[:min_len] for r in returns_matrix]

        try:
            corr_matrix = np.corrcoef(returns_matrix)
            # Find max off-diagonal correlation
            n = corr_matrix.shape[0]
            max_corr = 0.0
            high_corr_pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(corr_matrix[i, j]) > max_corr:
                        max_corr = abs(corr_matrix[i, j])
                    if abs(corr_matrix[i, j]) > self._max_correlation:
                        high_corr_pairs.append((symbols[i], symbols[j], float(corr_matrix[i, j])))

            if high_corr_pairs:
                return {
                    "type": "high_correlation",
                    "severity": "warning",
                    "max_correlation": round(max_corr, 3),
                    "pairs": high_corr_pairs[:5],
                }
        except Exception:
            pass

        return None

    async def run_cycle(self) -> None:
        if not self._risk_manager:
            return

        rm = self._risk_manager
        equity = rm.equity
        positions = list(rm.positions) if rm.positions else []

        # 1. Drawdown check
        drawdown_alert = self._check_drawdown(equity)
        if drawdown_alert:
            self._alerts.append(drawdown_alert)
            logger.warning("RiskAgent: %s — drawdown=%.1f%%",
                           drawdown_alert["severity"], drawdown_alert["drawdown_pct"])

            await self.send_message(
                target="broadcast",
                msg_type="risk_alert",
                payload=drawdown_alert,
            )

            # Auto-trigger circuit breaker on critical drawdown
            if drawdown_alert["severity"] == "critical" and self._kill_switch:
                try:
                    from src.execution.order_entry.kill_switch import KillReason
                    await self._kill_switch.arm(reason=KillReason.MAX_DRAWDOWN, detail="Critical drawdown breach (auto risk agent)")
                    logger.warning("RiskAgent: KILL SWITCH ARMED (critical drawdown)")
                except Exception as e:
                    logger.error("RiskAgent: failed to arm kill switch: %s", e)

        # 2. Volatility-based exposure adjustment
        vol_scale = self._check_volatility(positions)
        if vol_scale is not None and hasattr(rm, "_exposure_multiplier"):
            current_mult = getattr(rm, "_exposure_multiplier", 1.0)
            if vol_scale < current_mult:
                rm._exposure_multiplier = vol_scale
                logger.info("RiskAgent: reduced exposure multiplier to %.2f (high vol)", vol_scale)
                await self.send_message(
                    target="broadcast",
                    msg_type="exposure_adjusted",
                    payload={"new_multiplier": vol_scale, "reason": "high_volatility"},
                )

        # 3. Position correlation check
        corr_alert = self._calculate_position_correlation(positions)
        if corr_alert:
            self._alerts.append(corr_alert)
            await self.send_message(
                target="broadcast",
                msg_type="risk_alert",
                payload=corr_alert,
            )

        # Keep alerts bounded
        if len(self._alerts) > 50:
            self._alerts = self._alerts[-50:]

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["peak_equity"] = self._peak_equity
        status["recent_alerts"] = self._alerts[-5:]
        return status
