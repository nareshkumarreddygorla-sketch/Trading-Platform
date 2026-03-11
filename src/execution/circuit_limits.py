"""
NSE Circuit Limit Awareness.

Pre-order validation against exchange circuit limit bands and market-wide
circuit breakers. Prevents submitting orders that would be rejected by the
exchange or that arrive during a market halt.

Circuit limit bands (NSE):
  - F&O stocks: no individual circuit limits (price-band = operating range only)
  - Group A:  +/- 20%
  - Group B:  +/- 10%
  - Group T:  +/-  5%
  - Group Z:  +/-  5%
  - IPO listing day: +/- 20%

Index-level market-wide circuit breakers (SEBI mandated):
  Level 1: +/- 10%  -> 45 min halt (if triggered before 1:00 PM)
  Level 2: +/- 15%  -> 1h45m halt (if triggered before 1:00 PM)
  Level 3: +/- 20%  -> trading halt for remainder of day

Thread-safe: all state mutations are guarded by a threading.Lock.
"""

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitBand(str, Enum):
    """Individual stock circuit limit bands."""

    NONE = "NONE"  # F&O stocks: no circuit limit
    PCT_2 = "PCT_2"  # +/- 2%
    PCT_5 = "PCT_5"  # +/- 5%
    PCT_10 = "PCT_10"  # +/- 10%
    PCT_20 = "PCT_20"  # +/- 20%


class MarketCircuitLevel(str, Enum):
    """Market-wide circuit breaker levels."""

    NORMAL = "NORMAL"
    LEVEL_1 = "LEVEL_1"  # +/- 10% index move
    LEVEL_2 = "LEVEL_2"  # +/- 15% index move
    LEVEL_3 = "LEVEL_3"  # +/- 20% index move (market closed for day)


BAND_PERCENTAGES: dict[CircuitBand, float] = {
    CircuitBand.NONE: 0.0,
    CircuitBand.PCT_2: 2.0,
    CircuitBand.PCT_5: 5.0,
    CircuitBand.PCT_10: 10.0,
    CircuitBand.PCT_20: 20.0,
}

INDEX_CIRCUIT_THRESHOLDS: list[tuple[float, MarketCircuitLevel]] = [
    (10.0, MarketCircuitLevel.LEVEL_1),
    (15.0, MarketCircuitLevel.LEVEL_2),
    (20.0, MarketCircuitLevel.LEVEL_3),
]


@dataclass
class CircuitCheckResult:
    """Result of a pre-order circuit limit check."""

    allowed: bool
    reason: str = ""
    upper_limit: float = 0.0
    lower_limit: float = 0.0
    band: CircuitBand = CircuitBand.NONE
    market_circuit_level: MarketCircuitLevel = MarketCircuitLevel.NORMAL


class CircuitLimitChecker:
    """
    Pre-order validation against NSE circuit limit bands.

    Usage:
        checker = CircuitLimitChecker()
        checker.set_reference_price("RELIANCE", 2800.0)
        result = checker.check_order("RELIANCE", price=3500.0)
        if not result.allowed:
            reject_order(result.reason)
    """

    def __init__(
        self,
        fno_symbols: set | None = None,
        default_band: CircuitBand = CircuitBand.PCT_20,
    ):
        """
        Args:
            fno_symbols: Set of symbols in F&O segment (no circuit limits).
            default_band: Default circuit band for unknown stocks.
        """
        self._lock = threading.Lock()
        self._fno_symbols: set = fno_symbols or set()
        self._default_band = default_band

        # Previous close / reference prices (set daily at market open)
        self._reference_prices: dict[str, float] = {}

        # Per-symbol band overrides
        self._symbol_bands: dict[str, CircuitBand] = {}

        # Market-wide circuit breaker state
        self._market_circuit_level = MarketCircuitLevel.NORMAL
        self._market_circuit_triggered_at: datetime | None = None

        # Index reference for market-wide circuit breaker
        self._index_reference: dict[str, float] = {}  # e.g. {"NIFTY": 22000.0}

    # --- Configuration ---

    def set_reference_price(self, symbol: str, price: float) -> None:
        """Set previous close / reference price for circuit limit calculation."""
        if price <= 0:
            logger.warning("Invalid reference price for %s: %s", symbol, price)
            return
        with self._lock:
            self._reference_prices[symbol.upper()] = price

    def set_reference_prices_bulk(self, prices: dict[str, float]) -> None:
        """Bulk-set reference prices (e.g. at market open)."""
        with self._lock:
            for sym, px in prices.items():
                if px > 0:
                    self._reference_prices[sym.upper()] = px

    def set_index_reference(self, index_name: str, price: float) -> None:
        """Set index reference for market-wide circuit breaker calculation."""
        if price <= 0:
            return
        with self._lock:
            self._index_reference[index_name.upper()] = price

    def set_symbol_band(self, symbol: str, band: CircuitBand) -> None:
        """Override circuit band for a specific symbol."""
        with self._lock:
            self._symbol_bands[symbol.upper()] = band

    def register_fno_symbols(self, symbols: set) -> None:
        """Register F&O-eligible symbols (no circuit limits)."""
        with self._lock:
            self._fno_symbols = {s.upper() for s in symbols}

    # --- Circuit Limit Checks ---

    def get_band(self, symbol: str) -> CircuitBand:
        """Get the circuit band for a symbol."""
        clean = symbol.upper()
        with self._lock:
            if clean in self._fno_symbols:
                return CircuitBand.NONE
            return self._symbol_bands.get(clean, self._default_band)

    def get_limits(self, symbol: str) -> tuple[float, float]:
        """
        Get upper and lower circuit limits for a symbol.

        Returns:
            (lower_limit, upper_limit). Both 0.0 if no reference price or no circuit.
        """
        clean = symbol.upper()
        with self._lock:
            ref = self._reference_prices.get(clean)
            if ref is None or ref <= 0:
                return 0.0, 0.0

            band = CircuitBand.NONE
            if clean in self._fno_symbols:
                band = CircuitBand.NONE
            else:
                band = self._symbol_bands.get(clean, self._default_band)

            if band == CircuitBand.NONE:
                return 0.0, 0.0

            pct = BAND_PERCENTAGES[band] / 100.0
            lower = round(ref * (1 - pct), 2)
            upper = round(ref * (1 + pct), 2)
            return lower, upper

    def check_order(
        self,
        symbol: str,
        price: float,
        segment: str = "EQ",
    ) -> CircuitCheckResult:
        """
        Pre-order circuit limit check.

        Args:
            symbol: Trading symbol.
            price: Order price (limit price or estimated fill).
            segment: "EQ", "FO", or "IDX".

        Returns:
            CircuitCheckResult with allowed=True if order can proceed.
        """
        clean = symbol.upper()

        with self._lock:
            # Check market-wide circuit breaker first
            if self._market_circuit_level != MarketCircuitLevel.NORMAL:
                return CircuitCheckResult(
                    allowed=False,
                    reason=f"market_wide_circuit_breaker_{self._market_circuit_level.value}",
                    market_circuit_level=self._market_circuit_level,
                )

            # F&O and IDX segments: no individual circuit limits
            if segment in ("FO", "IDX") or clean in self._fno_symbols:
                return CircuitCheckResult(
                    allowed=True,
                    band=CircuitBand.NONE,
                    market_circuit_level=self._market_circuit_level,
                )

            # Get reference price
            ref = self._reference_prices.get(clean)
            if ref is None or ref <= 0:
                # No reference price available — allow with warning
                logger.debug("No reference price for %s; skipping circuit limit check", clean)
                return CircuitCheckResult(
                    allowed=True,
                    reason="no_reference_price",
                    band=self._symbol_bands.get(clean, self._default_band),
                )

            band = self._symbol_bands.get(clean, self._default_band)
            if band == CircuitBand.NONE:
                return CircuitCheckResult(allowed=True, band=band)

            pct = BAND_PERCENTAGES[band] / 100.0
            lower = round(ref * (1 - pct), 2)
            upper = round(ref * (1 + pct), 2)

            if price > upper:
                return CircuitCheckResult(
                    allowed=False,
                    reason=f"price {price:.2f} exceeds upper circuit limit {upper:.2f} "
                    f"(ref={ref:.2f}, band={band.value})",
                    upper_limit=upper,
                    lower_limit=lower,
                    band=band,
                )

            if price < lower:
                return CircuitCheckResult(
                    allowed=False,
                    reason=f"price {price:.2f} below lower circuit limit {lower:.2f} "
                    f"(ref={ref:.2f}, band={band.value})",
                    upper_limit=upper,
                    lower_limit=lower,
                    band=band,
                )

            return CircuitCheckResult(
                allowed=True,
                upper_limit=upper,
                lower_limit=lower,
                band=band,
            )

    # --- Market-Wide Circuit Breaker ---

    def update_index_level(self, index_name: str, current_price: float) -> MarketCircuitLevel:
        """
        Update index price and check for market-wide circuit breaker trigger.

        Args:
            index_name: Index name (e.g. "NIFTY").
            current_price: Current index level.

        Returns:
            Current market circuit level after update.
        """
        clean = index_name.upper()
        with self._lock:
            ref = self._index_reference.get(clean)
            if ref is None or ref <= 0:
                return self._market_circuit_level

            change_pct = abs((current_price - ref) / ref) * 100

            triggered_level = MarketCircuitLevel.NORMAL
            for threshold, level in INDEX_CIRCUIT_THRESHOLDS:
                if change_pct >= threshold:
                    triggered_level = level

            if triggered_level != MarketCircuitLevel.NORMAL:
                if (
                    self._market_circuit_level == MarketCircuitLevel.NORMAL
                    or triggered_level.value > self._market_circuit_level.value
                ):
                    self._market_circuit_level = triggered_level
                    self._market_circuit_triggered_at = datetime.now(UTC)
                    logger.critical(
                        "MARKET-WIDE CIRCUIT BREAKER TRIGGERED: %s index=%s ref=%.2f current=%.2f change=%.2f%%",
                        triggered_level.value,
                        clean,
                        ref,
                        current_price,
                        change_pct,
                    )

            return self._market_circuit_level

    def reset_market_circuit(self) -> None:
        """Reset market-wide circuit breaker (e.g. after halt period ends)."""
        with self._lock:
            self._market_circuit_level = MarketCircuitLevel.NORMAL
            self._market_circuit_triggered_at = None
            logger.info("Market-wide circuit breaker RESET")

    @property
    def market_circuit_level(self) -> MarketCircuitLevel:
        """Current market-wide circuit breaker level."""
        with self._lock:
            return self._market_circuit_level

    @property
    def is_market_halted(self) -> bool:
        """True if market-wide circuit breaker is active."""
        with self._lock:
            return self._market_circuit_level != MarketCircuitLevel.NORMAL
