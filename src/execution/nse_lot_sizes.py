"""
NSE Lot Size Validation.

Validates order quantities against NSE F&O lot sizes.
- Equity segment: lot size = 1 (any integer quantity valid).
- Index options (NIFTY, BANKNIFTY, etc.): fixed lot sizes per contract.
- Stock F&O: lot sizes vary per stock, loaded from deploy/nse_lot_sizes.json.

Thread-safe: uses a threading.Lock for hot-reload of lot size data.
"""

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Default config path (repo-relative)
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "deploy" / "nse_lot_sizes.json"


class NSELotSizeValidator:
    """
    Validates and rounds order quantities to NSE lot sizes.

    Usage:
        validator = NSELotSizeValidator()
        valid = validator.validate_quantity("RELIANCE", 500)   # True (500 is 2 * 250)
        valid = validator.validate_quantity("RELIANCE", 300)   # False (not multiple of 250)
        rounded = validator.round_to_lot_size("RELIANCE", 300) # 250
    """

    def __init__(self, config_path: str | None = None):
        self._lock = threading.Lock()
        self._config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self._index_lots: dict[str, int] = {}
        self._stock_lots: dict[str, int] = {}
        self._equity_default: int = 1
        self._loaded = False
        self._load()

    def _load(self) -> None:
        """Load lot sizes from JSON config. Thread-safe."""
        with self._lock:
            try:
                if not self._config_path.exists():
                    logger.warning(
                        "NSE lot sizes config not found at %s; using built-in defaults",
                        self._config_path,
                    )
                    self._use_builtin_defaults()
                    self._loaded = True
                    return

                with open(self._config_path) as f:
                    data = json.load(f)

                self._index_lots = {
                    k.upper(): int(v) for k, v in data.get("index_options", {}).items() if not k.startswith("_")
                }
                self._stock_lots = {
                    k.upper(): int(v) for k, v in data.get("stock_futures", {}).items() if not k.startswith("_")
                }
                self._equity_default = int(data.get("equity_default_lot_size", 1))
                self._loaded = True

                logger.info(
                    "Loaded NSE lot sizes: %d index, %d stock F&O entries",
                    len(self._index_lots),
                    len(self._stock_lots),
                )
            except Exception as e:
                logger.error("Failed to load NSE lot sizes: %s; using built-in defaults", e)
                self._use_builtin_defaults()
                self._loaded = True

    def _use_builtin_defaults(self) -> None:
        """Hardcoded fallback for critical lot sizes."""
        self._index_lots = {
            "NIFTY": 25,
            "BANKNIFTY": 15,
            "FINNIFTY": 25,
            "MIDCPNIFTY": 50,
        }
        self._stock_lots = {
            "RELIANCE": 250,
            "TCS": 175,
            "INFY": 300,
            "HDFCBANK": 550,
            "ICICIBANK": 700,
            "SBIN": 750,
            "HINDUNILVR": 300,
            "ITC": 1600,
            "BHARTIARTL": 475,
            "KOTAKBANK": 400,
            "LT": 150,
            "AXISBANK": 625,
            "BAJFINANCE": 125,
        }
        self._equity_default = 1

    def reload(self) -> None:
        """Hot-reload lot sizes from config file. Thread-safe."""
        self._load()
        logger.info("NSE lot sizes reloaded")

    def get_lot_size(self, symbol: str, segment: str = "EQ") -> int:
        """
        Get lot size for a symbol.

        Args:
            symbol: Trading symbol (e.g. "RELIANCE", "NIFTY", "NIFTY25MAR22000CE").
            segment: "EQ" for equity, "FO" for futures & options, "IDX" for index options.

        Returns:
            Lot size (integer). Returns 1 for unknown equity symbols.
        """
        clean = symbol.upper().strip()

        with self._lock:
            # Index options: match prefix (e.g. "NIFTY25MAR22000CE" -> "NIFTY")
            if segment in ("FO", "IDX") or self._is_index_symbol(clean):
                for idx_name, lot in self._index_lots.items():
                    if clean.startswith(idx_name):
                        return lot

            # Stock F&O
            if segment == "FO" or clean in self._stock_lots:
                return self._stock_lots.get(clean, self._equity_default)

            # Equity segment: lot = 1
            return self._equity_default

    def _is_index_symbol(self, symbol: str) -> bool:
        """Check if symbol looks like an index derivative (NIFTY*, BANKNIFTY*, etc.)."""
        for idx_name in self._index_lots:
            if symbol.startswith(idx_name) and len(symbol) > len(idx_name):
                # Has suffix (e.g. option chain strike/expiry) -> index derivative
                return True
        return False

    def validate_quantity(self, symbol: str, quantity: int, segment: str = "EQ") -> bool:
        """
        Validate that quantity is a valid multiple of the lot size.

        Args:
            symbol: Trading symbol.
            quantity: Order quantity.
            segment: Market segment ("EQ", "FO", "IDX").

        Returns:
            True if quantity is a positive multiple of the lot size.
        """
        if quantity <= 0:
            return False
        lot = self.get_lot_size(symbol, segment)
        return quantity % lot == 0

    def round_to_lot_size(self, symbol: str, quantity: int, segment: str = "EQ") -> int:
        """
        Round down to the nearest valid lot size multiple.

        Args:
            symbol: Trading symbol.
            quantity: Desired order quantity.
            segment: Market segment ("EQ", "FO", "IDX").

        Returns:
            Rounded quantity (may be 0 if quantity < lot_size).
        """
        if quantity <= 0:
            return 0
        lot = self.get_lot_size(symbol, segment)
        return (quantity // lot) * lot

    def validate_and_adjust(self, symbol: str, quantity: int, segment: str = "EQ") -> tuple[bool, int, str]:
        """
        Validate quantity and return adjusted quantity if invalid.

        Returns:
            (is_valid, adjusted_quantity, message)
        """
        lot = self.get_lot_size(symbol, segment)
        if quantity <= 0:
            return False, 0, f"quantity must be positive (lot_size={lot})"

        if quantity % lot == 0:
            return True, quantity, ""

        adjusted = self.round_to_lot_size(symbol, quantity, segment)
        if adjusted == 0:
            return False, 0, f"quantity {quantity} below minimum lot size {lot}"

        return (
            False,
            adjusted,
            f"quantity {quantity} not a multiple of lot size {lot}; rounded to {adjusted}",
        )

    @property
    def index_symbols(self) -> dict[str, int]:
        """Return a copy of loaded index lot sizes."""
        with self._lock:
            return dict(self._index_lots)

    @property
    def stock_symbols(self) -> dict[str, int]:
        """Return a copy of loaded stock F&O lot sizes."""
        with self._lock:
            return dict(self._stock_lots)
