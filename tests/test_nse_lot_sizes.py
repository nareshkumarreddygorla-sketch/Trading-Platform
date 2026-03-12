"""Unit tests for src.execution.nse_lot_sizes — NSE lot size validation.

Covers:
- NSELotSizeValidator construction (default and custom config)
- Built-in defaults: index lots (NIFTY, BANKNIFTY), stock lots (RELIANCE, TCS)
- get_lot_size(): equity default, index options, stock F&O, index derivative prefix matching
- validate_quantity(): valid multiples, invalid non-multiples, zero/negative
- round_to_lot_size(): rounding down to nearest valid lot
- validate_and_adjust(): combined validation + adjustment + messaging
- Loading from JSON config
- Properties: index_symbols, stock_symbols
- Thread safety (concurrent reads)
"""

import json
import threading

import pytest

from src.execution.nse_lot_sizes import NSELotSizeValidator

# ──────────────────────────────────────────────────
# Construction and defaults
# ──────────────────────────────────────────────────


class TestConstruction:
    def test_default_construction_loads(self):
        v = NSELotSizeValidator()
        assert v._loaded is True

    def test_builtin_defaults_when_no_config(self, tmp_path):
        # Point to nonexistent config — should fall back to built-in defaults
        v = NSELotSizeValidator(config_path=str(tmp_path / "nonexistent.json"))
        assert v._loaded is True
        assert v._index_lots.get("NIFTY") == 25

    def test_custom_config_loaded(self, tmp_path):
        config = {
            "index_options": {"NIFTY": 50, "BANKNIFTY": 30},
            "stock_futures": {"RELIANCE": 500},
            "equity_default_lot_size": 1,
        }
        config_path = tmp_path / "lot_sizes.json"
        config_path.write_text(json.dumps(config))
        v = NSELotSizeValidator(config_path=str(config_path))
        assert v._index_lots["NIFTY"] == 50
        assert v._stock_lots["RELIANCE"] == 500

    def test_invalid_json_falls_back(self, tmp_path):
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("NOT JSON")
        v = NSELotSizeValidator(config_path=str(bad_path))
        assert v._loaded is True
        # Falls back to built-in defaults
        assert v._index_lots.get("NIFTY") == 25


# ──────────────────────────────────────────────────
# Built-in defaults
# ──────────────────────────────────────────────────


class TestBuiltinDefaults:
    @pytest.fixture
    def v(self, tmp_path) -> NSELotSizeValidator:
        return NSELotSizeValidator(config_path=str(tmp_path / "nonexistent.json"))

    def test_nifty_lot_size(self, v):
        assert v._index_lots["NIFTY"] == 25

    def test_banknifty_lot_size(self, v):
        assert v._index_lots["BANKNIFTY"] == 15

    def test_reliance_lot_size(self, v):
        assert v._stock_lots["RELIANCE"] == 250

    def test_tcs_lot_size(self, v):
        assert v._stock_lots["TCS"] == 175


# ──────────────────────────────────────────────────
# get_lot_size()
# ──────────────────────────────────────────────────


class TestGetLotSize:
    @pytest.fixture
    def v(self, tmp_path) -> NSELotSizeValidator:
        return NSELotSizeValidator(config_path=str(tmp_path / "nonexistent.json"))

    def test_equity_default_is_one(self, v):
        assert v.get_lot_size("SOMESTOCK", "EQ") == 1

    def test_index_option_fo(self, v):
        assert v.get_lot_size("NIFTY", "FO") == 25

    def test_index_option_idx(self, v):
        assert v.get_lot_size("NIFTY", "IDX") == 25

    def test_index_derivative_prefix_matching(self, v):
        """NIFTY25MAR22000CE should match NIFTY prefix -> lot size 25."""
        assert v.get_lot_size("NIFTY25MAR22000CE", "FO") == 25

    def test_banknifty_derivative(self, v):
        assert v.get_lot_size("BANKNIFTY25MAR50000PE", "IDX") == 15

    def test_stock_fo(self, v):
        assert v.get_lot_size("RELIANCE", "FO") == 250

    def test_unknown_stock_fo(self, v):
        assert v.get_lot_size("UNKNOWNSTOCK", "FO") == 1

    def test_case_insensitive(self, v):
        assert v.get_lot_size("reliance", "FO") == 250
        assert v.get_lot_size("nifty", "FO") == 25

    def test_whitespace_stripped(self, v):
        assert v.get_lot_size("  RELIANCE  ", "FO") == 250


# ──────────────────────────────────────────────────
# validate_quantity()
# ──────────────────────────────────────────────────


class TestValidateQuantity:
    @pytest.fixture
    def v(self, tmp_path) -> NSELotSizeValidator:
        return NSELotSizeValidator(config_path=str(tmp_path / "nonexistent.json"))

    def test_valid_multiple(self, v):
        assert v.validate_quantity("RELIANCE", 500, "FO") is True  # 500 = 2 * 250

    def test_single_lot_valid(self, v):
        assert v.validate_quantity("RELIANCE", 250, "FO") is True

    def test_invalid_non_multiple(self, v):
        assert v.validate_quantity("RELIANCE", 300, "FO") is False

    def test_zero_quantity_invalid(self, v):
        assert v.validate_quantity("RELIANCE", 0, "FO") is False

    def test_negative_quantity_invalid(self, v):
        assert v.validate_quantity("RELIANCE", -100, "FO") is False

    def test_equity_any_positive_valid(self, v):
        assert v.validate_quantity("SOMESTOCK", 1, "EQ") is True
        assert v.validate_quantity("SOMESTOCK", 17, "EQ") is True

    def test_nifty_valid(self, v):
        assert v.validate_quantity("NIFTY", 75, "FO") is True  # 3 * 25
        assert v.validate_quantity("NIFTY", 30, "FO") is False


# ──────────────────────────────────────────────────
# round_to_lot_size()
# ──────────────────────────────────────────────────


class TestRoundToLotSize:
    @pytest.fixture
    def v(self, tmp_path) -> NSELotSizeValidator:
        return NSELotSizeValidator(config_path=str(tmp_path / "nonexistent.json"))

    def test_round_down_to_nearest(self, v):
        assert v.round_to_lot_size("RELIANCE", 300, "FO") == 250

    def test_already_valid_no_change(self, v):
        assert v.round_to_lot_size("RELIANCE", 500, "FO") == 500

    def test_below_lot_size_rounds_to_zero(self, v):
        assert v.round_to_lot_size("RELIANCE", 100, "FO") == 0

    def test_zero_returns_zero(self, v):
        assert v.round_to_lot_size("RELIANCE", 0, "FO") == 0

    def test_negative_returns_zero(self, v):
        assert v.round_to_lot_size("RELIANCE", -50, "FO") == 0

    def test_equity_rounds_normally(self, v):
        # Equity lot = 1, so any positive integer is valid
        assert v.round_to_lot_size("SOMESTOCK", 17, "EQ") == 17


# ──────────────────────────────────────────────────
# validate_and_adjust()
# ──────────────────────────────────────────────────


class TestValidateAndAdjust:
    @pytest.fixture
    def v(self, tmp_path) -> NSELotSizeValidator:
        return NSELotSizeValidator(config_path=str(tmp_path / "nonexistent.json"))

    def test_valid_quantity(self, v):
        is_valid, adjusted, msg = v.validate_and_adjust("RELIANCE", 500, "FO")
        assert is_valid is True
        assert adjusted == 500
        assert msg == ""

    def test_invalid_adjusted_to_nearest(self, v):
        is_valid, adjusted, msg = v.validate_and_adjust("RELIANCE", 300, "FO")
        assert is_valid is False
        assert adjusted == 250
        assert "rounded" in msg

    def test_below_lot_size_gives_zero(self, v):
        is_valid, adjusted, msg = v.validate_and_adjust("RELIANCE", 100, "FO")
        assert is_valid is False
        assert adjusted == 0
        assert "below minimum" in msg

    def test_zero_quantity(self, v):
        is_valid, adjusted, msg = v.validate_and_adjust("RELIANCE", 0, "FO")
        assert is_valid is False
        assert adjusted == 0
        assert "positive" in msg


# ──────────────────────────────────────────────────
# Reload
# ──────────────────────────────────────────────────


class TestReload:
    def test_reload_from_updated_config(self, tmp_path):
        config_path = tmp_path / "lots.json"
        config = {
            "index_options": {"NIFTY": 25},
            "stock_futures": {"RELIANCE": 250},
        }
        config_path.write_text(json.dumps(config))
        v = NSELotSizeValidator(config_path=str(config_path))
        assert v.get_lot_size("NIFTY", "FO") == 25

        # Update config and reload
        config["index_options"]["NIFTY"] = 75
        config_path.write_text(json.dumps(config))
        v.reload()
        assert v.get_lot_size("NIFTY", "FO") == 75


# ──────────────────────────────────────────────────
# Properties
# ──────────────────────────────────────────────────


class TestProperties:
    def test_index_symbols_returns_copy(self, tmp_path):
        v = NSELotSizeValidator(config_path=str(tmp_path / "nonexistent.json"))
        idx = v.index_symbols
        assert isinstance(idx, dict)
        assert "NIFTY" in idx
        # Modifying copy doesn't affect validator
        idx["NIFTY"] = 999
        assert v.index_symbols["NIFTY"] == 25

    def test_stock_symbols_returns_copy(self, tmp_path):
        v = NSELotSizeValidator(config_path=str(tmp_path / "nonexistent.json"))
        stocks = v.stock_symbols
        assert isinstance(stocks, dict)
        assert "RELIANCE" in stocks


# ──────────────────────────────────────────────────
# Thread safety
# ──────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_reads(self, tmp_path):
        v = NSELotSizeValidator(config_path=str(tmp_path / "nonexistent.json"))
        errors = []

        def read_lots(count: int):
            try:
                for _ in range(count):
                    v.get_lot_size("RELIANCE", "FO")
                    v.validate_quantity("NIFTY", 25, "FO")
                    v.round_to_lot_size("TCS", 300, "FO")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_lots, args=(50,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
