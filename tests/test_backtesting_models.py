"""Unit tests for backtesting models: slippage, dynamic slippage, and fill model.

Covers:
- SlippageModel: apply() and apply_random() for buy/sell sides
- DynamicSlippageConfig defaults
- DynamicSlippageModel: participation ratio, slippage bps calculation,
  fill price, max order size, edge cases (zero volume, capping)
- FillModelConfig defaults
- FillModel: fill price (open/close, slippage + spread), fill quantity
  (volume participation cap), commission, execute_at_bar_index (full flow,
  latency, out-of-range, zero volume, partial fill)
"""

# ──────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────
from datetime import UTC, datetime, timedelta

import pytest

from src.backtesting.dynamic_slippage import DynamicSlippageConfig, DynamicSlippageModel
from src.backtesting.fill_model import FillModel, FillModelConfig
from src.backtesting.slippage import SlippageModel
from src.core.events import Bar, Exchange


def _bar(
    close: float = 100.0,
    open_: float = 99.0,
    volume: float = 100_000,
    offset_days: int = 0,
) -> Bar:
    return Bar(
        symbol="TEST",
        exchange=Exchange.NSE,
        interval="1d",
        open=open_,
        high=max(close, open_) * 1.01,
        low=min(close, open_) * 0.99,
        close=close,
        volume=volume,
        ts=datetime.now(UTC) - timedelta(days=offset_days),
        source="test",
    )


# ──────────────────────────────────────────────────
# SlippageModel
# ──────────────────────────────────────────────────


class TestSlippageModel:
    def test_buy_slippage_increases_price(self):
        m = SlippageModel(bps=10.0)
        result = m.apply(1000.0, "BUY")
        assert result > 1000.0
        assert result == pytest.approx(1000.0 * (1 + 10 / 10_000), rel=1e-9)

    def test_sell_slippage_decreases_price(self):
        m = SlippageModel(bps=10.0)
        result = m.apply(1000.0, "SELL")
        assert result < 1000.0
        assert result == pytest.approx(1000.0 * (1 - 10 / 10_000), rel=1e-9)

    def test_zero_bps_no_change(self):
        m = SlippageModel(bps=0.0)
        assert m.apply(1000.0, "BUY") == 1000.0
        assert m.apply(1000.0, "SELL") == 1000.0

    def test_case_insensitive_side(self):
        m = SlippageModel(bps=5.0)
        assert m.apply(100.0, "buy") == m.apply(100.0, "BUY")
        assert m.apply(100.0, "sell") == m.apply(100.0, "SELL")

    def test_apply_random_with_seed_deterministic(self):
        m = SlippageModel(bps=5.0)
        r1 = m.apply_random(1000.0, "BUY", seed=42)
        r2 = m.apply_random(1000.0, "BUY", seed=42)
        assert r1 == r2

    def test_apply_random_buy_increases_price(self):
        m = SlippageModel(bps=10.0)
        result = m.apply_random(1000.0, "BUY", seed=42)
        assert result >= 1000.0  # random in [0, 2*bps] always >= 0

    def test_apply_random_sell_decreases_price(self):
        m = SlippageModel(bps=10.0)
        result = m.apply_random(1000.0, "SELL", seed=42)
        assert result <= 1000.0


# ──────────────────────────────────────────────────
# DynamicSlippageModel
# ──────────────────────────────────────────────────


class TestDynamicSlippageConfig:
    def test_defaults(self):
        cfg = DynamicSlippageConfig()
        assert cfg.base_bps == 5.0
        assert cfg.k1 == 10.0
        assert cfg.alpha == 1.5
        assert cfg.max_slippage_bps == 100.0


class TestDynamicSlippageModel:
    def test_participation_ratio_normal(self):
        m = DynamicSlippageModel()
        assert m.participation_ratio(100, 1000) == pytest.approx(0.1)

    def test_participation_ratio_capped_at_one(self):
        m = DynamicSlippageModel()
        assert m.participation_ratio(2000, 1000) == 1.0

    def test_participation_ratio_zero_volume(self):
        m = DynamicSlippageModel()
        assert m.participation_ratio(100, 0) == 1.0

    def test_slippage_bps_increases_with_participation(self):
        m = DynamicSlippageModel()
        low = m.slippage_bps(10, 10_000)  # 0.1% participation
        high = m.slippage_bps(5000, 10_000)  # 50% participation
        assert high > low

    def test_slippage_bps_capped_at_max(self):
        cfg = DynamicSlippageConfig(max_slippage_bps=50.0)
        m = DynamicSlippageModel(cfg)
        result = m.slippage_bps(10_000, 100, spread_bps=100.0, vol_regime_mult=10.0)
        assert result <= 50.0

    def test_slippage_bps_never_negative(self):
        m = DynamicSlippageModel()
        result = m.slippage_bps(0, 100_000, spread_bps=0.0, vol_regime_mult=0.0)
        assert result >= 0.0

    def test_fill_price_buy_higher(self):
        m = DynamicSlippageModel()
        assert m.fill_price(1000.0, "BUY", 10.0) > 1000.0

    def test_fill_price_sell_lower(self):
        m = DynamicSlippageModel()
        assert m.fill_price(1000.0, "SELL", 10.0) < 1000.0

    def test_fill_price_zero_slippage(self):
        m = DynamicSlippageModel()
        assert m.fill_price(1000.0, "BUY", 0.0) == 1000.0

    def test_max_order_size(self):
        cfg = DynamicSlippageConfig(max_volume_participation_pct=10.0)
        m = DynamicSlippageModel(cfg)
        assert m.max_order_size_for_participation(10_000) == 1000.0

    def test_vol_regime_increases_slippage(self):
        m = DynamicSlippageModel()
        calm = m.slippage_bps(100, 10_000, vol_regime_mult=1.0)
        volatile = m.slippage_bps(100, 10_000, vol_regime_mult=5.0)
        assert volatile > calm


# ──────────────────────────────────────────────────
# FillModelConfig
# ──────────────────────────────────────────────────


class TestFillModelConfig:
    def test_defaults(self):
        cfg = FillModelConfig()
        assert cfg.latency_bars == 1
        assert cfg.slippage_bps == 5.0
        assert cfg.spread_bps == 3.0
        assert cfg.max_volume_participation_pct == 10.0
        assert cfg.commission_pct == 0.05
        assert cfg.use_bar_open_for_fill is False


# ──────────────────────────────────────────────────
# FillModel
# ──────────────────────────────────────────────────


class TestFillModel:
    def test_fill_price_buy_uses_close_by_default(self):
        fm = FillModel()
        bar = _bar(close=100.0, open_=99.0)
        price = fm.fill_price(bar, "BUY")
        # Should be based on close (100), with slippage + spread => > 100
        assert price > 100.0

    def test_fill_price_sell_uses_close_by_default(self):
        fm = FillModel()
        bar = _bar(close=100.0, open_=99.0)
        price = fm.fill_price(bar, "SELL")
        # Sell: close minus slippage and spread => < 100
        assert price < 100.0

    def test_fill_price_uses_open_when_configured(self):
        cfg = FillModelConfig(use_bar_open_for_fill=True, slippage_bps=0, spread_bps=0)
        fm = FillModel(cfg)
        bar = _bar(close=100.0, open_=95.0)
        price = fm.fill_price(bar, "BUY")
        assert price == pytest.approx(95.0, rel=1e-6)

    def test_fill_quantity_capped_by_volume(self):
        fm = FillModel()
        bar = _bar(volume=1000)
        # 10% of 1000 = 100 max
        qty = fm.fill_quantity(500, bar)
        assert qty == 100.0

    def test_fill_quantity_below_cap(self):
        fm = FillModel()
        bar = _bar(volume=100_000)
        # 10% of 100k = 10k, request 500 => fills 500
        qty = fm.fill_quantity(500, bar)
        assert qty == 500.0

    def test_fill_quantity_zero_volume(self):
        fm = FillModel()
        bar = _bar(volume=0)
        assert fm.fill_quantity(500, bar) == 0.0

    def test_commission(self):
        fm = FillModel()
        comm = fm.commission(100_000.0)
        # 0.05% of 100k = 50
        assert comm == pytest.approx(50.0)

    def test_execute_at_bar_index_basic(self):
        fm = FillModel(FillModelConfig(latency_bars=1))
        bars = [_bar(close=100.0), _bar(close=105.0)]
        fill_bar, fill_price, fill_qty, comm = fm.execute_at_bar_index(
            signal_bar_index=0, bars=bars, side="BUY", requested_qty=100, price_hint=100.0
        )
        assert fill_bar is not None
        assert fill_price > 0
        assert fill_qty > 0
        assert comm > 0

    def test_execute_out_of_range(self):
        fm = FillModel(FillModelConfig(latency_bars=5))
        bars = [_bar()]
        fill_bar, fill_price, fill_qty, comm = fm.execute_at_bar_index(
            signal_bar_index=0, bars=bars, side="BUY", requested_qty=100, price_hint=100.0
        )
        assert fill_bar is None
        assert fill_price == 0.0
        assert fill_qty == 0.0
        assert comm == 0.0

    def test_execute_zero_volume_bar(self):
        fm = FillModel(FillModelConfig(latency_bars=0))
        bars = [_bar(volume=0)]
        fill_bar, fill_price, fill_qty, comm = fm.execute_at_bar_index(
            signal_bar_index=0, bars=bars, side="BUY", requested_qty=100, price_hint=100.0
        )
        assert fill_bar is not None
        assert fill_qty == 0.0
        assert comm == 0.0

    def test_execute_respects_latency(self):
        cfg = FillModelConfig(latency_bars=2, slippage_bps=0, spread_bps=0)
        fm = FillModel(cfg)
        bars = [_bar(close=100.0), _bar(close=110.0), _bar(close=120.0)]
        fill_bar, fill_price, _, _ = fm.execute_at_bar_index(
            signal_bar_index=0, bars=bars, side="BUY", requested_qty=50, price_hint=100.0
        )
        assert fill_bar is not None
        # Fill at bar index 2 (close=120), with 0 slippage/spread
        assert fill_price == pytest.approx(120.0, rel=1e-6)

    def test_execute_partial_fill_capped(self):
        cfg = FillModelConfig(latency_bars=0, max_volume_participation_pct=5.0)
        fm = FillModel(cfg)
        bars = [_bar(volume=1000)]
        _, _, fill_qty, _ = fm.execute_at_bar_index(
            signal_bar_index=0, bars=bars, side="BUY", requested_qty=200, price_hint=100.0
        )
        # 5% of 1000 = 50
        assert fill_qty == 50.0
