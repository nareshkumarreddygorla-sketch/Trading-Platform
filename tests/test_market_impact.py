"""Unit tests for src.execution.market_impact — Almgren-Chriss model.

Covers:
- ImpactEstimate: dataclass defaults, as_dict serialization
- NSEImpactConfig: defaults
- MarketImpactModel construction (default and config-based)
- estimate(): basic impact, edge cases (zero qty, zero price),
  participation rate, scaling behavior
- check_alpha_sufficient(): alpha > cost passes, alpha < cost reduces qty,
  binary search for optimal quantity
- should_trade(): convenience function
"""

import math

import pytest

from src.execution.market_impact import (
    ImpactEstimate,
    MarketImpactModel,
    NSEImpactConfig,
)

# ──────────────────────────────────────────────────
# ImpactEstimate
# ──────────────────────────────────────────────────


class TestImpactEstimate:
    def test_defaults(self):
        ie = ImpactEstimate()
        assert ie.temporary_impact_bps == 0.0
        assert ie.permanent_impact_bps == 0.0
        assert ie.total_impact_bps == 0.0
        assert ie.alpha_sufficient is True
        assert ie.recommended_qty == 0

    def test_as_dict(self):
        ie = ImpactEstimate(
            temporary_impact_bps=12.345,
            permanent_impact_bps=6.789,
            total_impact_bps=19.134,
            transaction_cost_bps=5.0,
            total_cost_bps=24.134,
            participation_rate=0.12345,
            recommended_qty=500,
            alpha_sufficient=True,
        )
        d = ie.as_dict()
        assert d["temporary_impact_bps"] == 12.35
        assert d["permanent_impact_bps"] == 6.79
        assert d["participation_rate"] == 0.1235
        assert d["recommended_qty"] == 500
        assert d["alpha_sufficient"] is True


# ──────────────────────────────────────────────────
# NSEImpactConfig
# ──────────────────────────────────────────────────


class TestNSEImpactConfig:
    def test_defaults(self):
        cfg = NSEImpactConfig()
        assert cfg.gamma == 0.25
        assert cfg.min_adv == 10_000
        assert cfg.default_sigma == 0.02
        assert cfg.transaction_cost_bps == 12.0


# ──────────────────────────────────────────────────
# MarketImpactModel construction
# ──────────────────────────────────────────────────


class TestMarketImpactModelConstruction:
    def test_default_construction(self):
        m = MarketImpactModel()
        assert m.gamma == 0.25
        assert m.min_adv == 10_000

    def test_custom_gamma(self):
        m = MarketImpactModel(gamma=0.5)
        assert m.gamma == 0.5

    def test_config_based_construction(self):
        cfg = NSEImpactConfig(gamma=0.3, min_adv=50_000)
        m = MarketImpactModel(config=cfg)
        assert m.gamma == 0.3
        assert m.min_adv == 50_000


# ──────────────────────────────────────────────────
# estimate()
# ──────────────────────────────────────────────────


class TestEstimate:
    def test_basic_impact(self):
        m = MarketImpactModel(gamma=0.25)
        result = m.estimate(quantity=1000, price=100.0, adv=100_000, sigma=0.02)
        assert result.temporary_impact_bps > 0
        assert result.permanent_impact_bps > 0
        assert result.total_impact_bps == pytest.approx(result.temporary_impact_bps + result.permanent_impact_bps)

    def test_zero_quantity_returns_empty(self):
        m = MarketImpactModel()
        result = m.estimate(quantity=0, price=100.0, adv=100_000)
        assert result.temporary_impact_bps == 0.0
        assert result.permanent_impact_bps == 0.0

    def test_zero_price_returns_empty(self):
        m = MarketImpactModel()
        result = m.estimate(quantity=1000, price=0.0, adv=100_000)
        assert result.temporary_impact_bps == 0.0

    def test_participation_rate_calculation(self):
        m = MarketImpactModel()
        result = m.estimate(quantity=5000, price=100.0, adv=100_000)
        assert result.participation_rate == pytest.approx(0.05)

    def test_adv_floor_applied(self):
        m = MarketImpactModel(min_adv=10_000)
        result = m.estimate(quantity=100, price=100.0, adv=5)
        # ADV clamped to 10_000 => participation = 100/10_000 = 0.01
        assert result.participation_rate == pytest.approx(0.01)

    def test_impact_increases_with_quantity(self):
        m = MarketImpactModel()
        small = m.estimate(quantity=100, price=100.0, adv=100_000)
        large = m.estimate(quantity=10_000, price=100.0, adv=100_000)
        assert large.total_impact_bps > small.total_impact_bps

    def test_impact_increases_with_volatility(self):
        m = MarketImpactModel()
        low_vol = m.estimate(quantity=1000, price=100.0, adv=100_000, sigma=0.01)
        high_vol = m.estimate(quantity=1000, price=100.0, adv=100_000, sigma=0.05)
        assert high_vol.temporary_impact_bps > low_vol.temporary_impact_bps

    def test_temporary_impact_formula(self):
        """Verify: temp = sigma * sqrt(participation) * 10000."""
        m = MarketImpactModel()
        result = m.estimate(quantity=1000, price=100.0, adv=100_000, sigma=0.02)
        participation = 1000 / 100_000
        expected_temp = 0.02 * math.sqrt(participation) * 10_000
        assert result.temporary_impact_bps == pytest.approx(expected_temp, rel=1e-6)

    def test_permanent_impact_formula(self):
        """Verify: perm = gamma * participation * 10000."""
        m = MarketImpactModel(gamma=0.25)
        result = m.estimate(quantity=1000, price=100.0, adv=100_000, sigma=0.02)
        participation = 1000 / 100_000
        expected_perm = 0.25 * participation * 10_000
        assert result.permanent_impact_bps == pytest.approx(expected_perm, rel=1e-6)

    def test_total_cost_includes_transaction(self):
        m = MarketImpactModel()
        result = m.estimate(quantity=1000, price=100.0, adv=100_000, transaction_cost_bps=10.0)
        assert result.total_cost_bps == pytest.approx(result.total_impact_bps + 10.0)


# ──────────────────────────────────────────────────
# check_alpha_sufficient()
# ──────────────────────────────────────────────────


class TestCheckAlphaSufficient:
    def test_alpha_exceeds_cost(self):
        m = MarketImpactModel()
        result = m.check_alpha_sufficient(quantity=100, price=100.0, adv=100_000, expected_alpha_bps=100.0)
        assert result.alpha_sufficient is True
        assert result.recommended_qty == 100

    def test_alpha_insufficient_reduces_qty(self):
        m = MarketImpactModel()
        result = m.check_alpha_sufficient(quantity=50_000, price=100.0, adv=100_000, expected_alpha_bps=5.0, sigma=0.02)
        assert result.alpha_sufficient is False
        assert result.recommended_qty < 50_000

    def test_alpha_insufficient_finds_valid_qty(self):
        m = MarketImpactModel()
        result = m.check_alpha_sufficient(
            quantity=10_000, price=100.0, adv=100_000, expected_alpha_bps=50.0, sigma=0.02
        )
        if not result.alpha_sufficient:
            # Recommended qty should produce cost <= alpha
            check = m.estimate(result.recommended_qty, 100.0, 100_000, 0.02)
            assert check.total_cost_bps <= 50.0 or result.recommended_qty == 0

    def test_zero_alpha_gives_zero_recommended(self):
        m = MarketImpactModel()
        result = m.check_alpha_sufficient(quantity=1000, price=100.0, adv=100_000, expected_alpha_bps=0.0)
        assert result.alpha_sufficient is False
        assert result.recommended_qty == 0


# ──────────────────────────────────────────────────
# should_trade()
# ──────────────────────────────────────────────────


class TestShouldTrade:
    def test_should_trade_with_high_alpha(self):
        m = MarketImpactModel()
        assert m.should_trade(100, 100.0, 100_000, expected_alpha_bps=100.0) is True

    def test_should_not_trade_with_zero_alpha(self):
        m = MarketImpactModel()
        assert m.should_trade(1000, 100.0, 100_000, expected_alpha_bps=0.0) is False

    def test_should_not_trade_large_order_small_alpha(self):
        m = MarketImpactModel()
        assert m.should_trade(50_000, 100.0, 100_000, expected_alpha_bps=2.0) is False
