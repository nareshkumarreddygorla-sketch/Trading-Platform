from src.core.events import Exchange
from src.strategy_engine import EMACrossoverStrategy, StrategyRegistry
from src.strategy_engine.base import MarketState


def test_ema_crossover_warm(sample_bars):
    strategy = EMACrossoverStrategy(fast=9, slow=21)
    state = MarketState(
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        bars=sample_bars,
        latest_price=sample_bars[-1].close,
        volume=sample_bars[-1].volume,
    )
    assert strategy.warm(state) is True


def test_ema_crossover_generates_signals(sample_bars):
    strategy = EMACrossoverStrategy(fast=9, slow=21)
    state = MarketState(
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        bars=sample_bars,
        latest_price=sample_bars[-1].close,
        volume=sample_bars[-1].volume,
    )
    signals = strategy.generate_signals(state)
    # May be 0 or 1 depending on data
    assert isinstance(signals, list)
    for s in signals:
        assert s.strategy_id == "ema_crossover"
        assert s.symbol == "RELIANCE"
        assert s.score >= 0 and s.score <= 1


def test_registry():
    reg = StrategyRegistry()
    reg.register(EMACrossoverStrategy())
    assert "ema_crossover" in reg.list_all()
    assert "ema_crossover" in reg.list_enabled()
    reg.disable("ema_crossover")
    assert "ema_crossover" not in reg.list_enabled()
