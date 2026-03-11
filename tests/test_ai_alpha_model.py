"""AI AlphaModel: predict deterministic fallback when no model loaded."""

from src.ai.alpha_model import AlphaModel
from src.core.events import Exchange, SignalSide


def test_predict_fallback_rsi_low():
    model = AlphaModel(strategy_id="test")
    features = {"rsi": 25.0, "ema_spread": 0.0}
    prob = model.predict(features)
    assert prob >= 0.6


def test_predict_fallback_rsi_high():
    model = AlphaModel(strategy_id="test")
    features = {"rsi": 75.0, "ema_spread": 0.0}
    prob = model.predict(features)
    assert prob <= 0.4


def test_to_signal():
    model = AlphaModel(strategy_id="test")
    signal = model.to_signal(0.7, "RELIANCE", Exchange.NSE, 2500.0)
    assert signal.symbol == "RELIANCE"
    assert signal.exchange == Exchange.NSE
    assert signal.side == SignalSide.BUY
    assert signal.score >= 0.3
    assert signal.strategy_id == "test"


def test_to_signal_sell():
    model = AlphaModel(strategy_id="test")
    signal = model.to_signal(0.3, "INFY", Exchange.NSE, 1500.0)
    assert signal.side == SignalSide.SELL
