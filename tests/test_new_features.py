"""
Comprehensive tests for new platform features:
- Options Greeks & chain
- Marketplace service
- LLM strategy generator
- Alternative data pipeline (news, FII/DII)
- Multi-broker gateway framework

Run:
    PYTHONPATH=. pytest tests/test_new_features.py -v --tb=short
"""

import asyncio

import pytest

# ────────────────────────────────────────────────────────
# 1. Options Greeks Calculator
# ────────────────────────────────────────────────────────


class TestOptionsGreeks:
    """Test Black-Scholes Greeks calculation."""

    def test_call_price_positive(self):
        from src.options.greeks import black_scholes

        result = black_scholes(100, 100, 0.25, 0.05, 0.2, "call")
        assert result.price > 0
        assert 0 < result.delta < 1
        assert result.gamma > 0
        assert result.theta < 0  # time decay
        assert result.vega > 0

    def test_put_price_positive(self):
        from src.options.greeks import black_scholes

        result = black_scholes(100, 100, 0.25, 0.05, 0.2, "put")
        assert result.price > 0
        assert -1 < result.delta < 0
        assert result.gamma > 0

    def test_deep_itm_call_delta_near_one(self):
        from src.options.greeks import black_scholes

        result = black_scholes(200, 100, 0.25, 0.05, 0.2, "call")
        assert result.delta > 0.95

    def test_deep_otm_call_delta_near_zero(self):
        from src.options.greeks import black_scholes

        result = black_scholes(50, 100, 0.25, 0.05, 0.2, "call")
        assert result.delta < 0.05

    def test_put_call_parity(self):
        """Put-Call parity: C - P = S - K * e^(-rT)"""
        import math

        from src.options.greeks import black_scholes

        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.2
        call = black_scholes(S, K, T, r, sigma, "call")
        put = black_scholes(S, K, T, r, sigma, "put")
        parity_diff = call.price - put.price - (S - K * math.exp(-r * T))
        assert abs(parity_diff) < 0.01

    def test_implied_volatility(self):
        from src.options.greeks import black_scholes, implied_volatility

        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.3
        result = black_scholes(S, K, T, r, sigma, "call")
        iv = implied_volatility(result.price, S, K, T, r, "call")
        assert abs(iv - sigma) < 0.001

    def test_zero_time_returns_zero(self):
        from src.options.greeks import black_scholes

        result = black_scholes(110, 100, 0, 0.05, 0.2, "call")
        assert result.price == 0


# ────────────────────────────────────────────────────────
# 2. Options Chain Manager
# ────────────────────────────────────────────────────────


class TestOptionsChain:
    """Test options chain building and analysis."""

    def test_build_chain_creates_calls_and_puts(self):
        from src.options.chain import OptionsChainManager

        mgr = OptionsChainManager()
        strikes = mgr.generate_strikes(22000, num_strikes=11)
        chain = mgr.build_chain("NIFTY", 22000, "2024-03-28", strikes, lot_size=50)
        assert len(chain.calls) == 11
        assert len(chain.puts) == 11
        assert chain.calls[0].lot_size == 50

    def test_max_pain(self):
        from src.options.chain import OptionsChainManager

        mgr = OptionsChainManager()
        strikes = mgr.generate_strikes(22000, num_strikes=11)
        chain = mgr.build_chain("NIFTY", 22000, "2024-03-28", strikes)
        pain = mgr.max_pain(chain)
        assert pain > 0

    def test_pcr(self):
        from src.options.chain import OptionsChainManager

        mgr = OptionsChainManager()
        strikes = mgr.generate_strikes(22000, num_strikes=11)
        chain = mgr.build_chain("NIFTY", 22000, "2024-03-28", strikes)
        pcr = mgr.pcr(chain)
        assert "pcr_volume" in pcr
        assert "pcr_oi" in pcr
        assert "sentiment" in pcr
        assert pcr["sentiment"] in ("bullish", "bearish", "neutral")

    def test_generate_strikes_small_price(self):
        from src.options.chain import OptionsChainManager

        mgr = OptionsChainManager()
        strikes = mgr.generate_strikes(150, num_strikes=9)
        assert len(strikes) == 9
        assert all(s > 0 for s in strikes)

    def test_generate_strikes_large_price(self):
        from src.options.chain import OptionsChainManager

        mgr = OptionsChainManager()
        strikes = mgr.generate_strikes(45000, num_strikes=11)
        assert len(strikes) == 11
        diffs = [strikes[i + 1] - strikes[i] for i in range(len(strikes) - 1)]
        assert len(set(diffs)) == 1  # All diffs equal

    def test_chain_caching(self):
        from src.options.chain import OptionsChainManager

        mgr = OptionsChainManager()
        strikes = mgr.generate_strikes(22000, num_strikes=5)
        mgr.build_chain("NIFTY", 22000, "2024-03-28", strikes)
        cached = mgr.get_chain("NIFTY", "2024-03-28")
        assert cached is not None
        assert cached.underlying == "NIFTY"

    def test_greeks_on_contracts(self):
        from src.options.chain import OptionsChainManager

        mgr = OptionsChainManager()
        strikes = mgr.generate_strikes(22000, num_strikes=5)
        chain = mgr.build_chain("NIFTY", 22000, "2024-03-28", strikes)
        for call in chain.calls:
            assert call.greeks is not None
            assert call.greeks.delta > 0


# ────────────────────────────────────────────────────────
# 3. Marketplace Service
# ────────────────────────────────────────────────────────


class TestMarketplace:
    """Test strategy marketplace service."""

    def test_list_seeded_strategies(self):
        from src.marketplace.service import MarketplaceService

        svc = MarketplaceService()
        strats = svc.list_strategies()
        assert len(strats) >= 5

    def test_filter_by_category(self):
        from src.marketplace.models import StrategyCategory
        from src.marketplace.service import MarketplaceService

        svc = MarketplaceService()
        mr = svc.list_strategies(category=StrategyCategory.MEAN_REVERSION)
        assert all(s.category == StrategyCategory.MEAN_REVERSION for s in mr)

    def test_sort_by_rating(self):
        from src.marketplace.service import MarketplaceService

        svc = MarketplaceService()
        strats = svc.list_strategies(sort_by="rating")
        if len(strats) >= 2:
            assert strats[0].rating >= strats[1].rating

    def test_subscribe_and_unsubscribe(self):
        from src.marketplace.service import MarketplaceService

        svc = MarketplaceService()
        strats = svc.list_strategies()
        lid = strats[0].listing_id

        sub = svc.subscribe("test_user", lid)
        assert sub is not None
        assert sub.user_id == "test_user"

        success = svc.unsubscribe(sub.subscription_id)
        assert success

    def test_review(self):
        from src.marketplace.service import MarketplaceService

        svc = MarketplaceService()
        strats = svc.list_strategies()
        lid = strats[0].listing_id

        review = svc.add_review(lid, "test_user", 4.5, "Good strategy")
        assert review is not None
        assert review.rating == 4.5

    def test_leaderboard(self):
        from src.marketplace.service import MarketplaceService

        svc = MarketplaceService()
        lb = svc.get_leaderboard(limit=3)
        assert len(lb) <= 3
        if len(lb) >= 2:
            assert lb[0]["sharpe_ratio"] >= lb[1]["sharpe_ratio"]

    def test_publish_strategy(self):
        from src.marketplace.service import MarketplaceService

        svc = MarketplaceService()
        listing = svc.publish_strategy(
            strategy_id="test_strat_1",
            name="Test Strategy",
            description="A test",
            category="momentum",
            risk_level="medium",
            author="test_author",
            indicators=["RSI", "EMA"],
            backtest_stats={"sharpe": 1.5, "max_dd": -5.0, "win_rate": 60},
            code="pass",
        )
        assert listing.name == "Test Strategy"
        assert listing.author == "test_author"


# ────────────────────────────────────────────────────────
# 4. LLM Strategy Generator
# ────────────────────────────────────────────────────────


class TestStrategyGenerator:
    """Test strategy generation from natural language."""

    @pytest.fixture
    def generator(self):
        from src.ai.llm.strategy_generator import StrategyGenerator

        return StrategyGenerator()

    def test_rsi_strategy(self, generator):
        result = asyncio.get_event_loop().run_until_complete(
            generator.generate("Buy when RSI drops below 30, sell when above 70")
        )
        assert result.validated
        assert "RSI" in result.code
        assert len(result.errors) == 0

    def test_macd_crossover(self, generator):
        result = asyncio.get_event_loop().run_until_complete(
            generator.generate("MACD crossover strategy on 15m timeframe")
        )
        assert result.validated
        assert "MACD" in result.code

    def test_bollinger_bands(self, generator):
        result = asyncio.get_event_loop().run_until_complete(
            generator.generate("Bollinger band bounce strategy with 1.5% stop loss")
        )
        assert result.validated
        assert "BBANDS" in result.code

    def test_vwap_strategy(self, generator):
        result = asyncio.get_event_loop().run_until_complete(generator.generate("VWAP breakout strategy"))
        assert result.validated

    def test_supertrend_strategy(self, generator):
        result = asyncio.get_event_loop().run_until_complete(generator.generate("Supertrend trend following on 1h"))
        assert result.validated
        assert "SUPERTREND" in result.code

    def test_multi_indicator(self, generator):
        result = asyncio.get_event_loop().run_until_complete(
            generator.generate(
                "Buy when RSI < 35 and MACD crosses up on 5m timeframe with 2% stop loss and 4% take profit"
            )
        )
        assert result.validated
        assert "RSI" in result.code
        assert "MACD" in result.code

    def test_generated_code_compiles(self, generator):
        result = asyncio.get_event_loop().run_until_complete(
            generator.generate("EMA crossover moving average strategy")
        )
        compile(result.code, "<test>", "exec")

    def test_list_and_get(self, generator):
        asyncio.get_event_loop().run_until_complete(generator.generate("RSI oversold bounce strategy"))
        strats = generator.list_generated()
        assert len(strats) >= 1
        name = strats[0].name
        s = generator.get_strategy(name)
        assert s is not None

    def test_delete(self, generator):
        asyncio.get_event_loop().run_until_complete(generator.generate("Simple moving average crossover"))
        strats = generator.list_generated()
        name = strats[0].name
        assert generator.delete_strategy(name)
        assert generator.get_strategy(name) is None

    def test_dangerous_patterns_rejected(self, generator):
        from src.ai.llm.strategy_generator import StrategyGenerator

        gen = StrategyGenerator()
        errors = gen._validate(
            "import os\nos.system('rm -rf /')",
            {
                "entry_rules": [{"condition": "True"}],
                "indicators": [{"name": "RSI"}],
                "risk": {"stop_loss_pct": 2, "max_position_pct": 5},
            },
        )
        assert any("Dangerous" in e for e in errors)


# ────────────────────────────────────────────────────────
# 5. FII/DII Flow Tracker
# ────────────────────────────────────────────────────────


class TestFIIDIITracker:
    def test_empty_tracker(self):
        """Tracker initialises with no data; get_recent_flows returns empty."""
        from src.data_pipeline.fii_dii_flow import FIIDIITracker

        tracker = FIIDIITracker()
        flows = tracker.get_recent_flows(10)
        assert len(flows) == 0  # No seed data by design

    def test_analyze(self):
        from src.data_pipeline.fii_dii_flow import FIIDIITracker

        tracker = FIIDIITracker()
        analysis = tracker.analyze()
        # Empty tracker returns neutral analysis
        assert analysis.trend in ("bullish", "bearish", "neutral")
        assert 0 <= analysis.signal_strength <= 1
        assert isinstance(analysis.recommendation, str)

    def test_exposure_multiplier(self):
        from src.data_pipeline.fii_dii_flow import FIIDIITracker

        tracker = FIIDIITracker()
        mult = tracker.to_exposure_multiplier()
        # Empty tracker returns neutral multiplier (1.0)
        assert 0.5 <= mult <= 1.3

    def test_add_flow(self):
        from src.data_pipeline.fii_dii_flow import FIIDIITracker, InstitutionalFlow

        tracker = FIIDIITracker()
        initial = len(tracker.get_recent_flows(100))
        tracker.add_flow(
            InstitutionalFlow(
                date="2024-01-15",
                fii_buy=5000,
                fii_sell=3000,
                dii_buy=4000,
                dii_sell=3500,
            )
        )
        assert len(tracker.get_recent_flows(100)) == initial + 1

    def test_flow_net_calculation(self):
        from src.data_pipeline.fii_dii_flow import InstitutionalFlow

        flow = InstitutionalFlow(
            date="2024-01-15",
            fii_buy=5000,
            fii_sell=3000,
            dii_buy=4000,
            dii_sell=3500,
        )
        assert flow.fii_net == 2000
        assert flow.dii_net == 500
        assert flow.total_net == 2500

    def test_max_capacity(self):
        from src.data_pipeline.fii_dii_flow import FIIDIITracker, InstitutionalFlow

        tracker = FIIDIITracker()
        for i in range(100):
            tracker.add_flow(
                InstitutionalFlow(
                    date=f"2024-06-{(i % 28) + 1:02d}",
                    fii_buy=5000,
                    fii_sell=3000,
                    dii_buy=4000,
                    dii_sell=3500,
                )
            )
        assert len(tracker._flows) <= 90


# ────────────────────────────────────────────────────────
# 6. News Aggregator
# ────────────────────────────────────────────────────────


class TestNewsAggregator:
    def test_aggregator_init(self):
        """Aggregator initialises with empty cache."""
        from src.data_pipeline.news_aggregator import NewsAggregator

        agg = NewsAggregator()
        assert agg._cache == []

    def test_sentiment_summary_with_articles(self):
        """Sentiment summary from manually created articles."""
        from src.data_pipeline.news_aggregator import NewsAggregator, NewsArticle

        agg = NewsAggregator()
        articles = [
            NewsArticle(title="Market rallies", source="Test", url="#", published_at="now", sentiment_score=0.8),
            NewsArticle(title="Stocks surge", source="Test", url="#", published_at="now", sentiment_score=0.7),
            NewsArticle(title="Mild correction", source="Test", url="#", published_at="now", sentiment_score=0.4),
        ]
        summary = agg.get_sentiment_summary(articles)
        assert summary.article_count == 3
        assert 0 <= summary.overall_score <= 1
        assert summary.sentiment_label in ("bullish", "bearish", "neutral")

    def test_empty_sentiment(self):
        from src.data_pipeline.news_aggregator import NewsAggregator

        agg = NewsAggregator()
        summary = agg.get_sentiment_summary([])
        assert summary.article_count == 0
        assert summary.overall_score == 0.5

    def test_article_id_generation(self):
        from src.data_pipeline.news_aggregator import NewsArticle

        a = NewsArticle(title="Test", source="Demo", url="#", published_at="now")
        assert len(a.article_id) == 12


# ────────────────────────────────────────────────────────
# 7. Multi-Broker Gateway Framework
# ────────────────────────────────────────────────────────


class TestBrokerGateway:
    def test_broker_types(self):
        from src.execution.gateways.base import BrokerType

        assert BrokerType.ANGEL_ONE.value == "angel_one"
        assert BrokerType.IBKR.value == "ibkr"
        assert BrokerType.ALPACA.value == "alpaca"

    def test_gateway_status(self):
        from src.execution.gateways.base import GatewayStatus

        assert GatewayStatus.CONNECTED.value == "connected"
        assert GatewayStatus.DISCONNECTED.value == "disconnected"

    def test_broker_order_dataclass(self):
        from src.execution.gateways.base import BrokerOrder

        order = BrokerOrder(
            broker_order_id="B123",
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            quantity=100,
            order_type="MARKET",
            limit_price=None,
            status="filled",
            filled_qty=100,
            avg_price=250.50,
        )
        assert order.broker_order_id == "B123"
        assert order.avg_price == 250.50

    def test_broker_health_dataclass(self):
        from src.execution.gateways.base import BrokerHealth, BrokerType, GatewayStatus

        health = BrokerHealth(
            broker=BrokerType.ALPACA,
            status=GatewayStatus.CONNECTED,
            latency_ms=15.5,
            last_heartbeat="2024-01-01T00:00:00Z",
        )
        assert health.status == GatewayStatus.CONNECTED
        assert health.latency_ms == 15.5

    def test_alpaca_gateway_paper_mode(self):
        from src.execution.gateways.alpaca_gateway import AlpacaGateway

        gw = AlpacaGateway()
        assert gw.paper is True
        exchanges = gw.supported_exchanges()
        assert "NYSE" in exchanges
        assert "NASDAQ" in exchanges

    def test_ibkr_gateway_paper_mode(self):
        from src.execution.gateways.ibkr_gateway import IBKRGateway

        gw = IBKRGateway()
        exchanges = gw.supported_exchanges()
        assert "NYSE" in exchanges
        assert "NSE" in exchanges
        assert len(exchanges) >= 10


# ────────────────────────────────────────────────────────
# 8. Broker Manager
# ────────────────────────────────────────────────────────


class TestBrokerManager:
    def test_exchange_broker_map(self):
        from src.execution.broker_manager import EXCHANGE_BROKER_MAP

        assert "NSE" in EXCHANGE_BROKER_MAP
        assert "NYSE" in EXCHANGE_BROKER_MAP
        assert "LSE" in EXCHANGE_BROKER_MAP

    def test_broker_manager_init(self):
        from src.execution.broker_manager import BrokerManager

        mgr = BrokerManager()
        assert mgr._gateways == {}

    def test_register_gateway(self):
        from src.execution.broker_manager import BrokerManager
        from src.execution.gateways.alpaca_gateway import AlpacaGateway
        from src.execution.gateways.base import BrokerType

        mgr = BrokerManager()
        gw = AlpacaGateway()
        mgr.register(gw)
        assert BrokerType.ALPACA in mgr._gateways
