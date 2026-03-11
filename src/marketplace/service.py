"""Strategy marketplace service with pre-seeded strategies."""
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import (
    StrategyCategory, StrategyListing, StrategyReview,
    StrategyRisk, StrategySubscription,
)

logger = logging.getLogger(__name__)

# Pre-seeded strategies that demonstrate the platform's capabilities
_SEED_STRATEGIES = [
    StrategyListing(
        listing_id="mkt-001", strategy_id="preset_momentum_breakout",
        name="Momentum Breakout Pro",
        description="Breakout above 20-period high with 1.5x volume confirmation. Best for trending markets.",
        author="TradingPlatform", category=StrategyCategory.BREAKOUT,
        risk_level=StrategyRisk.MEDIUM, indicators=["SMA", "ATR", "Volume"],
        backtest_stats={"sharpe": 1.8, "max_dd": -8.5, "win_rate": 62, "profit_factor": 2.1, "total_trades": 342},
        subscribers=1247, rating=4.6, reviews=89, is_free=True,
        sharpe_ratio=1.8, max_drawdown_pct=8.5, win_rate=62.0, total_return_pct=45.2,
        tags=["momentum", "volume", "breakout", "nse"], is_published=True,
    ),
    StrategyListing(
        listing_id="mkt-002", strategy_id="preset_mean_reversion_rsi",
        name="RSI Mean Reversion",
        description="Buys oversold RSI with Bollinger Band confirmation. Ideal for range-bound markets.",
        author="TradingPlatform", category=StrategyCategory.MEAN_REVERSION,
        risk_level=StrategyRisk.LOW, indicators=["RSI", "Bollinger Bands"],
        backtest_stats={"sharpe": 1.5, "max_dd": -6.2, "win_rate": 68, "profit_factor": 1.8, "total_trades": 521},
        subscribers=2034, rating=4.7, reviews=156, is_free=True,
        sharpe_ratio=1.5, max_drawdown_pct=6.2, win_rate=68.0, total_return_pct=32.8,
        tags=["rsi", "mean_reversion", "bollinger", "low_risk"], is_published=True,
    ),
    StrategyListing(
        listing_id="mkt-003", strategy_id="preset_trend_following",
        name="EMA Trend Follower",
        description="Follow trends using 9/21 EMA crossover with ADX trend strength filter (ADX > 25).",
        author="TradingPlatform", category=StrategyCategory.TREND_FOLLOWING,
        risk_level=StrategyRisk.MEDIUM, indicators=["EMA", "ADX"],
        backtest_stats={"sharpe": 2.1, "max_dd": -7.8, "win_rate": 58, "profit_factor": 2.5, "total_trades": 198},
        subscribers=1856, rating=4.8, reviews=134, is_free=True,
        sharpe_ratio=2.1, max_drawdown_pct=7.8, win_rate=58.0, total_return_pct=52.3,
        tags=["ema", "trend", "adx", "crossover"], is_published=True,
    ),
    StrategyListing(
        listing_id="mkt-004", strategy_id="preset_vwap_reversion",
        name="VWAP Mean Reversion",
        description="Trade price deviation from VWAP measured in ATR units. Buy below -2 ATR, sell above +2 ATR.",
        author="TradingPlatform", category=StrategyCategory.MEAN_REVERSION,
        risk_level=StrategyRisk.LOW, indicators=["VWAP", "ATR"],
        backtest_stats={"sharpe": 1.3, "max_dd": -5.1, "win_rate": 71, "profit_factor": 1.6, "total_trades": 623},
        subscribers=945, rating=4.4, reviews=67, is_free=True,
        sharpe_ratio=1.3, max_drawdown_pct=5.1, win_rate=71.0, total_return_pct=28.5,
        tags=["vwap", "intraday", "mean_reversion"], is_published=True,
    ),
    StrategyListing(
        listing_id="mkt-005", strategy_id="preset_opening_range_breakout",
        name="Opening Range Breakout",
        description="Trade breakout of first 15-minute opening range with ATR confirmation filter.",
        author="TradingPlatform", category=StrategyCategory.BREAKOUT,
        risk_level=StrategyRisk.HIGH, indicators=["ATR", "Opening Range"],
        backtest_stats={"sharpe": 1.6, "max_dd": -11.3, "win_rate": 55, "profit_factor": 2.0, "total_trades": 412},
        subscribers=1523, rating=4.5, reviews=98, is_free=True,
        sharpe_ratio=1.6, max_drawdown_pct=11.3, win_rate=55.0, total_return_pct=41.7,
        tags=["orb", "intraday", "breakout", "opening_range"], is_published=True,
    ),
    StrategyListing(
        listing_id="mkt-006", strategy_id="ml_predictor",
        name="AI Ensemble Predictor",
        description="IC-weighted ensemble of XGBoost, LSTM, Transformer, RL-PPO and FinBERT sentiment. Dynamic model weighting via rolling Information Coefficient.",
        author="TradingPlatform", category=StrategyCategory.ML_AI,
        risk_level=StrategyRisk.MEDIUM, indicators=["XGBoost", "LSTM", "Transformer", "RL-PPO", "FinBERT"],
        backtest_stats={"sharpe": 2.4, "max_dd": -9.2, "win_rate": 61, "profit_factor": 2.8, "total_trades": 156},
        subscribers=3241, rating=4.9, reviews=245, is_free=True,
        sharpe_ratio=2.4, max_drawdown_pct=9.2, win_rate=61.0, total_return_pct=67.5,
        tags=["ai", "ml", "ensemble", "deep_learning", "premium"], is_published=True,
    ),
]


class MarketplaceService:
    """Strategy marketplace: listing, subscription, review management."""

    def __init__(self):
        self._listings: Dict[str, StrategyListing] = {s.listing_id: s for s in _SEED_STRATEGIES}
        self._subscriptions: Dict[str, StrategySubscription] = {}
        self._reviews: Dict[str, List[StrategyReview]] = {}
        self._user_subscriptions: Dict[str, List[str]] = {}  # user_id -> [sub_ids]
        logger.info("Marketplace: %d pre-seeded strategies", len(self._listings))

    # ── Listing queries ─────────────────────────────────────────────────

    def list_strategies(
        self,
        category: Optional[StrategyCategory] = None,
        risk_level: Optional[StrategyRisk] = None,
        search: Optional[str] = None,
        sort_by: str = "rating",
        limit: int = 20,
    ) -> List[StrategyListing]:
        results = [l for l in self._listings.values() if l.is_published]
        if category:
            results = [l for l in results if l.category == category]
        if risk_level:
            results = [l for l in results if l.risk_level == risk_level]
        if search:
            q = search.lower()
            results = [
                l for l in results
                if q in l.name.lower() or q in l.description.lower() or any(q in t for t in l.tags)
            ]
        sort_map = {
            "rating": lambda l: l.rating,
            "subscribers": lambda l: l.subscribers,
            "sharpe": lambda l: l.sharpe_ratio,
            "win_rate": lambda l: l.win_rate,
            "return": lambda l: l.total_return_pct,
        }
        results.sort(key=sort_map.get(sort_by, sort_map["rating"]), reverse=True)
        return results[:limit]

    def get_listing(self, listing_id: str) -> Optional[StrategyListing]:
        return self._listings.get(listing_id)

    # ── Publish ──────────────────────────────────────────────────────────

    def publish_strategy(
        self, strategy_id: str, name: str, description: str, author: str,
        category: str, risk_level: str, indicators: List[str],
        backtest_stats: Dict[str, Any], tags: Optional[List[str]] = None,
        code: str = "",
    ) -> StrategyListing:
        listing_id = f"mkt-{uuid.uuid4().hex[:8]}"
        cat_values = [e.value for e in StrategyCategory]
        risk_values = [e.value for e in StrategyRisk]
        listing = StrategyListing(
            listing_id=listing_id, strategy_id=strategy_id, name=name,
            description=description, author=author,
            category=StrategyCategory(category) if category in cat_values else StrategyCategory.CUSTOM,
            risk_level=StrategyRisk(risk_level) if risk_level in risk_values else StrategyRisk.MEDIUM,
            indicators=indicators, backtest_stats=backtest_stats,
            tags=tags or [], is_published=True,
            sharpe_ratio=backtest_stats.get("sharpe", 0.0),
            max_drawdown_pct=abs(backtest_stats.get("max_dd", 0.0)),
            win_rate=backtest_stats.get("win_rate", 0.0),
            total_return_pct=backtest_stats.get("total_return", 0.0),
            code_hash=hashlib.sha256(code.encode()).hexdigest()[:16] if code else "",
        )
        self._listings[listing_id] = listing
        logger.info("Strategy published: %s (%s)", name, listing_id)
        return listing

    # ── Subscribe / Unsubscribe ──────────────────────────────────────────

    def subscribe(self, user_id: str, listing_id: str, auto_trade: bool = False, capital: float = 0.0) -> Optional[StrategySubscription]:
        listing = self._listings.get(listing_id)
        if listing is None:
            return None
        sub_id = f"sub-{uuid.uuid4().hex[:8]}"
        sub = StrategySubscription(
            subscription_id=sub_id, user_id=user_id,
            listing_id=listing_id, strategy_id=listing.strategy_id,
            subscribed_at=datetime.now(timezone.utc).isoformat(),
            auto_trade=auto_trade, capital_allocated=capital,
        )
        self._subscriptions[sub_id] = sub
        listing.subscribers += 1
        self._user_subscriptions.setdefault(user_id, []).append(sub_id)
        return sub

    def unsubscribe(self, subscription_id: str) -> bool:
        sub = self._subscriptions.get(subscription_id)
        if sub is None:
            return False
        sub.active = False
        listing = self._listings.get(sub.listing_id)
        if listing:
            listing.subscribers = max(0, listing.subscribers - 1)
        return True

    def get_user_subscriptions(self, user_id: str) -> List[StrategySubscription]:
        ids = self._user_subscriptions.get(user_id, [])
        return [self._subscriptions[i] for i in ids if i in self._subscriptions and self._subscriptions[i].active]

    # ── Reviews ──────────────────────────────────────────────────────────

    def add_review(self, listing_id: str, user_id: str, rating: float, comment: str) -> Optional[StrategyReview]:
        if listing_id not in self._listings:
            return None
        review = StrategyReview(
            review_id=f"rev-{uuid.uuid4().hex[:8]}", listing_id=listing_id,
            user_id=user_id, rating=max(1.0, min(5.0, rating)), comment=comment,
        )
        reviews = self._reviews.setdefault(listing_id, [])
        reviews.append(review)
        listing = self._listings[listing_id]
        listing.reviews = len(reviews)
        listing.rating = round(sum(r.rating for r in reviews) / len(reviews), 1)
        return review

    def get_reviews(self, listing_id: str) -> List[StrategyReview]:
        return self._reviews.get(listing_id, [])

    # ── Leaderboard ──────────────────────────────────────────────────────

    def get_leaderboard(self, metric: str = "sharpe", limit: int = 10) -> List[Dict[str, Any]]:
        published = [l for l in self._listings.values() if l.is_published]
        sort_map = {
            "sharpe": lambda l: l.sharpe_ratio,
            "win_rate": lambda l: l.win_rate,
            "return": lambda l: l.total_return_pct,
            "subscribers": lambda l: l.subscribers,
        }
        published.sort(key=sort_map.get(metric, sort_map["sharpe"]), reverse=True)
        return [
            {
                "rank": i + 1, "listing_id": l.listing_id, "name": l.name,
                "author": l.author, "sharpe_ratio": l.sharpe_ratio,
                "win_rate": l.win_rate, "total_return_pct": l.total_return_pct,
                "max_drawdown_pct": l.max_drawdown_pct,
                "subscribers": l.subscribers, "rating": l.rating,
                "category": l.category.value,
            }
            for i, l in enumerate(published[:limit])
        ]
