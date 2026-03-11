"""API router for strategy marketplace."""
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

_service = None


def _get_svc():
    global _service
    if _service is None:
        from src.marketplace.service import MarketplaceService
        _service = MarketplaceService()
    return _service


class PublishRequest(BaseModel):
    strategy_id: str
    name: str = Field(..., min_length=3, max_length=100)
    description: str = Field(..., min_length=10, max_length=1000)
    author: str = Field(default="anonymous", max_length=50)
    category: str = "custom"
    risk_level: str = "medium"
    indicators: List[str] = []
    backtest_stats: Dict[str, Any] = {}
    tags: List[str] = []


class SubscribeRequest(BaseModel):
    listing_id: str
    auto_trade: bool = False
    capital: float = Field(default=0.0, ge=0)


class ReviewRequest(BaseModel):
    listing_id: str
    rating: float = Field(..., ge=1.0, le=5.0)
    comment: str = Field(..., min_length=5, max_length=500)


@router.get("/marketplace/strategies")
async def list_strategies(
    category: Optional[str] = None,
    risk_level: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: str = Query(default="rating"),
    limit: int = Query(default=20, ge=1, le=100),
):
    svc = _get_svc()
    from src.marketplace.models import StrategyCategory, StrategyRisk
    cat_values = [e.value for e in StrategyCategory]
    risk_values = [e.value for e in StrategyRisk]
    cat = StrategyCategory(category) if category and category in cat_values else None
    risk = StrategyRisk(risk_level) if risk_level and risk_level in risk_values else None
    strategies = svc.list_strategies(category=cat, risk_level=risk, search=search, sort_by=sort_by, limit=limit)
    return {
        "strategies": [
            {
                "listing_id": s.listing_id, "strategy_id": s.strategy_id,
                "name": s.name, "description": s.description, "author": s.author,
                "category": s.category.value, "risk_level": s.risk_level.value,
                "indicators": s.indicators, "sharpe_ratio": s.sharpe_ratio,
                "max_drawdown_pct": s.max_drawdown_pct, "win_rate": s.win_rate,
                "total_return_pct": s.total_return_pct, "subscribers": s.subscribers,
                "rating": s.rating, "reviews": s.reviews, "tags": s.tags, "is_free": s.is_free,
            }
            for s in strategies
        ],
        "count": len(strategies),
    }


@router.get("/marketplace/strategies/{listing_id}")
async def get_strategy(listing_id: str):
    svc = _get_svc()
    listing = svc.get_listing(listing_id)
    if not listing:
        raise HTTPException(404, "Strategy not found")
    reviews = svc.get_reviews(listing_id)
    return {
        "listing": {
            "listing_id": listing.listing_id, "strategy_id": listing.strategy_id,
            "name": listing.name, "description": listing.description,
            "author": listing.author, "category": listing.category.value,
            "risk_level": listing.risk_level.value, "indicators": listing.indicators,
            "backtest_stats": listing.backtest_stats, "sharpe_ratio": listing.sharpe_ratio,
            "max_drawdown_pct": listing.max_drawdown_pct, "win_rate": listing.win_rate,
            "total_return_pct": listing.total_return_pct, "subscribers": listing.subscribers,
            "rating": listing.rating, "tags": listing.tags,
            "supported_exchanges": listing.supported_exchanges,
            "min_capital": listing.min_capital, "created_at": listing.created_at,
        },
        "reviews": [
            {"rating": r.rating, "comment": r.comment, "user_id": r.user_id, "created_at": r.created_at}
            for r in reviews[-10:]
        ],
    }


@router.post("/marketplace/publish")
async def publish_strategy(req: PublishRequest):
    svc = _get_svc()
    listing = svc.publish_strategy(
        strategy_id=req.strategy_id, name=req.name, description=req.description,
        author=req.author, category=req.category, risk_level=req.risk_level,
        indicators=req.indicators, backtest_stats=req.backtest_stats, tags=req.tags,
    )
    return {"listing_id": listing.listing_id, "message": "Strategy published"}


@router.post("/marketplace/subscribe")
async def subscribe(req: SubscribeRequest):
    svc = _get_svc()
    sub = svc.subscribe(user_id="user-1", listing_id=req.listing_id, auto_trade=req.auto_trade, capital=req.capital)
    if not sub:
        raise HTTPException(404, "Strategy not found")
    return {"subscription_id": sub.subscription_id, "strategy_id": sub.strategy_id, "message": "Subscribed"}


@router.post("/marketplace/review")
async def add_review(req: ReviewRequest):
    svc = _get_svc()
    review = svc.add_review(listing_id=req.listing_id, user_id="user-1", rating=req.rating, comment=req.comment)
    if not review:
        raise HTTPException(404, "Strategy not found")
    return {"review_id": review.review_id, "message": "Review added"}


@router.get("/marketplace/leaderboard")
async def leaderboard(
    metric: str = Query(default="sharpe"),
    limit: int = Query(default=10, ge=1, le=50),
):
    svc = _get_svc()
    return {"leaderboard": svc.get_leaderboard(metric=metric, limit=limit)}


@router.get("/marketplace/subscriptions")
async def my_subscriptions():
    svc = _get_svc()
    subs = svc.get_user_subscriptions("user-1")
    return {
        "subscriptions": [
            {
                "subscription_id": s.subscription_id, "listing_id": s.listing_id,
                "strategy_id": s.strategy_id, "auto_trade": s.auto_trade,
                "capital_allocated": s.capital_allocated,
                "pnl_since_subscribe": s.pnl_since_subscribe,
                "subscribed_at": s.subscribed_at,
            }
            for s in subs
        ],
    }
