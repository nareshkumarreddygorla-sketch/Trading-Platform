"""Strategy marketplace data models."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List


class StrategyCategory(str, Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING = "swing"
    ML_AI = "ml_ai"
    OPTIONS = "options"
    CUSTOM = "custom"


class StrategyRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class StrategyListing:
    listing_id: str
    strategy_id: str
    name: str
    description: str
    author: str
    category: StrategyCategory
    risk_level: StrategyRisk
    indicators: List[str]
    backtest_stats: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    subscribers: int = 0
    rating: float = 0.0
    reviews: int = 0
    is_free: bool = True
    price_monthly: float = 0.0
    min_capital: float = 10000.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    total_return_pct: float = 0.0
    monthly_returns: List[float] = field(default_factory=list)
    supported_exchanges: List[str] = field(default_factory=lambda: ["NSE"])
    tags: List[str] = field(default_factory=list)
    is_published: bool = False
    code_hash: str = ""


@dataclass
class StrategySubscription:
    subscription_id: str
    user_id: str
    listing_id: str
    strategy_id: str
    subscribed_at: str
    active: bool = True
    auto_trade: bool = False
    capital_allocated: float = 0.0
    pnl_since_subscribe: float = 0.0


@dataclass
class StrategyReview:
    review_id: str
    listing_id: str
    user_id: str
    rating: float
    comment: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
