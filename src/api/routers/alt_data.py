"""API router for alternative data: news sentiment, FII/DII flows, combined signals."""

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/alt-data")


# ── Response models ──


class NewsArticleResponse(BaseModel):
    title: str
    source: str
    url: str
    published_at: str
    summary: str
    sentiment_score: float
    symbols: list


class SentimentSummaryResponse(BaseModel):
    overall_score: float
    sentiment_label: str
    article_count: int
    top_positive: list
    top_negative: list
    symbol_sentiments: dict


class FlowResponse(BaseModel):
    date: str
    fii_buy: float
    fii_sell: float
    fii_net: float
    dii_buy: float
    dii_sell: float
    dii_net: float
    total_net: float


class FlowAnalysisResponse(BaseModel):
    trend: str
    fii_streak: int
    dii_streak: int
    avg_fii_net_5d: float
    avg_dii_net_5d: float
    total_fii_net_month: float
    total_dii_net_month: float
    signal_strength: float
    recommendation: str


class CombinedSignalResponse(BaseModel):
    news_sentiment: float
    news_label: str
    fii_dii_trend: str
    fii_dii_signal: float
    exposure_multiplier: float
    combined_score: float
    combined_label: str
    recommendation: str


# ── Helpers ──


def _get_news_aggregator(request: Request):
    agg = getattr(request.app.state, "news_aggregator", None)
    if agg is None:
        from src.data_pipeline.news_aggregator import NewsAggregator

        agg = NewsAggregator()
        request.app.state.news_aggregator = agg
    return agg


def _get_fii_dii_tracker(request: Request):
    tracker = getattr(request.app.state, "fii_dii_tracker", None)
    if tracker is None:
        from src.data_pipeline.fii_dii_flow import FIIDIITracker

        tracker = FIIDIITracker()
        request.app.state.fii_dii_tracker = tracker
    return tracker


# ── News endpoints ──


@router.get("/news", response_model=list[NewsArticleResponse])
async def get_news(
    request: Request,
    symbols: str | None = None,
    market: str = "india",
):
    """Fetch latest market news with sentiment scores."""
    agg = _get_news_aggregator(request)
    sym_list = [s.strip() for s in symbols.split(",")] if symbols else None
    articles = await agg.fetch_news(symbols=sym_list, market=market)
    return [
        NewsArticleResponse(
            title=a.title,
            source=a.source,
            url=a.url,
            published_at=a.published_at,
            summary=a.summary,
            sentiment_score=a.sentiment_score,
            symbols=a.symbols,
        )
        for a in articles[:30]
    ]


@router.get("/news/sentiment", response_model=SentimentSummaryResponse)
async def get_sentiment_summary(
    request: Request,
    symbols: str | None = None,
    market: str = "india",
):
    """Get aggregated sentiment summary from recent news."""
    agg = _get_news_aggregator(request)
    sym_list = [s.strip() for s in symbols.split(",")] if symbols else None
    articles = await agg.fetch_news(symbols=sym_list, market=market)
    summary = agg.get_sentiment_summary(articles)
    return SentimentSummaryResponse(
        overall_score=summary.overall_score,
        sentiment_label=summary.sentiment_label,
        article_count=summary.article_count,
        top_positive=summary.top_positive,
        top_negative=summary.top_negative,
        symbol_sentiments=summary.symbol_sentiments,
    )


# ── FII/DII endpoints ──


@router.get("/fii-dii/flows", response_model=list[FlowResponse])
async def get_fii_dii_flows(request: Request, days: int = 10):
    """Get recent FII/DII institutional flow data."""
    tracker = _get_fii_dii_tracker(request)
    flows = tracker.get_recent_flows(days=days)
    return [
        FlowResponse(
            date=f.date,
            fii_buy=f.fii_buy,
            fii_sell=f.fii_sell,
            fii_net=f.fii_net,
            dii_buy=f.dii_buy,
            dii_sell=f.dii_sell,
            dii_net=f.dii_net,
            total_net=f.total_net,
        )
        for f in flows
    ]


@router.get("/fii-dii/analysis", response_model=FlowAnalysisResponse)
async def get_fii_dii_analysis(request: Request):
    """Get FII/DII flow analysis with trend and signals."""
    tracker = _get_fii_dii_tracker(request)
    a = tracker.analyze()
    return FlowAnalysisResponse(
        trend=a.trend,
        fii_streak=a.fii_streak,
        dii_streak=a.dii_streak,
        avg_fii_net_5d=a.avg_fii_net_5d,
        avg_dii_net_5d=a.avg_dii_net_5d,
        total_fii_net_month=a.total_fii_net_month,
        total_dii_net_month=a.total_dii_net_month,
        signal_strength=a.signal_strength,
        recommendation=a.recommendation,
    )


# ── Combined signal ──


@router.get("/combined-signal", response_model=CombinedSignalResponse)
async def get_combined_signal(request: Request, market: str = "india"):
    """Get combined alternative data signal (news + FII/DII)."""
    agg = _get_news_aggregator(request)
    tracker = _get_fii_dii_tracker(request)

    articles = await agg.fetch_news(market=market)
    sentiment = agg.get_sentiment_summary(articles)
    flow_analysis = tracker.analyze()
    exposure_mult = tracker.to_exposure_multiplier()

    # Combine news sentiment (0-1) with flow signal
    news_score = sentiment.overall_score  # 0-1
    flow_score = (
        0.5
        + (
            flow_analysis.signal_strength
            * (1 if flow_analysis.trend == "bullish" else -1 if flow_analysis.trend == "bearish" else 0)
        )
        / 2
    )

    combined = 0.6 * news_score + 0.4 * flow_score
    if combined > 0.65:
        label, rec = "bullish", "Positive alt-data signals — favour long positions"
    elif combined < 0.35:
        label, rec = "bearish", "Negative alt-data signals — reduce exposure or hedge"
    else:
        label, rec = "neutral", "Mixed alt-data signals — maintain current positioning"

    return CombinedSignalResponse(
        news_sentiment=round(news_score, 3),
        news_label=sentiment.sentiment_label,
        fii_dii_trend=flow_analysis.trend,
        fii_dii_signal=round(flow_analysis.signal_strength, 3),
        exposure_multiplier=round(exposure_mult, 3),
        combined_score=round(combined, 3),
        combined_label=label,
        recommendation=rec,
    )
