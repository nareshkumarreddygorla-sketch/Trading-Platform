"""Multi-source news aggregator for trading signals."""
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    title: str
    source: str
    url: str
    published_at: str
    summary: str = ""
    sentiment_score: float = 0.5  # 0 = very bearish, 1 = very bullish
    relevance_score: float = 0.5
    symbols: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    article_id: str = ""

    def __post_init__(self):
        if not self.article_id:
            self.article_id = hashlib.md5(f"{self.title}:{self.source}".encode()).hexdigest()[:12]


@dataclass
class NewsSentimentSummary:
    overall_score: float
    sentiment_label: str
    article_count: int
    top_positive: List[str]
    top_negative: List[str]
    sector_sentiments: Dict[str, float]
    symbol_sentiments: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class NewsAggregator:
    """Aggregates market news from multiple sources with sentiment scoring."""

    def __init__(self):
        self._cache: List[NewsArticle] = []
        self._cache_ttl = 300.0
        self._last_fetch = 0.0
        self._finbert = None
        self._newsapi_key = os.environ.get("NEWSAPI_KEY", "")

    def _init_finbert(self):
        if self._finbert is None:
            try:
                from src.ai.models.sentiment_predictor import SentimentPredictor
                self._finbert = SentimentPredictor()
            except Exception:
                pass

    async def fetch_news(self, symbols: Optional[List[str]] = None, market: str = "india") -> List[NewsArticle]:
        now = time.time()
        if now - self._last_fetch < self._cache_ttl and self._cache:
            if symbols:
                return [a for a in self._cache if any(s.upper() in [x.upper() for x in a.symbols] for s in symbols) or not a.symbols]
            return self._cache

        articles: List[NewsArticle] = []

        # Source 1: NewsAPI (if key set)
        if self._newsapi_key:
            try:
                articles.extend(await self._fetch_newsapi(symbols, market))
            except Exception as e:
                logger.debug("NewsAPI fetch failed: %s", e)

        # Source 2: RSS feeds (free)
        try:
            articles.extend(await self._fetch_rss(market))
        except Exception as e:
            logger.debug("RSS fetch failed: %s", e)

        if not articles:
            logger.warning(
                "NewsAggregator: no articles from any source (NewsAPI key=%s, RSS attempted). "
                "Returning empty list -- no fabricated data.",
                "configured" if self._newsapi_key else "not set",
            )

        # Deduplicate
        seen: set = set()
        unique = []
        for a in articles:
            if a.article_id not in seen:
                seen.add(a.article_id)
                unique.append(a)

        # Sentiment analysis via FinBERT
        self._init_finbert()
        if self._finbert is not None:
            for art in unique:
                try:
                    pred = self._finbert.predict({}, {"symbol": None, "text": f"{art.title}. {art.summary}"})
                    if pred:
                        art.sentiment_score = pred.prob_up
                except Exception:
                    pass

        self._cache = unique
        self._last_fetch = now

        if symbols:
            return [a for a in unique if any(s.upper() in [x.upper() for x in a.symbols] for s in symbols) or not a.symbols]
        return unique

    async def _fetch_newsapi(self, symbols: Optional[List[str]], market: str) -> List[NewsArticle]:
        import httpx
        q = " OR ".join(symbols[:5]) if symbols else ("Indian stock market" if market == "india" else "stock market")
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://newsapi.org/v2/everything",
                params={"q": q, "sortBy": "publishedAt", "pageSize": 20, "apiKey": self._newsapi_key, "language": "en"},
            )
            resp.raise_for_status()
            return [
                NewsArticle(
                    title=a.get("title", ""), source=a["source"]["name"],
                    url=a.get("url", ""), published_at=a.get("publishedAt", ""),
                    summary=(a.get("description") or "")[:200],
                    symbols=symbols or [], categories=["market_news"],
                )
                for a in resp.json().get("articles", [])[:20]
                if a.get("title")
            ]

    async def _fetch_rss(self, market: str) -> List[NewsArticle]:
        import httpx
        feeds = {
            "india": [
                ("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "Economic Times"),
            ],
            "us": [
                ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US", "Yahoo Finance"),
            ],
        }
        out: List[NewsArticle] = []
        for url, source in feeds.get(market, feeds["india"]):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        continue
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(resp.text)
                    for item in root.iter("item"):
                        title = item.findtext("title", "")
                        if title:
                            out.append(NewsArticle(
                                title=title, source=source,
                                url=item.findtext("link", ""),
                                published_at=item.findtext("pubDate", ""),
                                summary=(item.findtext("description") or "")[:200],
                                categories=["rss"],
                            ))
            except Exception as e:
                logger.debug("RSS %s failed: %s", source, e)
        return out[:20]

    def get_sentiment_summary(self, articles: Optional[List[NewsArticle]] = None) -> NewsSentimentSummary:
        arts = articles or self._cache
        if not arts:
            return NewsSentimentSummary(overall_score=0.5, sentiment_label="neutral", article_count=0, top_positive=[], top_negative=[], sector_sentiments={}, symbol_sentiments={})

        scores = [a.sentiment_score for a in arts]
        overall = sum(scores) / len(scores)
        positive = sorted([a for a in arts if a.sentiment_score > 0.6], key=lambda a: -a.sentiment_score)
        negative = sorted([a for a in arts if a.sentiment_score < 0.4], key=lambda a: a.sentiment_score)

        sym_scores: Dict[str, List[float]] = {}
        for a in arts:
            for sym in a.symbols:
                sym_scores.setdefault(sym, []).append(a.sentiment_score)
        symbol_sentiments = {s: round(sum(v) / len(v), 3) for s, v in sym_scores.items()}

        label = "bullish" if overall > 0.6 else ("bearish" if overall < 0.4 else "neutral")
        return NewsSentimentSummary(
            overall_score=round(overall, 3), sentiment_label=label,
            article_count=len(arts), top_positive=[a.title for a in positive[:3]],
            top_negative=[a.title for a in negative[:3]],
            sector_sentiments={}, symbol_sentiments=symbol_sentiments,
        )
