"""
News Feed Aggregator: Collect financial news from multiple RSS sources.
Used for LLM sentiment analysis and pre-market briefing.
"""
import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Indian financial news RSS feeds
RSS_FEEDS = {
    "moneycontrol": "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "moneycontrol_markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "et_stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "livemint": "https://www.livemint.com/rss/markets",
    "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
    "reuters_markets": "https://feeds.reuters.com/reuters/globalmarketsNews",
}


@dataclass
class NewsArticle:
    """A single news article."""
    title: str
    source: str
    url: str
    published: Optional[datetime] = None
    summary: str = ""
    symbols: List[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None  # -1 to 1
    relevance_score: float = 0.0
    _hash: str = ""

    def __post_init__(self):
        if not self._hash:
            self._hash = hashlib.md5(
                f"{self.title}{self.url}".encode()
            ).hexdigest()[:16]


class NewsFeedAggregator:
    """
    Aggregates news from multiple RSS feeds.
    Deduplicates, extracts stock mentions, and caches.
    """

    def __init__(
        self,
        feeds: Optional[Dict[str, str]] = None,
        max_articles: int = 200,
        max_age_hours: int = 24,
    ):
        self._feeds = feeds or RSS_FEEDS
        self._articles: Dict[str, NewsArticle] = {}  # hash -> article
        self._max_articles = max_articles
        self._max_age_hours = max_age_hours
        self._last_fetch: Optional[datetime] = None

    async def fetch_all(self) -> List[NewsArticle]:
        """Fetch from all configured RSS feeds."""
        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser not installed, skipping news fetch")
            return []

        tasks = []
        for source, url in self._feeds.items():
            tasks.append(self._fetch_feed(source, url))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        new_count = 0
        for result in results:
            if isinstance(result, list):
                for article in result:
                    if article._hash not in self._articles:
                        self._articles[article._hash] = article
                        new_count += 1

        # Prune old articles
        self._prune_old()

        self._last_fetch = datetime.now(timezone.utc)
        logger.info("News fetch: %d new articles, %d total cached", new_count, len(self._articles))
        return self.get_recent()

    async def _fetch_feed(self, source: str, url: str) -> List[NewsArticle]:
        """Fetch a single RSS feed."""
        import feedparser

        try:
            # feedparser is synchronous, run in executor
            loop = asyncio.get_event_loop()
            feed = await loop.run_in_executor(None, feedparser.parse, url)

            articles = []
            for entry in feed.entries[:30]:  # Max 30 per feed
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    except Exception:
                        pass

                article = NewsArticle(
                    title=entry.get("title", "").strip(),
                    source=source,
                    url=entry.get("link", ""),
                    published=published,
                    summary=entry.get("summary", "")[:500],
                )

                # Extract mentioned symbols
                article.symbols = self._extract_symbols(article.title + " " + article.summary)
                articles.append(article)

            return articles
        except Exception as e:
            logger.debug("Feed %s error: %s", source, e)
            return []

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract NSE stock symbols mentioned in text."""
        # Common NSE stock names and their symbols
        SYMBOL_MAP = {
            "reliance": "RELIANCE", "tcs": "TCS", "infosys": "INFY",
            "hdfc bank": "HDFCBANK", "icici bank": "ICICIBANK",
            "sbi": "SBIN", "axis bank": "AXISBANK", "kotak": "KOTAKBANK",
            "hindustan unilever": "HINDUNILVR", "itc": "ITC",
            "bharti airtel": "BHARTIARTL", "asian paints": "ASIANPAINT",
            "bajaj finance": "BAJFINANCE", "sun pharma": "SUNPHARMA",
            "titan": "TITAN", "wipro": "WIPRO", "hcl tech": "HCLTECH",
            "maruti": "MARUTI", "tata motors": "TATAMOTORS",
            "larsen": "LT", "nifty": "NIFTY", "sensex": "SENSEX",
            "adani": "ADANIENT", "ultratech": "ULTRACEMCO",
        }

        text_lower = text.lower()
        found = []
        for name, symbol in SYMBOL_MAP.items():
            if name in text_lower:
                found.append(symbol)
        return list(set(found))

    def _prune_old(self) -> None:
        """Remove articles older than max_age_hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._max_age_hours)
        to_remove = []
        for h, article in self._articles.items():
            if article.published and article.published < cutoff:
                to_remove.append(h)
        for h in to_remove:
            del self._articles[h]

        # Also cap total count
        if len(self._articles) > self._max_articles:
            sorted_articles = sorted(
                self._articles.items(),
                key=lambda x: x[1].published or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )
            self._articles = dict(sorted_articles[:self._max_articles])

    def get_recent(self, hours: int = 12, limit: int = 50) -> List[NewsArticle]:
        """Get recent articles sorted by time."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent = [
            a for a in self._articles.values()
            if a.published is None or a.published >= cutoff
        ]
        recent.sort(
            key=lambda a: a.published or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return recent[:limit]

    def get_by_symbol(self, symbol: str, limit: int = 20) -> List[NewsArticle]:
        """Get articles mentioning a specific symbol."""
        matching = [
            a for a in self._articles.values()
            if symbol in a.symbols
        ]
        matching.sort(
            key=lambda a: a.published or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return matching[:limit]

    def get_headline_summary(self, max_headlines: int = 20) -> str:
        """Get formatted headline summary for LLM input."""
        recent = self.get_recent(hours=6, limit=max_headlines)
        if not recent:
            return "No recent news available."

        lines = []
        for a in recent:
            time_str = a.published.strftime("%H:%M") if a.published else "??:??"
            symbols_str = f" [{', '.join(a.symbols)}]" if a.symbols else ""
            lines.append(f"[{time_str}] {a.title}{symbols_str} ({a.source})")
        return "\n".join(lines)
