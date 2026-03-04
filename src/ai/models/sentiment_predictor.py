"""
Financial sentiment analyzer using FinBERT.
Fetches Indian market news from free RSS feeds and analyzes sentiment.
Implements BasePredictor contract for EnsembleEngine integration.
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BasePredictor, PredictionOutput

logger = logging.getLogger(__name__)

# Indian financial news RSS feeds (free, no API key needed)
NEWS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://www.livemint.com/rss/markets",
]

# Cache TTL in seconds
CACHE_TTL = 300  # 5 minutes


def _try_import_transformers():
    try:
        from transformers import pipeline
        return pipeline
    except ImportError:
        return None


def _try_import_feedparser():
    try:
        import feedparser
        return feedparser
    except ImportError:
        return None


class SentimentCache:
    """Simple in-memory cache for sentiment results."""

    def __init__(self, ttl: int = CACHE_TTL):
        self._cache: Dict[str, Tuple[float, Dict]] = {}  # key -> (expiry, result)
        self._ttl = ttl

    def get(self, key: str) -> Optional[Dict]:
        if key in self._cache:
            expiry, result = self._cache[key]
            if time.time() < expiry:
                return result
            del self._cache[key]
        return None

    def set(self, key: str, result: Dict) -> None:
        self._cache[key] = (time.time() + self._ttl, result)
        # Evict old entries
        now = time.time()
        expired = [k for k, (exp, _) in self._cache.items() if now >= exp]
        for k in expired:
            del self._cache[k]


class NewsFetcher:
    """Fetch news headlines from RSS feeds."""

    def __init__(self, feeds: List[str] = None):
        self._feeds = feeds or NEWS_FEEDS
        self._feedparser = _try_import_feedparser()

    def fetch_headlines(self, symbol: Optional[str] = None, max_items: int = 20) -> List[str]:
        """Fetch recent headlines, optionally filtered by symbol."""
        if self._feedparser is None:
            return []

        headlines = []
        for feed_url in self._feeds:
            try:
                feed = self._feedparser.parse(feed_url)
                for entry in feed.entries[:max_items]:
                    title = entry.get("title", "")
                    if title:
                        if symbol is None or symbol.lower() in title.lower():
                            headlines.append(title)
            except Exception as e:
                logger.debug("Failed to fetch feed %s: %s", feed_url, e)

        return headlines[:max_items]


class SentimentAnalyzer:
    """FinBERT-based financial sentiment analyzer."""

    def __init__(self):
        self._pipeline = None
        self._initialized = False

    def _init_pipeline(self):
        if self._initialized:
            return
        self._initialized = True
        pipeline_fn = _try_import_transformers()
        if pipeline_fn is None:
            logger.info("transformers not available; sentiment analyzer disabled")
            return
        try:
            self._pipeline = pipeline_fn(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                top_k=None,
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT sentiment pipeline loaded")
        except Exception as e:
            logger.warning("Failed to load FinBERT: %s", e)

    def analyze(self, texts: List[str]) -> Dict[str, float]:
        """Analyze sentiment of texts. Returns {positive, negative, neutral} scores."""
        self._init_pipeline()
        if self._pipeline is None or not texts:
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

        try:
            results = self._pipeline(texts[:10])  # Limit to 10 texts for speed
            pos_scores = []
            neg_scores = []
            neu_scores = []

            for result in results:
                for label_score in result:
                    label = label_score["label"].lower()
                    score = label_score["score"]
                    if label == "positive":
                        pos_scores.append(score)
                    elif label == "negative":
                        neg_scores.append(score)
                    else:
                        neu_scores.append(score)

            return {
                "positive": float(np.mean(pos_scores)) if pos_scores else 0.33,
                "negative": float(np.mean(neg_scores)) if neg_scores else 0.33,
                "neutral": float(np.mean(neu_scores)) if neu_scores else 0.34,
            }
        except Exception as e:
            logger.debug("Sentiment analysis failed: %s", e)
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}


class SentimentPredictor(BasePredictor):
    """Financial news sentiment predictor."""

    model_id = "sentiment_finbert"
    version = "v1"

    def __init__(self):
        self.path = ""
        self._analyzer = SentimentAnalyzer()
        self._fetcher = NewsFetcher()
        self._cache = SentimentCache()

    def predict(self, features: Dict[str, float], context: Optional[Dict[str, Any]] = None) -> PredictionOutput:
        symbol = None
        if context:
            symbol = context.get("symbol")

        cache_key = f"sentiment_{symbol or 'market'}"
        cached = self._cache.get(cache_key)
        if cached:
            return PredictionOutput(
                prob_up=cached["prob_up"],
                expected_return=cached["expected_return"],
                confidence=cached["confidence"],
                model_id=self.model_id,
                version=self.version,
                metadata=cached.get("metadata", {}),
            )

        # Fetch and analyze
        headlines = self._fetcher.fetch_headlines(symbol=symbol, max_items=15)
        if not headlines:
            return PredictionOutput(
                prob_up=0.5, expected_return=0.0, confidence=0.0,
                model_id=self.model_id, version=self.version,
                metadata={"reason": "no_headlines", "symbol": symbol},
            )

        sentiment = self._analyzer.analyze(headlines)

        # Map sentiment to prediction
        pos = sentiment["positive"]
        neg = sentiment["negative"]
        # Weighted: positive pushes prob_up higher, negative pushes it lower
        prob_up = 0.5 + (pos - neg) * 0.3  # Scale sentiment to +-30%
        prob_up = float(np.clip(prob_up, 0.1, 0.9))

        # Confidence based on sentiment strength
        confidence = abs(pos - neg)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        expected_return = (prob_up - 0.5) * 0.015

        result = {
            "prob_up": prob_up,
            "expected_return": expected_return,
            "confidence": confidence,
            "metadata": {
                "headlines_count": len(headlines),
                "sentiment": sentiment,
                "symbol": symbol,
            },
        }
        self._cache.set(cache_key, result)

        return PredictionOutput(
            prob_up=prob_up,
            expected_return=expected_return,
            confidence=confidence,
            model_id=self.model_id,
            version=self.version,
            metadata=result["metadata"],
        )
