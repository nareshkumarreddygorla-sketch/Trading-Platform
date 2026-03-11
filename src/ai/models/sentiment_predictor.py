"""
Financial sentiment analyzer using FinBERT.
Fetches Indian market news from free RSS feeds and analyzes sentiment.
Implements BasePredictor contract for EnsembleEngine integration.
"""
import logging
import re
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BasePredictor, PredictionOutput, estimate_empirical_return

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
        self._lock = threading.Lock()
        self._max_size = 1000

    def get(self, key: str) -> Optional[Dict]:
        with self._lock:
            if key in self._cache:
                expiry, result = self._cache[key]
                if time.time() < expiry:
                    return result
                del self._cache[key]
            return None

    def set(self, key: str, result: Dict) -> None:
        with self._lock:
            self._cache[key] = (time.time() + self._ttl, result)
            # Evict old entries
            now = time.time()
            expired = [k for k, (exp, _) in self._cache.items() if now >= exp]
            for k in expired:
                del self._cache[k]
            # Enforce max size cap
            if len(self._cache) > self._max_size:
                oldest = sorted(self._cache.items(), key=lambda x: x[1][0])[:len(self._cache) - self._max_size]
                for k, _ in oldest:
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
                try:
                    req = urllib.request.Request(feed_url, headers={"User-Agent": "AlphaForge/1.0"})
                    response = urllib.request.urlopen(req, timeout=10)
                    feed = self._feedparser.parse(response.read())
                except Exception as fetch_err:
                    logger.debug("Feed fetch timeout/error for %s: %s", feed_url, fetch_err)
                    continue
                for entry in feed.entries[:max_items]:
                    title = entry.get("title", "")
                    if title:
                        if symbol is None:
                            headlines.append(title)
                        elif symbol:
                            # Exact word boundary matching to avoid false positives
                            pattern = r'\b' + re.escape(symbol) + r'\b'
                            if re.search(pattern, title, re.IGNORECASE):
                                headlines.append(title)
            except Exception as e:
                logger.debug("Failed to fetch feed %s: %s", feed_url, e)

        return headlines[:max_items]


class SentimentAnalyzer:
    """FinBERT-based financial sentiment analyzer."""

    def __init__(self):
        self._pipeline = None
        self._initialized = False
        self._max_texts = 10
        self._init_lock = threading.Lock()

    def _init_pipeline(self):
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            logger.info("Loading FinBERT model (this may take several seconds on first call)...")
            pipeline_fn = _try_import_transformers()
            if pipeline_fn is None:
                logger.info("transformers not available; sentiment analyzer disabled")
                self._initialized = True
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
                self._initialized = True
            except Exception as e:
                logger.warning("Failed to load FinBERT: %s — sentiment disabled", e)
                self._initialized = True

    def analyze(self, texts: List[str]) -> Dict[str, float]:
        """Analyze sentiment of texts. Returns {positive, negative, neutral} scores."""
        self._init_pipeline()
        if self._pipeline is None or not texts:
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

        try:
            if len(texts) > self._max_texts:
                logger.debug("Sentiment analysis: truncating %d texts to %d", len(texts), self._max_texts)
            results = self._pipeline(texts[:self._max_texts])
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

    @property
    def is_ready(self) -> bool:
        return self._analyzer._pipeline is not None or not self._analyzer._initialized

    def __repr__(self) -> str:
        return f"<SentimentPredictor model_id={self.model_id!r} version={self.version!r} ready={self.is_ready}>"

    def __init__(self):
        self.path = ""
        self._analyzer = SentimentAnalyzer()
        self._fetcher = NewsFetcher()
        self._cache = SentimentCache()

    def predict(self, features: Dict[str, float], context: Optional[Dict[str, Any]] = None) -> Optional[PredictionOutput]:
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
            return None

        sentiment = self._analyzer.analyze(headlines)

        # Map sentiment to prediction
        pos = sentiment["positive"]
        neg = sentiment["negative"]
        # Calibrated mapping: cap sentiment contribution at +/- 15%
        # (empirical studies show sentiment has low signal-to-noise on Indian markets)
        sentiment_edge = (pos - neg) * 0.15  # Was 0.3, halved for production safety
        prob_up = 0.5 + sentiment_edge
        prob_up = float(np.clip(prob_up, 0.35, 0.65))  # Hard cap (was 0.1-0.9)

        # Confidence: sentiment is inherently noisy, cap at 0.6
        confidence = abs(pos - neg) * 0.5  # Was 1.0, halved
        confidence = float(np.clip(confidence, 0.0, 0.6))

        # Require minimum headline count for statistical significance
        if len(headlines) < 3:
            confidence *= 0.3  # Heavily penalize predictions from few headlines

        # Validate prediction meets minimum quality standards
        if not self.validate_prediction(prob_up, confidence):
            return None

        # Expected return estimate using empirical calibration
        expected_return = estimate_empirical_return(prob_up, self._empirical_returns)
        if expected_return is None:
            return None

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
