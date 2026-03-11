"""
Feature store: persist and retrieve time-series features for ML.
File-based persistence by default; production: TimescaleDB/ClickHouse.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .schema import FeatureVector

logger = logging.getLogger(__name__)

DEFAULT_STORE_DIR = os.environ.get("FEATURE_STORE_DIR", "/tmp/feature_store")


class FeatureStore:
    """
    Read/write feature vectors by symbol and time range.
    Uses JSONL file per version when connection_url is empty; else use DB.
    """

    def __init__(self, connection_url: str = ""):
        self.connection_url = connection_url
        self._engine = None
        self._store_dir = Path(DEFAULT_STORE_DIR)
        self._store_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, version: str) -> Path:
        return self._store_dir / f"features_{version}.jsonl"

    async def write(self, vector: FeatureVector, version: str = "v1") -> None:
        """Persist one feature vector (append to JSONL)."""
        path = self._path(version)
        ts_str = vector.ts.isoformat() if hasattr(vector.ts, "isoformat") else str(vector.ts)
        line = json.dumps({"symbol": vector.symbol, "ts": ts_str, "features": vector.features}) + "\n"
        with open(path, "a") as f:
            f.write(line)
        logger.debug("FeatureStore.write %s %s", vector.symbol, ts_str)

    async def read(
        self,
        symbol: str,
        from_ts: datetime,
        to_ts: datetime,
        version: str = "v1",
        feature_names: list[str] | None = None,
    ) -> list[FeatureVector]:
        """Load feature vectors in time range from file store."""
        path = self._path(version)
        if not path.exists():
            return []
        result = []
        from_ts_str = from_ts.isoformat() if hasattr(from_ts, "isoformat") else str(from_ts)
        to_ts_str = to_ts.isoformat() if hasattr(to_ts, "isoformat") else str(to_ts)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if row.get("symbol") != symbol:
                        continue
                    ts_str = row.get("ts", "")
                    if from_ts_str and ts_str < from_ts_str:
                        continue
                    if to_ts_str and ts_str > to_ts_str:
                        continue
                    features = row.get("features", {})
                    if feature_names:
                        features = {k: v for k, v in features.items() if k in feature_names}
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    except Exception as e:
                        logger.debug("Feature store parse ts %s: %s", ts_str, e)
                        ts = from_ts
                    result.append(FeatureVector(symbol=symbol, ts=ts, features=features))
                except (json.JSONDecodeError, KeyError):
                    continue
        result.sort(key=lambda v: v.ts)
        return result

    def compute_features(self, bars: list[Any], symbol: str, ts: datetime) -> FeatureVector:
        """Compute default technical features from OHLCV bars. Placeholder."""
        features = {}
        if len(bars) >= 20:
            closes = [b.close for b in bars]
            features["returns_1d"] = (closes[-1] - closes[-2]) / (closes[-2] + 1e-12) if len(closes) >= 2 else 0
            features["volatility_20d"] = 0.0  # np.std(closes[-20:])
        return FeatureVector(symbol=symbol, ts=ts, features=features)
