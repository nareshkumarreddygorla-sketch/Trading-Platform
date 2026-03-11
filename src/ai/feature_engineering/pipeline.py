"""
Full feature engineering pipeline: price, microstructure, regime, cross-asset.
Outputs versioned feature vectors to the feature store.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.core.events import Bar, OrderBookSnapshot

from src.feature_store.schema import FeatureVector
from src.feature_store.store import FeatureStore

from .specs import FEATURE_VERSION, FeatureGroup
from .price import compute_price_features
from .microstructure import compute_microstructure_features
from .regime_features import compute_regime_features
from .cross_asset import compute_cross_asset_features

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Orchestrates computation of all feature groups and writes to feature store
    with version. Configurable which groups to run.
    """

    def __init__(
        self,
        feature_store: FeatureStore,
        version: str = FEATURE_VERSION,
        enabled_groups: Optional[List[FeatureGroup]] = None,
    ):
        self.feature_store = feature_store
        self.version = version
        self.enabled_groups = enabled_groups or list(FeatureGroup)

    def run(
        self,
        symbol: str,
        bars: List[Bar],
        ts: Optional[datetime] = None,
        order_book: Optional[OrderBookSnapshot] = None,
        index_returns: Optional[Any] = None,
        sector_returns: Optional[List[Any]] = None,
        india_vix: Optional[float] = None,
        usdinr_return: Optional[float] = None,
        global_index_return: Optional[float] = None,
        buy_volume: Optional[float] = None,
        sell_volume: Optional[float] = None,
        vwap: Optional[float] = None,
    ) -> FeatureVector:
        """Compute all enabled feature groups and return a single FeatureVector."""
        ts = ts or (bars[-1].ts if bars else datetime.now(timezone.utc))
        features: Dict[str, float] = {}

        if FeatureGroup.PRICE in self.enabled_groups:
            features.update(compute_price_features(bars))

        if FeatureGroup.MICROSTRUCTURE in self.enabled_groups:
            features.update(
                compute_microstructure_features(
                    bars,
                    order_book=order_book,
                    buy_volume=buy_volume,
                    sell_volume=sell_volume,
                    vwap=vwap,
                )
            )

        if FeatureGroup.REGIME in self.enabled_groups:
            idx_ret = index_returns if hasattr(index_returns, "__len__") else None
            features.update(
                compute_regime_features(bars, index_returns=idx_ret, sector_returns=sector_returns)
            )

        if FeatureGroup.CROSS_ASSET in self.enabled_groups:
            idx_ret = index_returns if hasattr(index_returns, "__len__") else None
            features.update(
                compute_cross_asset_features(
                    bars,
                    index_returns=idx_ret,
                    india_vix=india_vix,
                    usdinr_return=usdinr_return,
                    global_index_return=global_index_return,
                )
            )

        vector = FeatureVector(symbol=symbol, ts=ts, features=features)
        return vector

    async def run_and_persist(
        self,
        symbol: str,
        bars: List[Bar],
        **kwargs: Any,
    ) -> FeatureVector:
        """Run pipeline and write to feature store."""
        vector = self.run(symbol=symbol, bars=bars, **kwargs)
        await self.feature_store.write(vector, version=self.version)
        return vector
