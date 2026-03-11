"""
Full feature engineering pipeline: price, microstructure, regime, cross-asset.
Outputs versioned feature vectors to the feature store.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from src.core.events import Bar, OrderBookSnapshot
from src.feature_store.schema import FeatureVector
from src.feature_store.store import FeatureStore

from .cross_asset import compute_cross_asset_features
from .microstructure import compute_microstructure_features
from .price import compute_price_features
from .regime_features import compute_regime_features
from .specs import FEATURE_VERSION, FeatureGroup

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
        enabled_groups: list[FeatureGroup] | None = None,
    ):
        self.feature_store = feature_store
        self.version = version
        self.enabled_groups = enabled_groups or list(FeatureGroup)

    def run(
        self,
        symbol: str,
        bars: list[Bar],
        ts: datetime | None = None,
        order_book: OrderBookSnapshot | None = None,
        index_returns: Any | None = None,
        sector_returns: list[Any] | None = None,
        india_vix: float | None = None,
        usdinr_return: float | None = None,
        global_index_return: float | None = None,
        buy_volume: float | None = None,
        sell_volume: float | None = None,
        vwap: float | None = None,
    ) -> FeatureVector:
        """Compute all enabled feature groups and return a single FeatureVector."""
        ts = ts or (bars[-1].ts if bars else datetime.now(UTC))
        features: dict[str, float] = {}

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
            features.update(compute_regime_features(bars, index_returns=idx_ret, sector_returns=sector_returns))

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
        bars: list[Bar],
        **kwargs: Any,
    ) -> FeatureVector:
        """Run pipeline and write to feature store."""
        vector = self.run(symbol=symbol, bars=bars, **kwargs)
        await self.feature_store.write(vector, version=self.version)
        return vector
