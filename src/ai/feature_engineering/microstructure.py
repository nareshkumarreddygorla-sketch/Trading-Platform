"""
Microstructure features: order flow imbalance, bid-ask spread, volume delta,
VWAP deviation, liquidity pressure.
Requires tick/orderbook data; falls back to OHLCV proxies when missing.
"""
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.events import Bar, OrderBookSnapshot


def compute_order_flow_imbalance(
    buy_volume: float, sell_volume: float
) -> float:
    """(buy - sell) / (buy + sell). -1 to 1."""
    total = buy_volume + sell_volume
    if total < 1e-12:
        return 0.0
    return (buy_volume - sell_volume) / total


def compute_bid_ask_spread_bps(bid: float, ask: float) -> float:
    if bid <= 0:
        return 0.0
    return (ask - bid) / bid * 10_000.0


def compute_volume_delta(buy_volume: float, sell_volume: float) -> float:
    return buy_volume - sell_volume


def compute_vwap_deviation_bps(price: float, vwap: float) -> float:
    if vwap <= 0:
        return 0.0
    return (price - vwap) / vwap * 10_000.0


def compute_liquidity_pressure(volume: float, depth: float) -> float:
    """Volume / depth proxy. High = pressure."""
    if depth < 1e-12:
        return 0.0
    return volume / (depth + 1e-12)


def compute_microstructure_features(
    bars: List[Bar],
    order_book: Optional[OrderBookSnapshot] = None,
    buy_volume: Optional[float] = None,
    sell_volume: Optional[float] = None,
    vwap: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute microstructure features. If order book / tick data not available,
    use OHLCV proxies (e.g. spread from high-low, volume delta from close vs open).
    """
    features: Dict[str, float] = {}
    if not bars:
        return features

    last = bars[-1]
    close = last.close
    volume = last.volume

    if order_book:
        features["bid_ask_spread_bps"] = compute_bid_ask_spread_bps(
            order_book.bid, order_book.ask
        )
        depth = order_book.bid_size + order_book.ask_size
        features["liquidity_pressure"] = compute_liquidity_pressure(volume, depth)
    else:
        # Proxy: spread from high-low as bps
        if last.low > 0:
            features["bid_ask_spread_bps"] = (last.high - last.low) / last.low * 10_000.0
        else:
            features["bid_ask_spread_bps"] = 0.0
        features["liquidity_pressure"] = 0.0  # no depth

    if buy_volume is not None and sell_volume is not None:
        features["order_flow_imbalance"] = compute_order_flow_imbalance(
            buy_volume, sell_volume
        )
        features["volume_delta"] = compute_volume_delta(buy_volume, sell_volume)
    else:
        # Proxy: close > open => buy pressure
        if len(bars) >= 2:
            prev = bars[-2]
            buy_proxy = volume if close > prev.close else 0.0
            sell_proxy = volume if close <= prev.close else 0.0
            features["order_flow_imbalance"] = compute_order_flow_imbalance(
                buy_proxy, sell_proxy
            )
        else:
            features["order_flow_imbalance"] = 0.0
        features["volume_delta"] = 0.0

    if vwap is not None:
        features["vwap_deviation_bps"] = compute_vwap_deviation_bps(close, vwap)
    else:
        # VWAP proxy from bars
        total_v = sum(b.volume for b in bars)
        if total_v > 0:
            vwap_proxy = sum(b.close * b.volume for b in bars) / total_v
            features["vwap_deviation_bps"] = compute_vwap_deviation_bps(close, vwap_proxy)
        else:
            features["vwap_deviation_bps"] = 0.0

    return features
