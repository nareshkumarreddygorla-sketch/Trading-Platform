"""
Feature engine: build_features(bars) from historical bars only. No lookahead bias.
Used by AI signal engine and autonomous loop. Deterministic output for same bars.

Features (35+):
  - Returns (1, 5, 10, 20 bars)
  - Volatility (rolling, ATR, Bollinger bandwidth)
  - Trend (RSI, MACD line/signal/histogram, EMA spread, Stochastic %K/%D)
  - Volume (spike, OBV slope, VWAP distance)
  - Price structure (Bollinger %B, position in day range, gap%)
  - Candlestick patterns (doji, hammer, engulfing)
  - Momentum (5, 10, 20 bars, rate of change)
"""
import logging
from typing import Any, Dict, List

import numpy as np

from src.core.events import Bar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primitive array extractors
# ---------------------------------------------------------------------------
def _closes(bars: List[Bar]) -> np.ndarray:
    return np.array([b.close for b in bars], dtype=float)

def _opens(bars: List[Bar]) -> np.ndarray:
    return np.array([b.open for b in bars], dtype=float)

def _highs(bars: List[Bar]) -> np.ndarray:
    return np.array([b.high for b in bars], dtype=float)

def _lows(bars: List[Bar]) -> np.ndarray:
    return np.array([b.low for b in bars], dtype=float)

def _volumes(bars: List[Bar]) -> np.ndarray:
    return np.array([b.volume for b in bars], dtype=float)


# ---------------------------------------------------------------------------
# Core indicators
# ---------------------------------------------------------------------------
def _returns(closes: np.ndarray, period: int) -> float:
    if len(closes) < period + 1 or closes[-1 - period] == 0:
        return 0.0
    return float((closes[-1] - closes[-1 - period]) / (closes[-1 - period] + 1e-12))


def _rolling_volatility(closes: np.ndarray, window: int) -> float:
    if len(closes) < window + 1:
        return 0.0
    ret = np.diff(closes[-window - 1:]) / (closes[-window - 1:-1] + 1e-12)
    return float(np.std(ret))


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
    if len(close) < period + 1:
        return 0.0
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    return float(np.mean(tr[-period:]))


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-period - 1:])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss < 1e-12:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / (avg_loss + 1e-12)
    return float(100.0 - (100.0 / (1.0 + rs)))


def _ema(series: np.ndarray, period: int) -> float:
    if len(series) < period:
        return float(series[-1]) if len(series) else 0.0
    mult = 2.0 / (period + 1)
    ema_val = float(series[:period].mean())
    for i in range(period, len(series)):
        ema_val = (series[i] - ema_val) * mult + ema_val
    return float(ema_val)


def _ema_array(series: np.ndarray, period: int) -> np.ndarray:
    """Return full EMA array for MACD calculation."""
    if len(series) < period:
        return series.copy()
    mult = 2.0 / (period + 1)
    out = np.empty_like(series, dtype=float)
    out[:period] = series[:period]
    ema_val = float(series[:period].mean())
    for i in range(len(series)):
        if i < period:
            out[i] = ema_val
        else:
            ema_val = (series[i] - ema_val) * mult + ema_val
            out[i] = ema_val
    return out


def _ema_spread(closes: np.ndarray, fast: int = 12, slow: int = 26) -> float:
    if len(closes) < slow:
        return 0.0
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    if ema_slow == 0:
        return 0.0
    return float((ema_fast - ema_slow) / (ema_slow + 1e-12))


def _momentum(closes: np.ndarray, period: int) -> float:
    return _returns(closes, period)


def _volume_spike(volumes: np.ndarray, window: int = 20) -> float:
    if len(volumes) < window + 1:
        return 0.0
    recent = volumes[-1]
    mean_vol = np.mean(volumes[-window - 1:-1])
    if mean_vol < 1e-12:
        return 0.0
    return float((recent - mean_vol) / (mean_vol + 1e-12))


# ---------------------------------------------------------------------------
# NEW indicators: MACD, Bollinger, Stochastic, OBV, VWAP, Candlestick
# ---------------------------------------------------------------------------
def _macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, histogram."""
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0
    ema_fast = _ema_array(closes, fast)
    ema_slow = _ema_array(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema_array(macd_line, signal)
    histogram = macd_line - signal_line
    price = closes[-1] if closes[-1] != 0 else 1.0
    return (
        float(macd_line[-1] / (price + 1e-12)),   # normalize by price
        float(signal_line[-1] / (price + 1e-12)),
        float(histogram[-1] / (price + 1e-12)),
    )


def _bollinger(closes: np.ndarray, period: int = 20, num_std: float = 2.0):
    """Bollinger %B (0-1 position within bands) and bandwidth."""
    if len(closes) < period:
        return 0.5, 0.0
    window = closes[-period:]
    sma = np.mean(window)
    std = np.std(window)
    if std < 1e-12:
        return 0.5, 0.0
    upper = sma + num_std * std
    lower = sma - num_std * std
    band_range = upper - lower
    pct_b = float((closes[-1] - lower) / (band_range + 1e-12))
    bandwidth = float(band_range / (sma + 1e-12))
    return pct_b, bandwidth


def _stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                k_period: int = 14, d_period: int = 3):
    """Stochastic %K and %D (smoothed %K)."""
    if len(close) < k_period:
        return 50.0, 50.0
    highest_high = np.max(high[-k_period:])
    lowest_low = np.min(low[-k_period:])
    range_hl = highest_high - lowest_low
    if range_hl < 1e-12:
        return 50.0, 50.0
    k = float((close[-1] - lowest_low) / (range_hl + 1e-12)) * 100.0

    # %D = simple moving average of %K over last d_period bars
    if len(close) < k_period + d_period:
        return k, k
    k_values = []
    for i in range(d_period):
        offset = d_period - 1 - i
        end = len(close) - offset
        start = end - k_period
        if start < 0:
            k_values.append(k)
            continue
        hh = np.max(high[start:end])
        ll = np.min(low[start:end])
        r = hh - ll
        if r < 1e-12:
            k_values.append(50.0)
        else:
            k_values.append(((close[end - 1] - ll) / (r + 1e-12)) * 100.0)
    d = float(np.mean(k_values))
    return k, d


def _obv_slope(closes: np.ndarray, volumes: np.ndarray, period: int = 10) -> float:
    """On-Balance Volume slope (normalized). Measures volume-price agreement."""
    if len(closes) < period + 1:
        return 0.0
    obv = np.zeros(len(closes))
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    # Slope of OBV over last `period` bars (linear regression coefficient)
    x = np.arange(period, dtype=float)
    y = obv[-period:]
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom < 1e-12:
        return 0.0
    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    # Normalize by average volume
    avg_vol = np.mean(volumes[-period:])
    if avg_vol < 1e-12:
        return 0.0
    return float(slope / (avg_vol + 1e-12))


def _vwap_distance(closes: np.ndarray, volumes: np.ndarray, period: int = 20) -> float:
    """Distance of current price from VWAP (normalized)."""
    if len(closes) < period:
        return 0.0
    c = closes[-period:]
    v = volumes[-period:]
    total_vol = np.sum(v)
    if total_vol < 1e-12:
        return 0.0
    vwap = np.sum(c * v) / total_vol
    if vwap < 1e-12:
        return 0.0
    return float((closes[-1] - vwap) / (vwap + 1e-12))


def _price_position(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    period: int = 20) -> float:
    """Where current price sits in the period's high-low range (0=bottom, 1=top)."""
    if len(close) < period:
        return 0.5
    highest = np.max(high[-period:])
    lowest = np.min(low[-period:])
    range_val = highest - lowest
    if range_val < 1e-12:
        return 0.5
    return float((close[-1] - lowest) / (range_val + 1e-12))


def _gap_pct(opens: np.ndarray, closes: np.ndarray) -> float:
    """Gap between previous close and current open (percentage)."""
    if len(opens) < 2 or len(closes) < 2:
        return 0.0
    prev_close = closes[-2]
    curr_open = opens[-1]
    if prev_close < 1e-12:
        return 0.0
    return float((curr_open - prev_close) / (prev_close + 1e-12))


def _candlestick_features(opens: np.ndarray, highs: np.ndarray,
                          lows: np.ndarray, closes: np.ndarray):
    """Encode candlestick patterns as numerical features."""
    if len(closes) < 2:
        return 0.0, 0.0, 0.0, 0.0

    o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
    body = abs(c - o)
    total_range = h - l
    if total_range < 1e-12:
        total_range = 1e-12

    # Body ratio: small body = indecision (doji)
    body_ratio = float(body / total_range)

    # Upper shadow ratio
    upper_shadow = h - max(o, c)
    upper_shadow_ratio = float(upper_shadow / total_range)

    # Lower shadow ratio
    lower_shadow = min(o, c) - l
    lower_shadow_ratio = float(lower_shadow / total_range)

    # Engulfing: current body engulfs previous body
    if len(closes) >= 2:
        prev_body = abs(closes[-2] - opens[-2])
        engulfing = float(body / (prev_body + 1e-12)) if prev_body > 0 else 0.0
        engulfing = min(engulfing, 3.0)  # cap at 3x
    else:
        engulfing = 0.0

    return body_ratio, upper_shadow_ratio, lower_shadow_ratio, engulfing


def _rate_of_change(closes: np.ndarray, period: int) -> float:
    """Rate of change: percentage change over period."""
    if len(closes) < period + 1 or closes[-1 - period] < 1e-12:
        return 0.0
    return float((closes[-1] / closes[-1 - period] - 1.0))


# ---------------------------------------------------------------------------
# Microstructure features (Sprint 4.3)
# ---------------------------------------------------------------------------
def _intraday_seasonality(bars) -> tuple:
    """
    Sin/cos encoding of minute-of-day for intraday seasonality capture.
    NSE session: 09:15 - 15:30 = 375 minutes.
    Returns (sin_component, cos_component).
    """
    if not bars:
        return 0.0, 0.0
    last_bar = bars[-1]
    ts = getattr(last_bar, "timestamp", None) or getattr(last_bar, "ts", None)
    if ts is None:
        return 0.0, 0.0
    try:
        if isinstance(ts, str):
            from datetime import datetime as _dt
            ts = _dt.fromisoformat(ts.replace("Z", "+00:00"))
        minute_of_day = ts.hour * 60 + ts.minute
        # NSE session: 09:15 (555 min) to 15:30 (930 min) = 375 min window
        nse_start = 9 * 60 + 15  # 555
        nse_end = 15 * 60 + 30    # 930
        nse_minutes = nse_end - nse_start  # 375
        relative_minute = max(0, minute_of_day - nse_start)
        fraction = relative_minute / max(nse_minutes, 1)
        sin_val = float(np.sin(2 * np.pi * fraction))
        cos_val = float(np.cos(2 * np.pi * fraction))
        return sin_val, cos_val
    except Exception:
        return 0.0, 0.0


def _hurst_exponent(closes: np.ndarray, max_lag: int = 20) -> float:
    """
    Rescaled range (R/S) Hurst exponent estimate.
    H < 0.5: mean-reverting, H = 0.5: random walk, H > 0.5: trending.
    """
    n = len(closes)
    if n < max_lag + 2:
        return 0.5  # neutral default
    try:
        returns = np.diff(np.log(np.maximum(closes, 1e-12)))
        if len(returns) < max_lag:
            return 0.5

        lags = range(2, min(max_lag + 1, len(returns)))
        rs_values = []
        for lag in lags:
            rs_list = []
            for start in range(0, len(returns) - lag + 1, lag):
                chunk = returns[start : start + lag]
                if len(chunk) < 2:
                    continue
                mean_r = np.mean(chunk)
                deviate = np.cumsum(chunk - mean_r)
                r = np.max(deviate) - np.min(deviate)
                s = np.std(chunk)
                if s > 1e-12:
                    rs_list.append(r / s)
            if rs_list:
                rs_values.append((np.log(lag), np.log(np.mean(rs_list))))

        if len(rs_values) < 3:
            return 0.5

        x = np.array([v[0] for v in rs_values])
        y = np.array([v[1] for v in rs_values])
        # Linear regression slope = Hurst exponent
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        denom = np.sum((x - x_mean) ** 2)
        if denom < 1e-12:
            return 0.5
        h = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
        return float(np.clip(h, 0.0, 1.0))
    except Exception:
        return 0.5


def _volume_profile_deviation(closes: np.ndarray, volumes: np.ndarray, period: int = 50) -> float:
    """
    Distance of current price from Point of Control (POC) of volume profile.
    POC = price level with highest traded volume (approximated via histogram).
    Returns normalized deviation (positive = above POC, negative = below).
    """
    if len(closes) < period or len(volumes) < period:
        return 0.0
    try:
        c = closes[-period:]
        v = volumes[-period:]
        if np.sum(v) < 1e-12:
            return 0.0

        # Create price histogram weighted by volume
        n_bins = min(20, period // 3)
        if n_bins < 3:
            return 0.0
        hist, bin_edges = np.histogram(c, bins=n_bins, weights=v)
        # POC = midpoint of bin with max volume
        poc_bin = np.argmax(hist)
        poc_price = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2.0
        if poc_price < 1e-12:
            return 0.0
        return float((closes[-1] - poc_price) / (poc_price + 1e-12))
    except Exception:
        return 0.0


def _vol_of_vol(closes: np.ndarray, inner_window: int = 5, outer_window: int = 20) -> float:
    """Volatility of volatility: std of rolling short-term vol over outer window."""
    n = len(closes)
    if n < inner_window + outer_window:
        return 0.0
    try:
        returns = np.diff(closes) / (closes[:-1] + 1e-12)
        if len(returns) < inner_window + outer_window:
            return 0.0
        # Rolling short-term vol
        vols = []
        for i in range(outer_window):
            end = len(returns) - i
            start = end - inner_window
            if start < 0:
                break
            vols.append(np.std(returns[start:end]))
        if len(vols) < 3:
            return 0.0
        return float(np.std(vols))
    except Exception:
        return 0.0


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Average Directional Index — trend strength (0-100)."""
    n = len(close)
    if n < period + 1:
        return 25.0  # neutral default

    # True Range, +DM, -DM
    tr_arr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    )
    plus_dm = np.where(
        (high[1:] - high[:-1]) > (low[:-1] - low[1:]),
        np.maximum(high[1:] - high[:-1], 0),
        0.0
    )
    minus_dm = np.where(
        (low[:-1] - low[1:]) > (high[1:] - high[:-1]),
        np.maximum(low[:-1] - low[1:], 0),
        0.0
    )

    # Smoothed averages (Wilder's smoothing)
    atr_s = np.mean(tr_arr[:period])
    plus_di_s = np.mean(plus_dm[:period])
    minus_di_s = np.mean(minus_dm[:period])

    for i in range(period, len(tr_arr)):
        atr_s = (atr_s * (period - 1) + tr_arr[i]) / period
        plus_di_s = (plus_di_s * (period - 1) + plus_dm[i]) / period
        minus_di_s = (minus_di_s * (period - 1) + minus_dm[i]) / period

    if atr_s < 1e-12:
        return 25.0

    plus_di = 100.0 * plus_di_s / atr_s
    minus_di = 100.0 * minus_di_s / atr_s
    di_sum = plus_di + minus_di
    if di_sum < 1e-12:
        return 25.0

    dx = 100.0 * abs(plus_di - minus_di) / di_sum
    return float(min(dx, 100.0))


def _williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> float:
    """Williams %R: momentum oscillator (-100 to 0)."""
    if len(close) < period:
        return -50.0
    highest = np.max(high[-period:])
    lowest = np.min(low[-period:])
    r = highest - lowest
    if r < 1e-12:
        return -50.0
    return float(((highest - close[-1]) / r) * -100.0)


def _mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         volumes: np.ndarray, period: int = 14) -> float:
    """Money Flow Index (volume-weighted RSI, 0-100)."""
    if len(close) < period + 1:
        return 50.0
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volumes

    pos_flow = 0.0
    neg_flow = 0.0
    for i in range(-period, 0):
        if typical_price[i] > typical_price[i - 1]:
            pos_flow += raw_money_flow[i]
        elif typical_price[i] < typical_price[i - 1]:
            neg_flow += raw_money_flow[i]

    if neg_flow < 1e-12:
        return 100.0 if pos_flow > 0 else 50.0
    mfi_ratio = pos_flow / (neg_flow + 1e-12)
    return float(100.0 - (100.0 / (1.0 + mfi_ratio)))


# ---------------------------------------------------------------------------
# Feature Engine class
# ---------------------------------------------------------------------------
class FeatureEngine:
    """
    Build comprehensive feature dict from bars. Uses only historical data;
    no lookahead. Deterministic: same bars -> same features.
    Outputs 35+ features for robust ML predictions.
    """

    def __init__(
        self,
        return_periods: List[int] = None,
        vol_window: int = 20,
        atr_period: int = 14,
        rsi_period: int = 14,
        ema_fast: int = 12,
        ema_slow: int = 26,
        momentum_periods: List[int] = None,
        volume_spike_window: int = 20,
    ):
        self.return_periods = return_periods or [1, 5, 10, 20]
        self.vol_window = vol_window
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.momentum_periods = momentum_periods or [5, 10, 20]
        self.volume_spike_window = volume_spike_window

    def build_features(self, bars: List[Bar]) -> Dict[str, float]:
        """
        Compute 35+ features from bars. Only uses data up to and including
        the last bar. Returns dict of feature name -> value.
        """
        if not bars:
            return {}

        closes = _closes(bars)
        opens = _opens(bars)
        highs = _highs(bars)
        lows = _lows(bars)
        volumes = _volumes(bars)
        n = len(closes)

        out: Dict[str, Any] = {}

        # ── Returns (4 features) ──
        for p in self.return_periods:
            out[f"returns_{p}"] = _returns(closes, p) if n > p else 0.0

        # ── Volatility (3 features) ──
        out["rolling_volatility"] = _rolling_volatility(closes, self.vol_window)
        out["atr"] = _atr(highs, lows, closes, self.atr_period)
        _, boll_bw = _bollinger(closes, 20)
        out["bollinger_bandwidth"] = boll_bw

        # ── Trend indicators (8 features) ──
        out["rsi"] = _rsi(closes, self.rsi_period)
        out["ema_spread"] = _ema_spread(closes, self.ema_fast, self.ema_slow)

        macd_line, macd_signal, macd_hist = _macd(closes)
        out["macd_line"] = macd_line
        out["macd_signal"] = macd_signal
        out["macd_histogram"] = macd_hist

        stoch_k, stoch_d = _stochastic(highs, lows, closes)
        out["stochastic_k"] = stoch_k
        out["stochastic_d"] = stoch_d

        out["adx"] = _adx(highs, lows, closes)

        # ── Momentum (4 features) ──
        for p in self.momentum_periods:
            out[f"momentum_{p}"] = _momentum(closes, p) if n > p else 0.0
        out["roc_10"] = _rate_of_change(closes, 10)

        # ── Volume indicators (3 features) ──
        out["volume_spike"] = _volume_spike(volumes, self.volume_spike_window)
        out["obv_slope"] = _obv_slope(closes, volumes, 10)
        out["vwap_distance"] = _vwap_distance(closes, volumes, 20)

        # ── Price structure (3 features) ──
        boll_b, _ = _bollinger(closes, 20)
        out["bollinger_pct_b"] = boll_b
        out["price_position"] = _price_position(highs, lows, closes, 20)
        out["gap_pct"] = _gap_pct(opens, closes)

        # ── Candlestick patterns (4 features) ──
        body_r, upper_s, lower_s, engulf = _candlestick_features(opens, highs, lows, closes)
        out["candle_body_ratio"] = body_r
        out["candle_upper_shadow"] = upper_s
        out["candle_lower_shadow"] = lower_s
        out["candle_engulfing"] = engulf

        # ── Additional oscillators (2 features) ──
        out["williams_r"] = _williams_r(highs, lows, closes)
        out["mfi"] = _mfi(highs, lows, closes, volumes)

        # ── Microstructure features (6 features) ──
        # Intraday seasonality (sin/cos of minute-of-day for NSE session)
        sin_mod, cos_mod = _intraday_seasonality(bars)
        out["minute_of_day_sin"] = sin_mod
        out["minute_of_day_cos"] = cos_mod

        # Hurst exponent: trend vs mean-reversion regime indicator
        out["hurst_exponent"] = _hurst_exponent(closes, max_lag=20)

        # Volume profile deviation from POC (Point of Control)
        out["volume_profile_deviation"] = _volume_profile_deviation(closes, volumes, period=50)

        # Volatility of volatility (regime change detector)
        out["vol_of_vol"] = _vol_of_vol(closes, inner_window=5, outer_window=20)

        # Rolling volatility (20-period) — named to match feature specs
        out["rolling_vol_20"] = _rolling_volatility(closes, 20)

        # ── Raw reference (2 features, normalized in training) ──
        out["close"] = float(closes[-1])
        out["volume"] = float(volumes[-1]) if len(volumes) else 0.0

        return out
