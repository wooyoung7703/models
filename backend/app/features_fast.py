from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable

import numpy as np

FAST_FEATURE_KEYS = [
    "ret_5",
    "ret_15",
    "ma_5",
    "ma_20",
    "ma_50",
    "ma_200",
    "vol_20",
    "vol_50",
    "bb_upper_20_2",
    "bb_lower_20_2",
    "bb_pct_b_20_2",
    "bb_bandwidth_20_2",
    "bb_pct_b_50_2",
    "bb_bandwidth_50_2",
    "stoch_k_14_3",
    "stoch_d_14_3",
    "williams_r_14",
    "cmf_20",
    "vol_z_20",
    "dist_min_close_20",
    "dist_min_close_50",
    "dist_min_close_100",
    "drawdown_from_max_20",
]


def _deque_to_array(values: Deque[float]) -> np.ndarray:
    if not values:
        return np.empty(0, dtype=np.float64)
    return np.fromiter(values, dtype=np.float64)


def _rolling_return(closes: np.ndarray, window: int) -> float:
    if closes.size >= window and closes[-window] != 0:
        return (closes[-1] - closes[-window]) / closes[-window]
    return 0.0


def _moving_average(closes: np.ndarray, window: int) -> float:
    if closes.size >= window:
        return float(closes[-window:].mean())
    return 0.0


def _volatility(returns: np.ndarray, window: int) -> float:
    if returns.size >= window:
        segment = returns[-window:]
        mean = segment.mean()
        var = np.mean((segment - mean) ** 2)
        return float(np.sqrt(var))
    return 0.0


def _bollinger(closes: np.ndarray, window: int) -> Dict[str, float]:
    if closes.size >= window:
        segment = closes[-window:]
        mean = segment.mean()
        std = segment.std(ddof=0)
        upper = mean + 2 * std
        lower = mean - 2 * std
        if window == 20:
            pct = (closes[-1] - lower) / (upper - lower) if upper != lower else 0.5
            bandwidth = (upper - lower) / mean if mean != 0 else 0.0
            return {
                "bb_upper_20_2": float(upper),
                "bb_lower_20_2": float(lower),
                "bb_pct_b_20_2": float(pct),
                "bb_bandwidth_20_2": float(bandwidth),
            }
        pct = (closes[-1] - lower) / (upper - lower) if upper != lower else 0.5
        bandwidth = (upper - lower) / mean if mean != 0 else 0.0
        return {
            "bb_pct_b_50_2": float(pct),
            "bb_bandwidth_50_2": float(bandwidth),
        }
    if window == 20:
        return {
            "bb_upper_20_2": 0.0,
            "bb_lower_20_2": 0.0,
            "bb_pct_b_20_2": 0.5,
            "bb_bandwidth_20_2": 0.0,
        }
    return {
        "bb_pct_b_50_2": 0.5,
        "bb_bandwidth_50_2": 0.0,
    }


def _stochastic(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Dict[str, float]:
    window = 14
    if highs.size < window or lows.size < window or closes.size < window:
        return {"stoch_k_14_3": 50.0, "stoch_d_14_3": 50.0}
    high_slice = highs[-window:]
    low_slice = lows[-window:]
    hh = high_slice.max()
    ll = low_slice.min()
    if hh != ll:
        k = (closes[-1] - ll) / (hh - ll) * 100.0
    else:
        k = 50.0
    k_values = []
    max_shift = min(3, closes.size - window + 1)
    for shift in range(max_shift):
        start = -(window + shift)
        end = None if shift == 0 else -shift
        sub_high = highs[start:end]
        sub_low = lows[start:end]
        sub_close = closes[start:end]
        if sub_high.size < window:
            break
        hh_sub = sub_high.max()
        ll_sub = sub_low.min()
        if hh_sub != ll_sub:
            k_values.append((sub_close[-1] - ll_sub) / (hh_sub - ll_sub) * 100.0)
        else:
            k_values.append(50.0)
    d = float(np.mean(k_values)) if k_values else k
    return {"stoch_k_14_3": float(k), "stoch_d_14_3": d}


def _williams_r(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
    window = 14
    if highs.size < window or lows.size < window or closes.size < window:
        return -50.0
    high_slice = highs[-window:]
    low_slice = lows[-window:]
    hh = high_slice.max()
    ll = low_slice.min()
    if hh == ll:
        return -50.0
    return float(-100.0 * (hh - closes[-1]) / (hh - ll))


def _cmf(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> float:
    window = 20
    if (
        highs.size < window
        or lows.size < window
        or closes.size < window
        or volumes.size < window
    ):
        return 0.0
    highs_slice = highs[-window:]
    lows_slice = lows[-window:]
    closes_slice = closes[-window:]
    volumes_slice = volumes[-window:]
    vol_sum = volumes_slice.sum()
    if vol_sum == 0:
        return 0.0
    range_slice = highs_slice - lows_slice
    mfm = np.zeros_like(closes_slice)
    valid = range_slice != 0
    mfm[valid] = (
        (closes_slice[valid] - lows_slice[valid])
        - (highs_slice[valid] - closes_slice[valid])
    ) / range_slice[valid]
    adl_components = mfm * volumes_slice
    return float(adl_components.sum() / vol_sum)


def _volume_z(volumes: np.ndarray, window: int, current_volume: float) -> float:
    if volumes.size >= window:
        seg = volumes[-window:]
        mean = seg.mean()
        std = seg.std(ddof=0)
        if std == 0:
            return 0.0
        return float((current_volume - mean) / std)
    return 0.0


def _distance_metrics(closes: np.ndarray, windows: Iterable[int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for window in windows:
        key = f"dist_min_close_{window}"
        if closes.size >= window:
            segment = closes[-window:]
            mn = segment.min()
            out[key] = float((closes[-1] - mn) / mn) if mn != 0 else 0.0
        else:
            out[key] = 0.0
    if closes.size >= 20:
        segment = closes[-20:]
        mx = segment.max()
        out["drawdown_from_max_20"] = float((closes[-1] - mx) / mx) if mx != 0 else 0.0
    else:
        out["drawdown_from_max_20"] = 0.0
    return out


def compute_fast_features(
    closes: Deque[float],
    highs: Deque[float],
    lows: Deque[float],
    volumes: Deque[float],
    returns: Deque[float],
    close: float,
    volume: float,
) -> Dict[str, float]:
    """Compute core rolling indicators using NumPy-based helpers."""

    closes_arr = _deque_to_array(closes)
    highs_arr = _deque_to_array(highs)
    lows_arr = _deque_to_array(lows)
    volumes_arr = _deque_to_array(volumes)
    returns_arr = _deque_to_array(returns)

    features: Dict[str, float] = {
        "ret_5": _rolling_return(closes_arr, 5),
        "ret_15": _rolling_return(closes_arr, 15),
        "ma_5": _moving_average(closes_arr, 5),
        "ma_20": _moving_average(closes_arr, 20),
        "ma_50": _moving_average(closes_arr, 50),
        "ma_200": _moving_average(closes_arr, 200),
        "vol_20": _volatility(returns_arr, 20),
        "vol_50": _volatility(returns_arr, 50),
    }
    features.update(_bollinger(closes_arr, 20))
    features.update(_bollinger(closes_arr, 50))
    features.update(_stochastic(highs_arr, lows_arr, closes_arr))
    features["williams_r_14"] = _williams_r(highs_arr, lows_arr, closes_arr)
    features["cmf_20"] = _cmf(highs_arr, lows_arr, closes_arr, volumes_arr)
    features["vol_z_20"] = _volume_z(volumes_arr, 20, volume)
    features.update(_distance_metrics(closes_arr, (20, 50, 100)))
    return features
