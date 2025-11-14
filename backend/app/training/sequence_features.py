"""Canonical ordered feature lists for real Candle-derived training and inference.

This module centralizes the mapping so that:
- Sequence model training (LSTM / Transformer) uses EXACT feature order.
- Runtime sequence buffer construction / adapters can reference the same order.
- Stacking and calibration can rely on stable column ordering.

Versioning strategy:
FULL_FEATURE_SET_V1 is the initial curated set of technical/price/volume derived
features. Subsets (e.g. SEQUENCE_FEATURES_16) are used when a smaller dimensionality
is needed (current Transformer checkpoint expects 16). A future retrain of LSTM
will likely move from padded(55) to a true subset (32 or 40) to reduce noise.

Guidelines for adding a new feature:
1. Append ONLY at the end (preserve indices of existing features).
2. Update the appropriate version constant (e.g., create FULL_FEATURE_SET_V2).
3. Bump MODEL_METADATA version in training artifacts to guard mismatches.

Runtime alignment:
- seq_buffer.extract_vector_from_candle currently outputs the same 16 feature subset.
- Adapters infer feature_dim from checkpoint; LSTM pads if checkpoint expects more.

"""
from __future__ import annotations
from typing import List

# 16-D subset used by current runtime Transformer (matches seq_buffer order)
SEQUENCE_FEATURES_16: List[str] = [
    "close",                # 0 price
    "rsi_14",               # 1 momentum
    "bb_pct_b_20_2",        # 2 volatility position
    "macd_hist",            # 3 momentum histogram
    "vol_z_20",             # 4 volume anomaly
    "williams_r_14",        # 5 oversold metric
    "drawdown_from_max_20", # 6 local drawdown
    "atr_14",               # 7 volatility (range)
    "cci_20",               # 8 typical price deviation
    "run_up",               # 9 consecutive up closes
    "run_down",             # 10 consecutive down closes
    "obv",                  # 11 on-balance volume
    "mfi_14",               # 12 money flow index
    "cmf_20",               # 13 chaikin money flow
    "body_pct_of_range",    # 14 candle anatomy proportion
    "vwap_20_dev",          # 15 deviation from short VWAP
]

# Extended candidate feature set (first retrain target ~32 features)
# Includes additional trend, multi-timeframe volatility and band context.
FULL_FEATURE_SET_V1: List[str] = [
    # Core price / returns
    "close", "ret_1", "ret_5", "ret_15",
    # Trend / moving averages
    "ma_5", "ma_20", "ema_12", "ema_26",
    # Momentum & oscillators
    "rsi_14", "rsi_7", "rsi_21", "macd_hist", "williams_r_14", "cci_20",
    # Volatility & ranges
    "atr_14", "vol_20", "vol_50", "bb_pct_b_20_2", "bb_bandwidth_20_2",
    # Volume & flow
    "vol_z_20", "obv", "mfi_14", "cmf_20",
    # Structure / pattern
    "drawdown_from_max_20", "dist_min_close_20", "run_up", "run_down", "body_pct_of_range",
    # VWAP deviation
    "vwap_20_dev",
    # Additional derived band/price position for robustness
    "price_to_ma_20", "price_to_ma_50"
]

# Simple utility to build vector from candle-like object
def build_vector_from_candle(candle, feature_list: List[str], default: float = 0.0) -> List[float]:
    out: List[float] = []
    for name in feature_list:
        try:
            v = getattr(candle, name)
            out.append(float(v if v is not None else default))
        except Exception:
            out.append(default)
    return out

__all__ = [
    "SEQUENCE_FEATURES_16",
    "FULL_FEATURE_SET_V1",
    "build_vector_from_candle",
]
