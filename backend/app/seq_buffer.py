"""
Lightweight per-symbol rolling sequence buffers for sequence models.

We capture a compact feature vector from each closed candle and keep the last N.
Adapters can read from these buffers to build inputs for LSTM/Transformer.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

from .core.config import settings


class SequenceBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._q: Deque[List[float]] = deque(maxlen=self.capacity)

    def append(self, vec: List[float]) -> None:
        self._q.append(vec)

    def to_list(self) -> List[List[float]]:
        return list(self._q)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._q)


_buffers: Dict[str, SequenceBuffer] = {}


def get_buffer(symbol: str) -> SequenceBuffer:
    sym = symbol.lower()
    buf = _buffers.get(sym)
    if buf is None:
        buf = SequenceBuffer(settings.SEQ_LEN)
        _buffers[sym] = buf
    return buf


def extract_vector_from_candle(candle: "Candle") -> List[float]:  # type: ignore[name-defined]
    """Return sequence feature vector.

    Expanded to 16 dimensions to better feed the Transformer checkpoint (feature_dim=16).
    LSTM adapter will still pad to larger feature_dim (e.g., 55) until retraining aligns.
    Order must remain stable; append new features only at the end.
    """
    def _f(name: str, default: float) -> float:
        try:
            v = getattr(candle, name)
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    vec = [
        _f("close", 0.0),            # 0 price
        _f("rsi_14", 50.0),          # 1 momentum
        _f("bb_pct_b_20_2", 0.5),    # 2 volatility position
        _f("macd_hist", 0.0),        # 3 momentum histogram
        _f("vol_z_20", 0.0),         # 4 volume anomaly
        _f("williams_r_14", -50.0),  # 5 oversold metric
        _f("drawdown_from_max_20", 0.0),  # 6 local drawdown
        _f("atr_14", 0.0),           # 7 volatility (range)
        _f("cci_20", 0.0),           # 8 typical price deviation
        _f("run_up", 0.0),           # 9 consecutive up closes
        _f("run_down", 0.0),         # 10 consecutive down closes
        _f("obv", 0.0),              # 11 on-balance volume
        _f("mfi_14", 50.0),          # 12 money flow index
        _f("cmf_20", 0.0),           # 13 chaikin money flow
        _f("body_pct_of_range", 0.0),# 14 candle body proportion
        _f("vwap_20_dev", 0.0),      # 15 deviation from short VWAP
    ]
    return vec
