"""Reusable heuristic bottom-score calculation used by RealTimePredictor.

이 모듈은 기존 `predictor.py` 내부에 흩어져 있던 스코어 계산식을
독립적으로 분리해 두 가지 이점을 제공합니다.

1. **가독성/테스트 용이성**: pure function 형태로 노출하므로 단위 테스트에서
   캔들 모형 없이도 입력/출력 관계를 검증할 수 있습니다.
2. **차후 대체 용이성**: 동일 인터페이스를 유지한 채 학습형 경량 모델로
   교체하거나 가중치를 조정하는 작업이 쉬워집니다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class HeuristicScore:
    prob: float
    logit: float
    components: Dict[str, float]


def _to_float(row: Any, name: str, default: float) -> float:
    try:
        value = getattr(row, name, default)
    except Exception:
        return default
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


DEFAULT_WEIGHTS = {
    "rsi": 1.8,
    "bb": 1.5,
    "drawdown": 1.2,
    "macd": 0.6,
    "volume": 0.8,
    "williams": 0.9,
    "bias": -2.0,
}


def compute_bottom_score(row: Any, *, price_override: Optional[float] = None) -> HeuristicScore:
    close = float(price_override) if price_override is not None else _to_float(row, "close", 0.0)
    rsi = _to_float(row, "rsi_14", 50.0)
    bb_pct_b = _to_float(row, "bb_pct_b_20_2", 0.5)
    dd20 = _to_float(row, "drawdown_from_max_20", 0.0)
    macd_hist = _to_float(row, "macd_hist", 0.0)
    volz = _to_float(row, "vol_z_20", 0.0)
    willr = _to_float(row, "williams_r_14", -50.0)

    s_rsi = max(0.0, min(1.0, (30.0 - rsi) / 30.0))
    s_bb = max(0.0, min(1.0, (0.25 - bb_pct_b) / 0.25))
    s_dd = max(0.0, min(1.0, -dd20 / 0.05))
    s_macd = max(0.0, min(1.0, -macd_hist / (abs(macd_hist) + 1e-6))) * 0.5
    s_vol = max(0.0, min(1.0, (volz - 1.0) / 3.0))
    s_wr = max(0.0, min(1.0, (-80.0 - willr) / 20.0))

    logit = (
        DEFAULT_WEIGHTS["rsi"] * s_rsi
        + DEFAULT_WEIGHTS["bb"] * s_bb
        + DEFAULT_WEIGHTS["drawdown"] * s_dd
        + DEFAULT_WEIGHTS["macd"] * s_macd
        + DEFAULT_WEIGHTS["volume"] * s_vol
        + DEFAULT_WEIGHTS["williams"] * s_wr
        + DEFAULT_WEIGHTS["bias"]
    )
    prob = float(_sigmoid(logit))

    components = {
        "close": close,
        "rsi_14": rsi,
        "bb_pct_b_20_2": bb_pct_b,
        "drawdown_from_max_20": dd20,
        "macd_hist": macd_hist,
        "vol_z_20": volz,
        "williams_r_14": willr,
        "s_rsi": s_rsi,
        "s_bb": s_bb,
        "s_dd": s_dd,
        "s_macd": s_macd,
        "s_vol": s_vol,
        "s_wr": s_wr,
        "logit": logit,
    }
    return HeuristicScore(prob=prob, logit=logit, components=components)


__all__ = ["HeuristicScore", "compute_bottom_score", "DEFAULT_WEIGHTS"]
