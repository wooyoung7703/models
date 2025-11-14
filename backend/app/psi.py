import math
from typing import Dict, List, Tuple, Optional
import numpy as np
from sqlmodel import Session, select

from .models import Candle


BASE_FIELDS = {
    "id","symbol","exchange_type","interval","open_time","close_time",
    "open","high","low","close","volume","trades"
}


def _get_feature_values(
    session: Session,
    symbol: str,
    exchange_type: str,
    interval: str,
    features: List[str],
    start_ts,
    end_ts,
) -> Dict[str, np.ndarray]:
    q = (
        select(Candle)
        .where((Candle.symbol == symbol) & (Candle.exchange_type == exchange_type) & (Candle.interval == interval)
               & (Candle.open_time >= start_ts) & (Candle.open_time < end_ts))  # type: ignore[func-returns-value]
        .order_by(Candle.open_time.asc())
    )
    rows = list(session.exec(q).all())
    out: Dict[str, List[float]] = {f: [] for f in features}
    for r in rows:
        for f in features:
            try:
                v = getattr(r, f)
                if v is not None and isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    out[f].append(float(v))
            except Exception:
                pass
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def _psi_from_arrays(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> Optional[float]:
    if ref is None or cur is None or len(ref) < 50 or len(cur) < 50:
        return None
    try:
        # Quantile-based bins from reference
        qs = np.linspace(0, 1, bins + 1)
        edges = np.quantile(ref, qs)
        # ensure strictly increasing edges
        edges = np.unique(edges)
        if len(edges) <= 2:  # not enough variability
            return 0.0
        # histogram proportions
        ref_hist, _ = np.histogram(ref, bins=edges)
        cur_hist, _ = np.histogram(cur, bins=edges)
        # Last bin may exclude right edge; include it
        if len(cur_hist) != len(ref_hist):
            m = min(len(cur_hist), len(ref_hist))
            ref_hist = ref_hist[:m]
            cur_hist = cur_hist[:m]
        # Convert to proportions with small epsilon
        eps = 1e-6
        p_ref = np.maximum(ref_hist / max(1, ref.size), eps)
        p_cur = np.maximum(cur_hist / max(1, cur.size), eps)
        psi = float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))
        return psi
    except Exception:
        return None


def compute_feature_psi(
    session: Session,
    symbol: str,
    exchange_type: str,
    interval: str,
    features: List[str],
    ref_range: Tuple,
    cur_range: Tuple,
    bins: int = 10,
) -> Dict[str, Optional[float]]:
    ref_vals = _get_feature_values(session, symbol, exchange_type, interval, features, ref_range[0], ref_range[1])
    cur_vals = _get_feature_values(session, symbol, exchange_type, interval, features, cur_range[0], cur_range[1])
    out: Dict[str, Optional[float]] = {}
    for f in features:
        out[f] = _psi_from_arrays(ref_vals.get(f, np.array([])), cur_vals.get(f, np.array([])), bins=bins)
    return out
