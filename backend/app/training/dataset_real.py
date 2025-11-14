import logging
from typing import List, Tuple, Optional
from datetime import datetime, timedelta, timezone
import numpy as np
from sqlmodel import Session, select

from ..core.config import settings
from ..models import Candle
from ..db import engine  # ensure we always have an engine

log = logging.getLogger(__name__)


def _fetch_candles(days: int, interval: str) -> List[Candle]:
    """Fetch candles for the configured symbol/interval ordered by open_time ascending.

    Enforces a lower bound on open_time based on the provided number of days.
    This ensures training/inference can be limited to a recent window (e.g., 21 days).
    """
    # Compute cutoff timestamp for filtering
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))
    except Exception:
        cutoff = None
    with Session(engine) as session:  # type: ignore[arg-type]
        q = (
            select(Candle)
            .where(
                (Candle.symbol == settings.SYMBOL)
                & (Candle.exchange_type == settings.EXCHANGE_TYPE)
                & (Candle.interval == interval)
                & ((Candle.open_time >= cutoff) if cutoff is not None else True)  # type: ignore[func-returns-value]
            )
        )
        rows = list(session.exec(q).all())
    # Ensure ascending chronological order for sequence slicing
    rows.sort(key=lambda r: r.open_time)
    return rows


def _vector_from_row(row: Candle, feature_list: List[str], default: float = 0.0) -> List[float]:
    out: List[float] = []
    for name in feature_list:
        try:
            v = getattr(row, name)
            out.append(float(v if v is not None else default))
        except Exception:
            out.append(default)
    return out


def build_tabular_dataset_real(
    *,
    days: int = 14,
    interval: str = "1m",
    target: str = "next_ret",
    feature_subset: Optional[List[str]] = None,
    scale: bool = False,  # unused; placeholder for parity
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rows = _fetch_candles(days=days, interval=interval)
    if not rows or len(rows) < 3:
        raise RuntimeError("Insufficient DB rows for real dataset")
    # Default feature list: if none provided, use a compact 16-D set
    if not feature_subset:
        try:
            from .sequence_features import SEQUENCE_FEATURES_16 as feature_subset  # type: ignore
        except Exception:
            feature_subset = [
                "close","rsi_14","bb_pct_b_20_2","macd_hist","vol_z_20","williams_r_14",
                "drawdown_from_max_20","atr_14","cci_20","run_up","run_down","obv","mfi_14","cmf_20",
                "body_pct_of_range","vwap_20_dev",
            ]
    X: List[List[float]] = []
    y: List[float] = []
    closes = [float(r.close) for r in rows]
    n = len(rows)
    for i in range(0, n - 1):
        X.append(_vector_from_row(rows[i], feature_subset))
        if target == "next_ret":
            c0 = closes[i]
            c1 = closes[i + 1]
            ret = (c1 - c0) / c0 if c0 != 0 else 0.0
            y.append(float(ret))
        else:
            y.append(0.0)
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    log.info("[dataset][real][tabular] rows=%d features=%d target=%s", X_arr.shape[0], X_arr.shape[1], target)
    return X_arr, y_arr, list(feature_subset)


def build_bottom_tabular_dataset_real(
    *,
    days: int = 14,
    interval: str = "1m",
    past_window: int = 15,
    future_window: int = 60,
    tolerance_pct: float = 0.004,
    feature_subset: Optional[List[str]] = None,
    scale: bool = False,  # unused; placeholder for parity
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rows = _fetch_candles(days=days, interval=interval)
    if not rows or len(rows) < (past_window + future_window + 2):
        raise RuntimeError("Insufficient DB rows for real bottom dataset")
    if not feature_subset:
        try:
            from .sequence_features import SEQUENCE_FEATURES_16 as feature_subset  # type: ignore
        except Exception:
            feature_subset = [
                "close","rsi_14","bb_pct_b_20_2","macd_hist","vol_z_20","williams_r_14",
                "drawdown_from_max_20","atr_14","cci_20","run_up","run_down","obv","mfi_14","cmf_20",
                "body_pct_of_range","vwap_20_dev",
            ]
    closes = [float(r.close) for r in rows]
    n = len(rows)
    X: List[List[float]] = []
    y: List[int] = []
    # Volatility-adaptive tolerance setup
    enable_vol = getattr(settings, 'ENABLE_VOL_LABELING', False)
    atr_field = getattr(settings, 'VOL_LABEL_ATR_FEATURE', 'atr_14')
    base_atr_pct = float(getattr(settings, 'VOL_LABEL_BASE_ATR_PCT', 0.01))
    min_scale = float(getattr(settings, 'VOL_LABEL_MIN_SCALE', 0.5))
    max_scale = float(getattr(settings, 'VOL_LABEL_MAX_SCALE', 2.0))

    for i in range(past_window, n - future_window - 1):
        # Past minimum condition
        past_min = min(closes[i - past_window : i + 1])
        is_local_min = closes[i] <= past_min
        # Future rebound
        fut_max = max(closes[i + 1 : i + 1 + future_window])
        rebound = (fut_max - closes[i]) / closes[i] if closes[i] != 0 else 0.0
        tol_i = tolerance_pct
        if enable_vol and closes[i] > 0:
            try:
                atr_val = getattr(rows[i], atr_field, None)
                if atr_val is not None and float(atr_val) > 0 and base_atr_pct > 0:
                    atr_pct = float(atr_val) / float(closes[i])
                    scale = max(min_scale, min(max_scale, atr_pct / base_atr_pct))
                    tol_i = float(tolerance_pct) * float(scale)
            except Exception:
                pass
        label = 1 if (is_local_min and rebound >= tol_i) else 0
        X.append(_vector_from_row(rows[i], feature_subset))
        y.append(label)
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    pos = int((y_arr == 1).sum()); neg = int((y_arr == 0).sum())
    log.info(
        "[dataset][real][bottom] rows=%d features=%d class_counts(pos=%d,neg=%d)",
        X_arr.shape[0], X_arr.shape[1], pos, neg
    )
    return X_arr, y_arr, list(feature_subset)


def build_bottom_ordinal_tabular_dataset_real(
    *,
    days: int = 14,
    interval: str = "1m",
    past_window: int = 15,
    future_window: int = 60,
    tolerance_pct: float = 0.004,
    strong_mult: float = 2.0,
    feature_subset: Optional[List[str]] = None,
    scale: bool = False,  # placeholder for parity
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Real-data ordinal bottom labels: 0=none, 1=weak(>=tol), 2=strong(>=strong_mult*tol).

    Uses the same volatility-adaptive tolerance logic as binary labeling.
    """
    rows = _fetch_candles(days=days, interval=interval)
    if not rows or len(rows) < (past_window + future_window + 2):
        raise RuntimeError("Insufficient DB rows for real bottom ordinal dataset")
    if not feature_subset:
        try:
            from .sequence_features import SEQUENCE_FEATURES_16 as feature_subset  # type: ignore
        except Exception:
            feature_subset = [
                "close","rsi_14","bb_pct_b_20_2","macd_hist","vol_z_20","williams_r_14",
                "drawdown_from_max_20","atr_14","cci_20","run_up","run_down","obv","mfi_14","cmf_20",
                "body_pct_of_range","vwap_20_dev",
            ]
    closes = [float(r.close) for r in rows]
    n = len(rows)
    X: List[List[float]] = []
    y: List[int] = []
    # Volatility-adaptive tolerance setup
    enable_vol = getattr(settings, 'ENABLE_VOL_LABELING', False)
    atr_field = getattr(settings, 'VOL_LABEL_ATR_FEATURE', 'atr_14')
    base_atr_pct = float(getattr(settings, 'VOL_LABEL_BASE_ATR_PCT', 0.01))
    min_scale = float(getattr(settings, 'VOL_LABEL_MIN_SCALE', 0.5))
    max_scale = float(getattr(settings, 'VOL_LABEL_MAX_SCALE', 2.0))

    for i in range(past_window, n - future_window - 1):
        past_min = min(closes[i - past_window : i + 1])
        is_local_min = closes[i] <= past_min
        fut_max = max(closes[i + 1 : i + 1 + future_window])
        rebound = (fut_max - closes[i]) / closes[i] if closes[i] != 0 else 0.0
        tol_i = tolerance_pct
        if enable_vol and closes[i] > 0:
            try:
                atr_val = getattr(rows[i], atr_field, None)
                if atr_val is not None and float(atr_val) > 0 and base_atr_pct > 0:
                    atr_pct = float(atr_val) / float(closes[i])
                    scale = max(min_scale, min(max_scale, atr_pct / base_atr_pct))
                    tol_i = float(tolerance_pct) * float(scale)
            except Exception:
                pass
        strong_tol = float(strong_mult) * float(tol_i)
        label = 0
        if is_local_min and rebound >= tol_i:
            label = 1
            if rebound >= strong_tol:
                label = 2
        X.append(_vector_from_row(rows[i], feature_subset))
        y.append(label)
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int32)
    c2 = int((y_arr == 2).sum()); c1 = int((y_arr == 1).sum()); c0 = int((y_arr == 0).sum())
    log.info(
        "[dataset][real][bottom_ord] rows=%d features=%d class_counts(c2=%d,c1=%d,c0=%d)",
        X_arr.shape[0], X_arr.shape[1], c2, c1, c0
    )
    return X_arr, y_arr, list(feature_subset)
