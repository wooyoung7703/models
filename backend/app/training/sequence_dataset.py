import logging
from typing import Tuple, List
import numpy as np

from .dataset import build_tabular_dataset, build_bottom_tabular_dataset

log = logging.getLogger(__name__)


def build_sequence_dataset(
    days: int = 14,
    seq_len: int = 30,
    interval: str = "1m",
    feature_fill_value: float = 0.0,
    *,
    scale: bool = False,
    feature_subset: list | None = None,
    cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (N, T, F) sequences and aligned targets from tabular X,y.

    This implementation intentionally matches dataset.build_tabular_dataset signature (no symbol/exchange args)
    to avoid API mismatch and synthetic fallback during tests.
    """
    X_tab, y_tab, feat_names = build_tabular_dataset(days=days, interval=interval, feature_fill_value=feature_fill_value, scale=scale, feature_subset=feature_subset, cache=cache)
    n, f = X_tab.shape
    if n < seq_len + 10:
        raise RuntimeError(f"Insufficient rows for sequence dataset: n={n}, seq_len={seq_len}")
    X_seq: List[np.ndarray] = []
    y_seq: List[float] = []
    for i in range(seq_len - 1, n):
        X_seq.append(X_tab[i - seq_len + 1 : i + 1])
        y_seq.append(y_tab[i])
    X = np.stack(X_seq).astype(np.float32)
    y = np.array(y_seq, dtype=np.float32)
    log.info(
        "[dataset][seq_reg] interval=%s days=%d rows=%d seq_len=%d sequences=%d features=%d",
        interval, days, n, seq_len, X.shape[0], f
    )
    return X, y


def build_bottom_sequence_dataset(
    days: int = 14,
    seq_len: int = 30,
    interval: str = "1m",
    past_window: int = 15,
    future_window: int = 60,
    min_gap: int = 20,
    tolerance_pct: float = 0.004,
    feature_fill_value: float = 0.0,
    *,
    scale: bool = False,
    feature_subset: list | None = None,
    cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build classification sequences for bottom detection from tabular bottom dataset.

    Matches dataset.build_bottom_tabular_dataset signature (no symbol/exchange args).
    """
    X_tab, y_tab, feat_names = build_bottom_tabular_dataset(
        days=days,
        interval=interval,
        past_window=past_window,
        future_window=future_window,
        min_gap=min_gap,
        tolerance_pct=tolerance_pct,
        feature_fill_value=feature_fill_value,
        scale=scale,
        feature_subset=feature_subset,
        cache=cache,
    )
    n, f = X_tab.shape
    if n < seq_len + 10:
        raise RuntimeError(f"Insufficient rows for bottom sequence dataset: n={n}, seq_len={seq_len}")
    X_seq: List[np.ndarray] = []
    y_seq: List[int] = []
    for i in range(seq_len - 1, n):
        X_seq.append(X_tab[i - seq_len + 1 : i + 1])
        y_seq.append(int(y_tab[i]))
    X = np.stack(X_seq).astype(np.float32)
    y = np.array(y_seq, dtype=np.float32)
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    log.info(
        "[dataset][seq_bottom] interval=%s days=%d rows=%d seq_len=%d sequences=%d features=%d class_counts(pos=%d,neg=%d)",
        interval, days, n, seq_len, X.shape[0], f, pos, neg
    )
    return X, y
