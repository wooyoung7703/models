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
) -> Tuple[np.ndarray, np.ndarray]:
    X_tab, y_tab, feat_names = build_tabular_dataset(days=days, interval=interval, feature_fill_value=feature_fill_value)
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
        "[dataset][seq_reg] source=follow(tabular) interval=%s days=%d rows=%d seq_len=%d sequences=%d features=%d",
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
) -> Tuple[np.ndarray, np.ndarray]:
    X_tab, y_tab, feat_names = build_bottom_tabular_dataset(
        days=days,
        interval=interval,
        past_window=past_window,
        future_window=future_window,
        min_gap=min_gap,
        tolerance_pct=tolerance_pct,
        feature_fill_value=feature_fill_value,
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
        "[dataset][seq_bottom] source=follow(bottom_tabular) interval=%s days=%d rows=%d seq_len=%d sequences=%d features=%d class_counts(pos=%d,neg=%d)",
        interval, days, n, seq_len, X.shape[0], f, pos, neg
    )
    return X, y
import logging
from typing import Tuple, List, Optional
import numpy as np

from .dataset import build_tabular_dataset

log = logging.getLogger(__name__)


def build_sequence_dataset(
    days: int = 14,
    seq_len: int = 30,
    symbol: Optional[str] = None,
    exchange_type: Optional[str] = None,
    interval: str = "1m",
    feature_fill_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a simple sequence dataset (N, seq_len, F), (N,) using the tabular X,y.
    Window is built so that y[i] aligns with the end of window X[i-seq_len+1:i+1].

    If not enough data in DB, it falls back to a tiny synthetic dataset for quick pipeline check.
    """
    try:
        X_tab, y_tab, feat_names = build_tabular_dataset(
            days=days,
            symbol=symbol,
            exchange_type=exchange_type,
            interval=interval,
            feature_fill_value=feature_fill_value,
        )
        n, f = X_tab.shape
        if n < seq_len + 10:
            raise RuntimeError(f"Insufficient real rows for sequence dataset: n={n}, seq_len={seq_len}")
        X_seq: List[np.ndarray] = []
        y_seq: List[float] = []
        for i in range(seq_len - 1, n):
            X_seq.append(X_tab[i - seq_len + 1 : i + 1])
            y_seq.append(y_tab[i])
        X = np.stack(X_seq).astype(np.float32)
        y = np.array(y_seq, dtype=np.float32)
        log.info(
            "[dataset][seq_reg] source=DB interval=%s days=%d rows=%d seq_len=%d sequences=%d features=%d",
            interval,
            days,
            n,
            seq_len,
            X.shape[0],
            f,
        )
        return X, y
    except Exception as e:
        log.warning("Falling back to synthetic sequence dataset due to: %s", e)
        # Tiny synthetic dataset to verify the training loop. Non-sense data.
        N = 128
        F = 16
        X = np.random.randn(N, seq_len, F).astype(np.float32)
        w = np.random.randn(F).astype(np.float32)
        y = (X[:, -1, :] @ w) * 0.01 + 0.001 * np.random.randn(N).astype(np.float32)
        log.info("[dataset][seq_reg] source=synthetic sequences=%d features=%d", N, F)
        return X, y


def build_bottom_sequence_dataset(
    days: int = 14,
    seq_len: int = 30,
    symbol: Optional[str] = None,
    exchange_type: Optional[str] = None,
    interval: str = "1m",
    past_window: int = 15,
    future_window: int = 60,
    min_gap: int = 20,
    tolerance_pct: float = 0.004,
    feature_fill_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build classification sequences for bottom detection.

    Steps:
    1. Build tabular bottom dataset (X_tab, y_tab).
    2. Construct sliding windows of length seq_len ending at index i.
    3. Assign label y_tab[i] to the sequence ending at i (no future leakage; label already derived using future_window inside builder).
    4. Return (X_seq, y_seq).

    If insufficient rows, fall back to synthetic binary dataset.
    """
    try:
        from .dataset import build_bottom_tabular_dataset
    except Exception:
        # script mode fallback
        import sys
        sys.path.append('.')
        from backend.app.training.dataset import build_bottom_tabular_dataset
    try:
        X_tab, y_tab, feat_names = build_bottom_tabular_dataset(
            days=days,
            symbol=symbol,
            exchange_type=exchange_type,
            interval=interval,
            past_window=past_window,
            future_window=future_window,
            min_gap=min_gap,
            tolerance_pct=tolerance_pct,
            feature_fill_value=feature_fill_value,
        )
        n = len(X_tab)
        if n < seq_len + 10:
            raise RuntimeError(f"Insufficient rows for bottom sequence dataset: n={n}, seq_len={seq_len}")
        X_seq: List[np.ndarray] = []
        y_seq: List[int] = []
        for i in range(seq_len - 1, n):
            window = X_tab[i - seq_len + 1 : i + 1]
            X_seq.append(window)
            y_seq.append(int(y_tab[i]))
        X = np.stack(X_seq).astype(np.float32)
        y = np.array(y_seq, dtype=np.float32)
        pos = int((y == 1).sum()); neg = int((y == 0).sum())
        log.info(
            "[dataset][seq_bottom] source=DB interval=%s days=%d rows=%d seq_len=%d sequences=%d features=%d class_counts(pos=%d,neg=%d) labeling(past=%d,future=%d,min_gap=%d,tol=%.4f)",
            interval,
            days,
            n,
            seq_len,
            X.shape[0],
            X.shape[2],
            pos,
            neg,
            past_window,
            future_window,
            min_gap,
            tolerance_pct,
        )
        return X, y
    except Exception as e:
        log.warning("Falling back to synthetic bottom sequence dataset due to: %s", e)
        N = 128
        F = 16
        X = np.random.randn(N, seq_len, F).astype(np.float32)
        # Synthetic sparse positives
        y = (np.random.rand(N) > 0.95).astype(np.float32)
        pos = int((y == 1).sum()); neg = int((y == 0).sum())
        log.info("[dataset][seq_bottom] source=synthetic sequences=%d features=%d class_counts(pos=%d,neg=%d)", N, F, pos, neg)
        return X, y
