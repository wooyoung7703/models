import logging
from typing import List, Tuple

import numpy as np

log = logging.getLogger(__name__)


BASE_FIELDS = {
    "id","symbol","exchange_type","interval","open_time","close_time",
    "open","high","low","close","volume","trades"
}


def _build_synthetic_tabular(n_rows: int = 2000, n_features: int = 55) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(n_rows, n_features)).astype(np.float32)
    w = rng.normal(0, 0.5, size=(n_features,)).astype(np.float32)
    noise = rng.normal(0, 0.1, size=(n_rows,)).astype(np.float32)
    y = (X @ w) * 0.001 + noise
    features = [f"f{i}" for i in range(n_features)]
    log.info("[dataset][tabular] source=synthetic rows=%d features=%d", n_rows, n_features)
    return X, y.astype(np.float32), features


def build_tabular_dataset(
    days: int = 14,
    interval: str = "1m",
    target: str = "next_ret",
    feature_fill_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # Use synthetic dataset in current workspace
    return _build_synthetic_tabular()


def build_bottom_tabular_dataset(
    days: int = 14,
    interval: str = "1m",
    past_window: int = 15,
    future_window: int = 60,
    min_gap: int = 20,
    tolerance_pct: float = 0.004,
    feature_fill_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # Synthetic dataset with sparse positives
    n_rows = 2000; n_features = 55
    X = np.random.randn(n_rows, n_features).astype(np.float32)
    y = (np.random.rand(n_rows) < 0.01).astype(np.int32)
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    features = [f"f{i}" for i in range(n_features)]
    log.info(
        "[dataset][bottom] source=synthetic interval=%s rows=%d features=%d class_counts(pos=%d,neg=%d) labeling(past=%d,future=%d,min_gap=%d,tol=%.4f)",
        interval, n_rows, n_features, pos, neg, past_window, future_window, min_gap, tolerance_pct
    )
    return X, y, features


def train_val_split_time_order(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert 0 < val_ratio < 1
    n = len(X)
    if n == 0:
        raise RuntimeError("Empty dataset")
    split = max(1, int(n * (1 - val_ratio)))
    return X[:split], X[split:], y[:split], y[split:]

