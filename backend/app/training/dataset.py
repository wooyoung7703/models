import logging
import os
import hashlib
from typing import List, Tuple, Optional

import numpy as np
try:
    from sklearn.preprocessing import StandardScaler  # type: ignore
except Exception:
    StandardScaler = None  # type: ignore

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
    *,
    scale: bool = False,
    feature_subset: Optional[List[str]] = None,
    cache: bool = False,
    cache_dir: str = "backend/app/training/cache",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build tabular dataset with optional scaling, feature sub-selection and caching.

    Since current implementation is synthetic, caching mainly avoids recomputation when running many HPO trials.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"tabular|days={days}|interval={interval}|target={target}|nfeat=55|scale={int(scale)}|subset={','.join(feature_subset) if feature_subset else 'all'}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"tab_{cache_hash}.npz")
    if cache and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            X = data['X']; y = data['y']; features = data['features'].tolist()
            log.info(f"[dataset][tabular][cache_hit] path={cache_path} rows={len(X)} features={len(features)}")
            return X, y, features
        except Exception:
            pass
    X, y, features = _build_synthetic_tabular()
    # feature subset filtering
    if feature_subset:
        # map requested features to indices if present
        idx = [features.index(f) for f in feature_subset if f in features]
        if len(idx) == 0:
            log.warning("[dataset][tabular] feature_subset provided but none matched; using all features")
        else:
            X = X[:, idx]
            features = [features[i] for i in idx]
            log.info(f"[dataset][tabular] feature_subset applied count={len(features)}")
    # scaling
    if scale and StandardScaler is not None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
        log.info("[dataset][tabular] StandardScaler applied")
    elif scale:
        log.warning("[dataset][tabular] scale requested but sklearn not available")
    if cache:
        try:
            np.savez_compressed(cache_path, X=X, y=y, features=np.array(features))
            log.info(f"[dataset][tabular][cache_save] path={cache_path}")
        except Exception as e:
            log.warning(f"[dataset][tabular] cache save failed: {e}")
    return X, y, features


def build_bottom_tabular_dataset(
    days: int = 14,
    interval: str = "1m",
    past_window: int = 15,
    future_window: int = 60,
    min_gap: int = 20,
    tolerance_pct: float = 0.004,
    feature_fill_value: float = 0.0,
    *,
    scale: bool = False,
    feature_subset: Optional[List[str]] = None,
    cache: bool = False,
    cache_dir: str = "backend/app/training/cache",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Synthetic bottom classification dataset with optional scaling/subset/caching."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"bottom|days={days}|interval={interval}|nfeat=55|scale={int(scale)}|subset={','.join(feature_subset) if feature_subset else 'all'}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"bottom_{cache_hash}.npz")
    if cache and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            X = data['X']; y = data['y']; features = data['features'].tolist()
            pos = int((y == 1).sum()); neg = int((y == 0).sum())
            log.info(f"[dataset][bottom][cache_hit] path={cache_path} rows={len(X)} pos={pos} neg={neg}")
            return X, y, features
        except Exception:
            pass
    n_rows = 2000; n_features = 55
    X = np.random.randn(n_rows, n_features).astype(np.float32)
    y = (np.random.rand(n_rows) < 0.01).astype(np.int32)
    features = [f"f{i}" for i in range(n_features)]
    # subset
    if feature_subset:
        idx = [features.index(f) for f in feature_subset if f in features]
        if idx:
            X = X[:, idx]
            features = [features[i] for i in idx]
            log.info(f"[dataset][bottom] feature_subset applied count={len(features)}")
        else:
            log.warning("[dataset][bottom] feature_subset provided but none matched; using all features")
    # scale
    if scale and StandardScaler is not None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
        log.info("[dataset][bottom] StandardScaler applied")
    elif scale:
        log.warning("[dataset][bottom] scale requested but sklearn not available")
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    log.info(
        "[dataset][bottom] source=synthetic interval=%s rows=%d features=%d class_counts(pos=%d,neg=%d) labeling(past=%d,future=%d,min_gap=%d,tol=%.4f)",
        interval, X.shape[0], X.shape[1], pos, neg, past_window, future_window, min_gap, tolerance_pct
    )
    if cache:
        try:
            np.savez_compressed(cache_path, X=X, y=y, features=np.array(features))
            log.info(f"[dataset][bottom][cache_save] path={cache_path}")
        except Exception as e:
            log.warning(f"[dataset][bottom] cache save failed: {e}")
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

