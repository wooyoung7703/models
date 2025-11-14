import logging
import os
import hashlib
from typing import List, Tuple, Optional

import numpy as np
try:
    from sklearn.preprocessing import StandardScaler  # type: ignore
except Exception:
    StandardScaler = None  # type: ignore

try:
    from ..core.config import settings  # type: ignore
except Exception:
    class _FallbackSettings:
        ENABLE_VOL_LABELING = False
        VOL_LABEL_ATR_FEATURE = "atr_14"
        VOL_LABEL_BASE_ATR_PCT = 0.01
        VOL_LABEL_MIN_SCALE = 0.5
        VOL_LABEL_MAX_SCALE = 2.0

    settings = _FallbackSettings()  # type: ignore

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
    cache_key = (
        f"bottom|days={days}|interval={interval}|nfeat=55|scale={int(scale)}|"
        f"subset={','.join(feature_subset) if feature_subset else 'all'}|tol={tolerance_pct:.6f}|"
        f"vol={int(getattr(settings,'ENABLE_VOL_LABELING', False))}|"
        f"base_atr_pct={float(getattr(settings,'VOL_LABEL_BASE_ATR_PCT', 0.01)):.6f}|"
        f"min_scale={float(getattr(settings,'VOL_LABEL_MIN_SCALE', 0.5)):.3f}|"
        f"max_scale={float(getattr(settings,'VOL_LABEL_MAX_SCALE', 2.0)):.3f}"
    )
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
    rng = np.random.default_rng(123)
    X = rng.normal(0, 1, size=(n_rows, n_features)).astype(np.float32)
    # Synthetic volatility regime via ATR% approximation
    enable_vol = bool(getattr(settings, 'ENABLE_VOL_LABELING', False))
    base_atr_pct = float(getattr(settings, 'VOL_LABEL_BASE_ATR_PCT', 0.01))
    min_scale = float(getattr(settings, 'VOL_LABEL_MIN_SCALE', 0.5))
    max_scale = float(getattr(settings, 'VOL_LABEL_MAX_SCALE', 2.0))
    if enable_vol and base_atr_pct > 0:
        atr_pct = np.clip(rng.normal(loc=base_atr_pct, scale=base_atr_pct * 0.5, size=n_rows), a_min=base_atr_pct*0.1, a_max=base_atr_pct*5)
        scale_vec = np.clip(atr_pct / base_atr_pct, a_min=min_scale, a_max=max_scale)
        tol_vec = float(tolerance_pct) * scale_vec
        # Base positive rate ~1%; higher tol -> fewer positives
        base_pos_rate = 0.01
        p_pos = np.clip(base_pos_rate * (float(tolerance_pct) / tol_vec), 0.001, 0.2)
        y = (rng.random(n_rows) < p_pos).astype(np.int32)
        # Optionally imprint regime into first feature for realism
        X[:, 0] = (atr_pct / base_atr_pct).astype(np.float32)
    else:
        y = (rng.random(n_rows) < 0.01).astype(np.int32)
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


def build_bottom_ordinal_tabular_dataset(
    days: int = 14,
    interval: str = "1m",
    past_window: int = 15,
    future_window: int = 60,
    min_gap: int = 20,
    tolerance_pct: float = 0.004,
    strong_mult: float = 2.0,
    feature_fill_value: float = 0.0,
    *,
    scale: bool = False,
    feature_subset: Optional[List[str]] = None,
    cache: bool = False,
    cache_dir: str = "backend/app/training/cache",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Synthetic ordinal bottom dataset with three classes: 0=none, 1=weak(>=tol), 2=strong(>=strong_mult*tol).

    We simulate a latent rebound magnitude and bin labels by thresholds tied to tolerance.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = (
        f"bottom_ord|days={days}|interval={interval}|nfeat=55|scale={int(scale)}|"
        f"subset={','.join(feature_subset) if feature_subset else 'all'}|tol={tolerance_pct:.6f}|strong={strong_mult:.3f}"
    )
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"bottom_ord_{cache_hash}.npz")
    if cache and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            X = data['X']; y = data['y']; features = data['features'].tolist()
            c2 = int((y == 2).sum()); c1 = int((y == 1).sum()); c0 = int((y == 0).sum())
            log.info(f"[dataset][bottom_ord][cache_hit] path={cache_path} rows={len(X)} c2={c2} c1={c1} c0={c0}")
            return X, y, features
        except Exception:
            pass
    n_rows = 2000; n_features = 55
    rng = np.random.default_rng(321)
    X = rng.normal(0, 1, size=(n_rows, n_features)).astype(np.float32)
    # Simulate rebound magnitude r >= 0 using an exponential-like distribution with occasional large spikes
    base = rng.exponential(scale=tolerance_pct, size=n_rows)
    spikes = (rng.random(n_rows) < 0.02).astype(np.float32) * rng.exponential(scale=strong_mult * tolerance_pct, size=n_rows)
    r = base + spikes
    # Ordinal labels based on thresholds
    t1 = float(tolerance_pct)
    t2 = float(strong_mult) * t1
    y = np.zeros(n_rows, dtype=np.int32)
    y[r >= t1] = 1
    y[r >= t2] = 2
    features = [f"f{i}" for i in range(n_features)]
    if feature_subset:
        idx = [features.index(f) for f in feature_subset if f in features]
        if idx:
            X = X[:, idx]
            features = [features[i] for i in idx]
            log.info(f"[dataset][bottom_ord] feature_subset applied count={len(features)}")
        else:
            log.warning("[dataset][bottom_ord] feature_subset provided but none matched; using all features")
    if scale and StandardScaler is not None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
        log.info("[dataset][bottom_ord] StandardScaler applied")
    elif scale:
        log.warning("[dataset][bottom_ord] scale requested but sklearn not available")
    c2 = int((y == 2).sum()); c1 = int((y == 1).sum()); c0 = int((y == 0).sum())
    log.info(
        "[dataset][bottom_ord] source=synthetic interval=%s rows=%d features=%d class_counts(c2=%d,c1=%d,c0=%d) labeling(past=%d,future=%d,min_gap=%d,tol=%.4f,strong=%.2f)",
        interval, X.shape[0], X.shape[1], c2, c1, c0, past_window, future_window, min_gap, tolerance_pct, strong_mult
    )
    if cache:
        try:
            np.savez_compressed(cache_path, X=X, y=y, features=np.array(features))
            log.info(f"[dataset][bottom_ord][cache_save] path={cache_path}")
        except Exception as e:
            log.warning(f"[dataset][bottom_ord] cache save failed: {e}")
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


def train_val_test_split_time_order(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Time-ordered 3-way split: past->train, middle->val, most recent->test.

    Notes:
    - Ratios are relative to total length; they should satisfy 0 <= test_ratio < 1 and 0 < val_ratio < 1 and (val_ratio + test_ratio) < 1.
    - If test_ratio == 0, an empty test set is returned.
    - No shuffling; preserves temporal order.
    """
    assert 0 <= test_ratio < 1, "test_ratio must be in [0,1)"
    assert 0 < val_ratio < 1, "val_ratio must be in (0,1)"
    assert (val_ratio + test_ratio) < 1, "val_ratio + test_ratio must be < 1"
    n = len(X)
    if n == 0:
        raise RuntimeError("Empty dataset")
    n_test = int(round(n * test_ratio)) if test_ratio > 0 else 0
    n_remain = n - n_test
    n_val = int(round(n_remain * val_ratio))
    # Ensure at least 1 sample in train when possible
    n_train = max(1, n - n_val - n_test)
    # Compute indices
    train_end = n_train
    val_end = n_train + n_val
    X_tr = X[:train_end]
    y_tr = y[:train_end]
    X_va = X[train_end:val_end]
    y_va = y[train_end:val_end]
    X_te = X[val_end:] if n_test > 0 else X[0:0]
    y_te = y[val_end:] if n_test > 0 else y[0:0]
    return X_tr, X_va, X_te, y_tr, y_va, y_te

