import logging
from typing import Tuple, List
import numpy as np

try:
    from ..core.config import settings  # type: ignore
except Exception:
    settings = None  # type: ignore

from .dataset import build_tabular_dataset, build_bottom_tabular_dataset, build_bottom_ordinal_tabular_dataset

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
    data_source: str = "synthetic",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (N, T, F) sequences and aligned targets from tabular X,y.

    This implementation intentionally matches dataset.build_tabular_dataset signature (no symbol/exchange args)
    to avoid API mismatch and synthetic fallback during tests.
    """
    if data_source == "real":
        try:
            from .dataset_real import build_tabular_dataset_real
            X_tab, y_tab, feat_names = build_tabular_dataset_real(days=days, interval=interval, target="next_ret", feature_subset=feature_subset, scale=scale)
        except Exception as e:
            log.warning(f"[dataset][seq] real data build failed ({e}); falling back to synthetic")
            X_tab, y_tab, feat_names = build_tabular_dataset(days=days, interval=interval, feature_fill_value=feature_fill_value, scale=scale, feature_subset=feature_subset, cache=cache)
    else:
        X_tab, y_tab, feat_names = build_tabular_dataset(days=days, interval=interval, feature_fill_value=feature_fill_value, scale=scale, feature_subset=feature_subset, cache=cache)
    # Optional AE augmentation on tabular vectors before building sequences
    if settings is not None and getattr(settings, 'AE_AUGMENT', False):
        try:
            from .ae import load_ae, transform_latent
            pkg = load_ae(getattr(settings, 'AE_MODEL_PATH', ''))
            if pkg is not None and pkg.model.input_dim == X_tab.shape[1]:
                Z = transform_latent(pkg, X_tab)
                X_tab = np.concatenate([X_tab, Z], axis=1).astype(np.float32)
                log.info(f"[dataset][seq_reg][ae] augmented features: {X_tab.shape[1]} (latent {Z.shape[1]})")
            else:
                log.warning("[dataset][seq_reg][ae] model missing or input_dim mismatch; skipping AE augment")
        except Exception as e:
            log.warning(f"[dataset][seq_reg][ae] augment failed: {e}")
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
    data_source: str = "synthetic",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build classification sequences for bottom detection from tabular bottom dataset.

    Matches dataset.build_bottom_tabular_dataset signature (no symbol/exchange args).
    """
    if data_source == "real":
        try:
            from .dataset_real import build_bottom_tabular_dataset_real
            X_tab, y_tab, feat_names = build_bottom_tabular_dataset_real(
                days=days,
                interval=interval,
                past_window=past_window,
                future_window=future_window,
                tolerance_pct=tolerance_pct,
                feature_subset=feature_subset,
                scale=scale,
            )
        except Exception as e:
            log.warning(f"[dataset][seq_bottom] real data build failed ({e}); falling back to synthetic")
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
    else:
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
    # Optional AE augmentation on tabular vectors
    if settings is not None and getattr(settings, 'AE_AUGMENT', False):
        try:
            from .ae import load_ae, transform_latent
            pkg = load_ae(getattr(settings, 'AE_MODEL_PATH', ''))
            if pkg is not None and pkg.model.input_dim == X_tab.shape[1]:
                Z = transform_latent(pkg, X_tab)
                X_tab = np.concatenate([X_tab, Z], axis=1).astype(np.float32)
                log.info(f"[dataset][seq_bottom][ae] augmented features: {X_tab.shape[1]} (latent {Z.shape[1]})")
            else:
                log.warning("[dataset][seq_bottom][ae] model missing or input_dim mismatch; skipping AE augment")
        except Exception as e:
            log.warning(f"[dataset][seq_bottom][ae] augment failed: {e}")
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


def build_bottom_ordinal_sequence_dataset(
    days: int = 14,
    seq_len: int = 30,
    interval: str = "1m",
    past_window: int = 15,
    future_window: int = 60,
    min_gap: int = 20,
    tolerance_pct: float = 0.004,
    strong_mult: float = 2.0,
    feature_fill_value: float = 0.0,
    *,
    scale: bool = False,
    feature_subset: list | None = None,
    cache: bool = False,
    data_source: str = "synthetic",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ordinal classification sequences (0/1/2) for bottom strength.

    0=none, 1=weak(>=tol), 2=strong(>=strong_mult*tol)
    """
    if data_source == "real":
        try:
            from .dataset_real import build_bottom_ordinal_tabular_dataset_real
            X_tab, y_tab, feat_names = build_bottom_ordinal_tabular_dataset_real(
                days=days,
                interval=interval,
                past_window=past_window,
                future_window=future_window,
                tolerance_pct=tolerance_pct,
                strong_mult=strong_mult,
                feature_subset=feature_subset,
                scale=scale,
            )
        except Exception as e:
            log.warning(f"[dataset][seq_bottom_ord] real data build failed ({e}); falling back to synthetic")
            X_tab, y_tab, feat_names = build_bottom_ordinal_tabular_dataset(
                days=days,
                interval=interval,
                past_window=past_window,
                future_window=future_window,
                min_gap=min_gap,
                tolerance_pct=tolerance_pct,
                strong_mult=strong_mult,
                feature_fill_value=feature_fill_value,
                scale=scale,
                feature_subset=feature_subset,
                cache=cache,
            )
    else:
        X_tab, y_tab, feat_names = build_bottom_ordinal_tabular_dataset(
            days=days,
            interval=interval,
            past_window=past_window,
            future_window=future_window,
            min_gap=min_gap,
            tolerance_pct=tolerance_pct,
            strong_mult=strong_mult,
            feature_fill_value=feature_fill_value,
            scale=scale,
            feature_subset=feature_subset,
            cache=cache,
        )
    # Optional AE augmentation on tabular vectors
    if settings is not None and getattr(settings, 'AE_AUGMENT', False):
        try:
            from .ae import load_ae, transform_latent
            pkg = load_ae(getattr(settings, 'AE_MODEL_PATH', ''))
            if pkg is not None and pkg.model.input_dim == X_tab.shape[1]:
                Z = transform_latent(pkg, X_tab)
                X_tab = np.concatenate([X_tab, Z], axis=1).astype(np.float32)
                log.info(f"[dataset][seq_bottom_ord][ae] augmented features: {X_tab.shape[1]} (latent {Z.shape[1]})")
            else:
                log.warning("[dataset][seq_bottom_ord][ae] model missing or input_dim mismatch; skipping AE augment")
        except Exception as e:
            log.warning(f"[dataset][seq_bottom_ord][ae] augment failed: {e}")
    n, f = X_tab.shape
    if n < seq_len + 10:
        raise RuntimeError(f"Insufficient rows for bottom ordinal sequence dataset: n={n}, seq_len={seq_len}")
    X_seq: List[np.ndarray] = []
    y_seq: List[int] = []
    for i in range(seq_len - 1, n):
        X_seq.append(X_tab[i - seq_len + 1 : i + 1])
        y_seq.append(int(y_tab[i]))
    X = np.stack(X_seq).astype(np.float32)
    y = np.array(y_seq, dtype=np.int64)
    c2 = int((y == 2).sum()); c1 = int((y == 1).sum()); c0 = int((y == 0).sum())
    log.info(
        "[dataset][seq_bottom_ord] interval=%s days=%d rows=%d seq_len=%d sequences=%d features=%d class_counts(c2=%d,c1=%d,c0=%d)",
        interval, days, n, seq_len, X.shape[0], f, c2, c1, c0
    )
    return X, y
