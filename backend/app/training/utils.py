import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, precision_recall_fscore_support
except Exception:  # optional dependency
    roc_auc_score = None
    average_precision_score = None
    log_loss = None
    precision_recall_fscore_support = None


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(pref: str = 'auto') -> torch.device:
    if pref == 'cpu':
        return torch.device('cpu')
    if pref == 'cuda':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # auto
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_device(x, device: torch.device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x


@dataclass
class EarlyStopping:
    patience: int = 10
    mode: str = 'min'  # 'min' or 'max'

    def __post_init__(self):
        self.best: Optional[float] = None
        self.count: int = 0
        self.should_stop: bool = False

    def step(self, value: float) -> bool:
        """Return True if improved, update internal state."""
        if self.best is None:
            self.best = value
            self.count = 0
            return True
        improved = (value < self.best) if self.mode == 'min' else (value > self.best)
        if improved:
            self.best = value
            self.count = 0
            return True
        else:
            self.count += 1
            if self.count >= self.patience:
                self.should_stop = True
            return False


def compute_classification_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = y_true.astype(np.int32).ravel()
    y_prob = y_prob.astype(np.float32).ravel()
    y_pred = (y_prob >= threshold).astype(np.int32)

    # Precision/Recall/F1
    if precision_recall_fscore_support is not None:
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    else:
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0

    report = {
        'precision': float(p),
        'recall': float(r),
        'f1': float(f1),
    }
    # AUROC/AUPRC/logloss are optional
    if roc_auc_score is not None:
        try:
            report['auroc'] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            report['auroc'] = None
    if average_precision_score is not None:
        try:
            report['auprc'] = float(average_precision_score(y_true, y_prob))
        except Exception:
            report['auprc'] = None
    if log_loss is not None:
        try:
            report['logloss'] = float(log_loss(y_true, y_prob, labels=[0,1]))
        except Exception:
            # fallback negative log loss unavailable
            pass
    return report


def precision_recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, ks: Optional[List[float]] = None) -> Dict[str, float]:
    """Compute Precision@K and Recall@K for given percentages (0-1).
    Ensures at least 1 sample is selected for each K.
    Keys: precision_at_{pct} , recall_at_{pct} where pct is like 0p1 for 0.1% or 1p0 for 1%.
    """
    y_true = y_true.astype(np.int32).ravel()
    y_prob = y_prob.astype(np.float32).ravel()
    n = len(y_true)
    if n == 0:
        return {}
    if ks is None:
        ks = [0.001, 0.005, 0.01]  # 0.1%, 0.5%, 1%
    order = np.argsort(-y_prob)  # descending by prob
    metrics: Dict[str, float] = {}
    total_pos = max(1, int((y_true == 1).sum()))
    for k in ks:
        k = float(k)
        topk = max(1, int(round(n * k)))
        idx = order[:topk]
        tp = int((y_true[idx] == 1).sum())
        precision = tp / topk if topk > 0 else 0.0
        recall = tp / total_pos if total_pos > 0 else 0.0
        tag = f"{k*100:.1f}".replace('.', 'p')  # 0.1 -> '0p1'
        metrics[f'precision_at_{tag}'] = float(precision)
        metrics[f'recall_at_{tag}'] = float(recall)
    return metrics


def volatility_mask(X: np.ndarray, percentile: float = 0.5, feature_index: int = 0) -> np.ndarray:
    """Return boolean mask for low-volatility regime based on per-sequence std of a feature.
    X: (N, T, D). Compute std over time for feature_index; choose sequences below given percentile.
    """
    if X.ndim != 3 or X.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    feat = X[:, :, min(feature_index, X.shape[-1]-1)]
    stds = np.std(feat, axis=1)
    thresh = np.percentile(stds, percentile * 100.0)
    return stds <= thresh


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))
