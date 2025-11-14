import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

DEFAULT_BOTTOM_TRAIN_DAYS = int(os.getenv("BOTTOM_TRAIN_DAYS", "14"))
try:
    from ..core.config import settings
except Exception:
    if '.' not in sys.path:
        sys.path.append('.')
    try:
        from backend.app.core.config import settings  # type: ignore
    except Exception:
        settings = None  # type: ignore
if 'settings' in locals() and settings is not None:
    DEFAULT_BOTTOM_TRAIN_DAYS = getattr(settings, 'BOTTOM_TRAIN_DAYS', DEFAULT_BOTTOM_TRAIN_DAYS)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
try:
    from torch import amp  # type: ignore
except Exception:
    amp = None  # type: ignore
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None  # type: ignore

try:
    from .utils import set_seed, to_device, get_device, compute_classification_report
except Exception:
    if '.' not in sys.path:
        sys.path.append('.')
    from backend.app.training.utils import set_seed, to_device, get_device, compute_classification_report

try:
    from .sequence_dataset import build_bottom_sequence_dataset
except Exception:
    sys.path.append('.')
    from backend.app.training.sequence_dataset import build_bottom_sequence_dataset

# Optional feature set presets
try:
    from .sequence_features import SEQUENCE_FEATURES_16, FULL_FEATURE_SET_V1
except Exception:
    SEQUENCE_FEATURES_16 = None  # type: ignore
    FULL_FEATURE_SET_V1 = None  # type: ignore

try:
    from .train_lstm import LSTMModel
except Exception:
    sys.path.append('.')
    from backend.app.training.train_lstm import LSTMModel

try:
    from .train_transformer import TransformerModel
except Exception:
    sys.path.append('.')
    from backend.app.training.train_transformer import TransformerModel

try:
    import xgboost as xgb
except Exception:
    xgb = None  # type: ignore

try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None  # type: ignore

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
log = logging.getLogger('stacking_oof')


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array(self.y[idx]))


def _train_lstm(X_tr, y_tr, X_va, device, batch_size, num_workers, lr, hidden_dim, amp_flag):
    ds_tr = SeqDataset(X_tr, y_tr)
    ds_va = SeqDataset(X_va, np.zeros(len(X_va)))
    pin = (device.type == 'cuda')
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    model = LSTMModel(feature_dim=X_tr.shape[-1], hidden_dim=hidden_dim, num_layers=1, dropout=0.0, mode='cls_bottom').to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos = float((y_tr == 1).sum()); neg = float((y_tr == 0).sum())
    pos_weight_value = max(1.0, (neg / max(1.0, pos)))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device))
    try:
        scaler = amp.GradScaler('cuda') if (amp is not None and amp_flag and device.type == 'cuda') else None  # type: ignore[attr-defined]
    except Exception:
        scaler = None
    model.train()
    for xb, yb in dl_tr:
        xb, yb = to_device(xb, device), to_device(yb, device)
        opt.zero_grad(set_to_none=True)
        try:
            ctx = amp.autocast(device_type='cuda', enabled=(amp is not None and amp_flag and device.type == 'cuda'))  # type: ignore[attr-defined]
        except Exception:
            class _Noop:
                def __enter__(self): return None
                def __exit__(self, *a): return False
            ctx = _Noop()
        with ctx:
            logits = model(xb)
            loss = loss_fn(logits, yb.float())
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
    # val logits
    model.eval(); val_logits=[]
    with torch.no_grad():
        for xb, _ in dl_va:
            xb = to_device(xb, device)
            logits = model(xb)
            val_logits.append(logits.detach().cpu().numpy())
    vlogit = np.concatenate(val_logits) if val_logits else np.zeros((len(X_va),), dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-vlogit))


def _train_tf(X_tr, y_tr, X_va, device, batch_size, num_workers, lr, model_dim, amp_flag):
    ds_tr = SeqDataset(X_tr, y_tr)
    ds_va = SeqDataset(X_va, np.zeros(len(X_va)))
    pin = (device.type == 'cuda')
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    model = TransformerModel(feature_dim=X_tr.shape[-1], model_dim=model_dim, nhead=4, num_layers=1, dropout=0.1, mode='cls_bottom').to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos = float((y_tr == 1).sum()); neg = float((y_tr == 0).sum())
    pos_weight_value = max(1.0, (neg / max(1.0, pos)))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device))
    try:
        scaler = amp.GradScaler('cuda') if (amp is not None and amp_flag and device.type == 'cuda') else None  # type: ignore[attr-defined]
    except Exception:
        scaler = None
    model.train()
    for xb, yb in dl_tr:
        xb, yb = to_device(xb, device), to_device(yb, device)
        opt.zero_grad(set_to_none=True)
        try:
            ctx = amp.autocast(device_type='cuda', enabled=(amp is not None and amp_flag and device.type == 'cuda'))  # type: ignore[attr-defined]
        except Exception:
            class _Noop:
                def __enter__(self): return None
                def __exit__(self, *a): return False
            ctx = _Noop()
        with ctx:
            logits = model(xb)
            loss = loss_fn(logits, yb.float())
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
    model.eval(); val_logits=[]
    with torch.no_grad():
        for xb, _ in dl_va:
            xb = to_device(xb, device)
            logits = model(xb)
            val_logits.append(logits.detach().cpu().numpy())
    vlogit = np.concatenate(val_logits) if val_logits else np.zeros((len(X_va),), dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-vlogit))


def _train_xgb(X_tr, y_tr, X_va):
    if xgb is None:
        raise RuntimeError('xgboost not available')
    pos = float((y_tr == 1).sum()); neg = float((y_tr == 0).sum())
    scale_pos_weight = (neg / max(1.0, pos))
    clf = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )
    n = len(X_tr)
    clf.fit(X_tr.reshape(n, -1), y_tr)
    prob = clf.predict_proba(X_va.reshape(len(X_va), -1))[:, 1]
    return prob


def _time_series_folds(n: int, k: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create simple non-overlapping time-ordered folds with balanced val sizes.

    Strategy:
      - Split the timeline into (k+1) equal chunks.
      - For each i in [1..k], use chunks [0..i-1] as train, chunk i as validation.
      - Ensures each validation fold has ~n/(k+1) samples and avoids degenerate tiny folds.
    """
    if k <= 0 or n < (k + 2):
        # Fallback to single split with 80/20
        split = max(1, int(n * 0.8))
        return [(np.arange(0, split), np.arange(split, n))]
    chunk = max(1, n // (k + 1))
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, k + 1):
        train_end = chunk * i
        val_start = train_end
        val_end = min(n, val_start + chunk)
        if val_end - val_start <= 0:
            break
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        folds.append((train_idx, val_idx))
    return folds


def train(args):
    set_seed(args.seed)
    # TensorBoard (optional)
    writer = None
    if getattr(args, 'tb', False):
        if SummaryWriter is None:
            log.warning("TensorBoard not available; install 'tensorboard' or disable --tb")
        else:
            run_name = f"stacking_oof_{args.interval}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            log_dir = os.path.join(getattr(args, 'log_dir', 'runs'), run_name)
            try:
                os.makedirs(log_dir, exist_ok=True)
                writer = SummaryWriter(log_dir=log_dir)
                writer.add_text('run/config', json.dumps({k: v for k, v in vars(args).items()}, ensure_ascii=False), 0)
            except Exception as e:
                log.warning("Failed to initialize SummaryWriter: %s", e)
    # Resolve optional feature subset presets
    feature_subset = None
    preset = getattr(args, 'use_feature_set', '')
    if preset:
        if preset == '16' and SEQUENCE_FEATURES_16 is not None:
            feature_subset = list(SEQUENCE_FEATURES_16)
        elif preset == 'full_v1' and FULL_FEATURE_SET_V1 is not None:
            feature_subset = list(FULL_FEATURE_SET_V1)

    X, y = build_bottom_sequence_dataset(days=args.days, seq_len=args.seq_len, interval=args.interval,
                                         past_window=args.past_window, future_window=args.future_window,
                                         min_gap=args.min_gap, tolerance_pct=args.tolerance_pct,
                                         feature_subset=feature_subset,
                                         data_source=getattr(args, 'data_source', 'synthetic'))
    n = len(X)
    device = get_device(args.device)

    folds = _time_series_folds(n, args.folds)
    if len(folds) == 0:
        raise RuntimeError('Not enough data to create folds')

    # Collect OOF predictions for each base
    base_names = [m for m in args.models]
    oof_probs: Dict[str, np.ndarray] = {m: np.zeros(n, dtype=np.float32) for m in base_names}
    oof_mask = np.zeros(n, dtype=bool)

    for i, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        if 'lstm' in args.models:
            oof_probs['lstm'][va_idx] = _train_lstm(X_tr, y_tr, X_va, device, args.batch_size, args.num_workers, args.lstm_lr, args.lstm_hidden_dim, args.amp)
        if 'tf' in args.models:
            oof_probs['tf'][va_idx] = _train_tf(X_tr, y_tr, X_va, device, args.batch_size, args.num_workers, args.tf_lr, args.tf_model_dim, args.amp)
        if 'xgb' in args.models:
            oof_probs['xgb'][va_idx] = _train_xgb(X_tr, y_tr, X_va)
        oof_mask[va_idx] = True
        log.info("Fold %d complete: val size=%d", i, len(va_idx))

    # Restrict to OOF-covered indices
    idx = np.where(oof_mask)[0]
    y_oof = y[idx]
    X_meta = np.stack([oof_probs[m][idx] for m in base_names], axis=1)

    if LogisticRegression is None:
        raise RuntimeError('scikit-learn not available for meta model')
    ensemble = getattr(args, 'ensemble', 'logistic')
    dynamic_weights = None
    meta = None
    if ensemble == 'dynamic':
        from backend.app.training.utils import logit as _logit, sigmoid as _sigmoid
        recent_frac = float(getattr(args, 'recent_frac', 0.2))
        start = max(0, int(len(y_oof) * (1 - recent_frac)))
        y_recent = y_oof[start:]
        X_recent = X_meta[start:]
        try:
            from sklearn.metrics import average_precision_score
        except Exception:
            average_precision_score = None  # type: ignore
        weights = []
        for i in range(X_meta.shape[1]):
            p = X_recent[:, i]
            score = 0.0
            if average_precision_score is not None and len(np.unique(y_recent)) > 1:
                try:
                    score = float(average_precision_score(y_recent, p))
                except Exception:
                    score = 0.0
            if score == 0.0:
                k = max(1, int(round(len(y_recent) * 0.01)))
                order = np.argsort(-p)
                idx = order[:k]
                tp = int((y_recent[idx] == 1).sum())
                score = tp / k
            weights.append(score)
        w = np.array(weights, dtype=np.float32)
        if w.sum() <= 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w.sum()
        dynamic_weights = {m: float(wi) for m, wi in zip(base_names, w.tolist())}
        z = np.sum(w.reshape(1, -1) * _logit(X_meta), axis=1)
        meta_prob = _sigmoid(z)
    else:
        meta = LogisticRegression(class_weight='balanced', max_iter=1000)
        meta.fit(X_meta, y_oof)
        meta_prob = meta.predict_proba(X_meta)[:, 1]

    # metrics and precision-oriented threshold
    report = compute_classification_report(y_oof, meta_prob)
    from backend.app.training.utils import precision_recall_at_k as _prk
    prk_metrics = _prk(y_oof, meta_prob)
    # regime metrics on OOF indices
    metrics_lowvol = {}
    prk_lowvol = {}
    try:
        from backend.app.training.utils import volatility_mask
        mask_all = volatility_mask(X, percentile=getattr(args, 'regime_percentile', 0.5), feature_index=0) if getattr(args, 'regime_filter', 'none') == 'low_vol' else None
        if mask_all is not None:
            mask = mask_all[idx]
            if mask.any():
                report_low = compute_classification_report(y_oof[mask], meta_prob[mask])
                metrics_lowvol.update(report_low)
                prk_lowvol = _prk(y_oof[mask], meta_prob[mask])
    except Exception as e:
        log.warning(f"[regime][oof] low_vol failed: {e}")
    thresholds = np.linspace(getattr(args, 't_low', 0.50), getattr(args, 't_high', 0.995), 60)
    best = {'t': 0.5, 'precision': 0.0, 'coverage': 0.0}
    n_oof = len(y_oof)
    for t in thresholds:
        pred = (meta_prob >= t).astype(int)
        pos_c = int(pred.sum())
        if pos_c == 0:
            continue
        tp = int(((pred == 1) & (y_oof == 1)).sum())
        precision = tp / pos_c
        coverage = pos_c / max(1, n_oof)
        if coverage >= args.min_coverage and precision > best['precision']:
            best = {'t': float(t), 'precision': float(precision), 'coverage': float(coverage)}

    # double threshold quick stats
    t_low = getattr(args, 't_low', 0.50); t_high = getattr(args, 't_high', 0.995)
    pred_low = (meta_prob >= t_low).astype(int)
    pred_high = (meta_prob >= t_high).astype(int)
    high_pos = int(pred_high.sum()); low_pos = int(pred_low.sum())
    high_tp = int(((pred_high == 1) & (y_oof == 1)).sum()) if high_pos > 0 else 0
    low_tp = int(((pred_low == 1) & (y_oof == 1)).sum()) if low_pos > 0 else 0
    high_precision = high_tp / max(1, high_pos)
    low_precision = low_tp / max(1, low_pos)
    high_coverage = high_pos / max(1, n_oof)
    low_coverage = low_pos / max(1, n_oof)

    payload = {
        'timestamp_utc': datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'models': base_names,
        'metrics': report,
        'metrics_prk': prk_metrics,
        'metrics_lowvol': metrics_lowvol,
        'metrics_prk_lowvol': prk_lowvol,
        'best_threshold_precision': best['t'],
        'precision_at_best_t': best['precision'],
        'coverage_at_best_t': best['coverage'],
        'double_t_low': t_low,
        'double_t_high': t_high,
        'precision_high': high_precision,
        'precision_low': low_precision,
        'coverage_high': high_coverage,
        'coverage_low': low_coverage,
        'folds': len(folds),
        'oof_count': int(n_oof),
    }

    # --- Regime-aware dynamic weights (optional) ---
    # Compute per-model weights on recent data in low/high vol regimes using AP or top-k hit ratio
    regime_weights = {}
    try:
        from backend.app.training.utils import volatility_mask
        # Build recent window indices
        recent_frac = float(getattr(args, 'recent_frac', 0.2))
        start_idx = max(0, int(len(y) * (1 - recent_frac)))
        # We need full X mask but map to OOF idx
        mask_low_all = volatility_mask(X, percentile=getattr(args, 'regime_percentile', 0.5), feature_index=0)
        # Define high vol as inverse of low vol quantile region
        mask_high_all = ~mask_low_all
        def _weights_for(mask_all):
            # Restrict to recent and OOF indices
            select_mask = np.zeros(len(y), dtype=bool)
            select_mask[start_idx:] = True
            m = mask_all & select_mask
            m_oof = m[idx]
            yr = y_oof[m_oof]
            Xm = {mname: oof_probs[mname][idx][m_oof] for mname in base_names}
            from sklearn.metrics import average_precision_score
            ws = []
            for mname in base_names:
                p = Xm[mname]
                s = 0.0
                if len(yr) > 1 and len(np.unique(yr)) > 1:
                    try:
                        s = float(average_precision_score(yr, p))
                    except Exception:
                        s = 0.0
                if s == 0.0 and len(yr) > 10:
                    # fallback: top-1% hit ratio
                    k = max(1, int(round(len(yr) * 0.01)))
                    order = np.argsort(-p)
                    tp = int((yr[order[:k]] == 1).sum())
                    s = tp / k
                ws.append(s)
            w = np.array(ws, dtype=np.float32)
            if w.sum() <= 0:
                w = np.ones_like(w) / len(w)
            else:
                w = w / w.sum()
            return {mname: float(wi) for mname, wi in zip(base_names, w.tolist())}
        if mask_low_all is not None and mask_low_all.any():
            regime_weights['low_vol'] = _weights_for(mask_low_all)
        if mask_high_all is not None and mask_high_all.any():
            regime_weights['high_vol'] = _weights_for(mask_high_all)
    except Exception as e:
        log.warning(f"[regime][weights] failed: {e}")

    if writer is not None:
        try:
            for k, v in payload.get('metrics', {}).items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f'metrics/{k}', float(v), 0)
            for k, v in payload.get('metrics_prk', {}).items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f'prk/{k}', float(v), 0)
            writer.add_scalar('oof/count', payload['oof_count'], 0)
            writer.add_scalar('thresholds/best_t_precision', payload['best_threshold_precision'], 0)
            writer.add_scalar('thresholds/precision_high', payload['precision_high'], 0)
            writer.add_scalar('thresholds/coverage_high', payload['coverage_high'], 0)
            writer.add_scalar('thresholds/precision_low', payload['precision_low'], 0)
            writer.add_scalar('thresholds/coverage_low', payload['coverage_low'], 0)
        except Exception:
            pass

    # Save artifacts
    out_dir = os.path.dirname(args.meta_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # Save meta config (coefficients for reproducible inference)
    meta_conf = {
        'model_order': base_names,
        'ensemble': ensemble,
        'version': '1.0',
    }
    if ensemble == 'logistic' and meta is not None:
        try:
            meta_conf['coef'] = meta.coef_.ravel().tolist()
            meta_conf['intercept'] = float(meta.intercept_.ravel()[0])
            # Try to persist full sklearn model for reproducible inference
            try:
                import joblib  # type: ignore
                joblib.dump(meta, os.path.splitext(args.meta_out)[0] + '.joblib')
            except Exception:
                pass
        except Exception:
            pass
    if dynamic_weights is not None:
        meta_conf['dynamic_weights'] = dynamic_weights
    if regime_weights:
        meta_conf['regime_weights'] = regime_weights

    # --- Calibration (optional) ---
    # Supports platt (logistic) or isotonic regression on meta_prob vs y_oof
    calibration_method = getattr(args, 'calibration', 'none')
    calibration_block = {}
    if calibration_method in {'platt', 'isotonic'}:
        try:
            if calibration_method == 'platt':
                # Fit logistic regression on logit(meta_prob)
                from sklearn.linear_model import LogisticRegression as _LogReg  # type: ignore
                eps = 1e-9
                p_clip = np.clip(meta_prob, eps, 1 - eps)
                logit_vals = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
                lr_cal = _LogReg(class_weight='balanced', max_iter=1000)
                lr_cal.fit(logit_vals, y_oof)
                a = float(lr_cal.coef_.ravel()[0])
                b = float(lr_cal.intercept_.ravel()[0])
                calibration_block = {'method': 'platt', 'a': a, 'b': b}
            elif calibration_method == 'isotonic':
                from sklearn.isotonic import IsotonicRegression  # type: ignore
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(meta_prob, y_oof)
                # store mapping points for lightweight inference (x,y pairs)
                xs = iso.X_thresholds_.tolist()
                ys = iso.y_thresholds_.tolist()
                calibration_block = {'method': 'isotonic', 'points': list(map(list, zip(xs, ys)))}
        except Exception as e:
            log.warning(f"[calibration] failed: {e}")
    if calibration_block:
        meta_conf['calibration'] = calibration_block
        payload['calibration'] = calibration_block
    with open(args.meta_out, 'w') as f:
        json.dump(meta_conf, f, ensure_ascii=False, indent=2)

    # Save sidecar metrics
    sidecar = os.path.splitext(args.meta_out)[0] + '.metrics.json'
    with open(sidecar, 'w') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info('OOF Stacking complete -> %s', args.meta_out)
    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--days', type=int, default=DEFAULT_BOTTOM_TRAIN_DAYS)
    p.add_argument('--seq-len', type=int, default=16)
    p.add_argument('--interval', type=str, default='1m')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--past-window', type=int, default=15)
    p.add_argument('--future-window', type=int, default=60)
    p.add_argument('--min-gap', type=int, default=20)
    p.add_argument('--tolerance-pct', type=float, default=0.004)

    p.add_argument('--models', nargs='+', default=['lstm','tf','xgb'], choices=['lstm','tf','xgb'])
    p.add_argument('--folds', type=int, default=3)
    p.add_argument('--min-coverage', type=float, default=0.005)
    p.add_argument('--ensemble', type=str, default='logistic', choices=['logistic','dynamic'])
    p.add_argument('--recent-frac', type=float, default=0.2)
    p.add_argument('--regime-filter', type=str, default='none', choices=['none','low_vol'])
    p.add_argument('--regime-percentile', type=float, default=0.5)
    p.add_argument('--t-low', type=float, default=0.50)
    p.add_argument('--t-high', type=float, default=0.995)
    # feature selection
    p.add_argument('--use-feature-set', type=str, default='', choices=['','16','full_v1'], help='Predefined feature set alias')
    # data source
    p.add_argument('--data-source', type=str, default='synthetic', choices=['synthetic','real'], help='Use real Candle-derived features from DB or synthetic dataset')
    # logging
    p.add_argument('--tb', action='store_true', help='Enable TensorBoard logging')
    p.add_argument('--log-dir', type=str, default='runs', help='TensorBoard log directory')

    # quick base params
    p.add_argument('--lstm-lr', type=float, default=1e-3)
    p.add_argument('--lstm-hidden-dim', type=int, default=32)
    p.add_argument('--tf-lr', type=float, default=1e-3)
    p.add_argument('--tf-model-dim', type=int, default=32)

    # calibration
    p.add_argument('--calibration', type=str, default='none', choices=['none','platt','isotonic'], help='Optional probability calibration on OOF ensemble output')

    p.add_argument('--meta-out', type=str, default='backend/app/training/models/stacking_meta.json')
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
