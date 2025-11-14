import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import List

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
    from .utils import set_seed, EarlyStopping, to_device, get_device, compute_classification_report
except Exception:
    if '.' not in sys.path:
        sys.path.append('.')
    from backend.app.training.utils import set_seed, EarlyStopping, to_device, get_device, compute_classification_report

try:
    from .sequence_dataset import build_bottom_sequence_dataset, build_bottom_ordinal_sequence_dataset
except Exception:
    sys.path.append('.')
    from backend.app.training.sequence_dataset import build_bottom_sequence_dataset, build_bottom_ordinal_sequence_dataset

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
try:
    from sklearn.neural_network import MLPClassifier  # type: ignore
except Exception:
    MLPClassifier = None  # type: ignore
try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None  # type: ignore

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
log = logging.getLogger('stacking')


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array(self.y[idx]))


def _save_metrics_sidecar(model_out: str, payload: dict) -> None:
    try:
        sidecar = os.path.splitext(model_out)[0] + '.metrics.json'
        with open(sidecar, 'w') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        log.info("Metrics saved -> %s", sidecar)
    except Exception as e:
        log.warning("Failed to write metrics sidecar: %s", e)


def _train_lstm_base(X_tr, y_tr, X_va, device, args):
    ds_tr = SeqDataset(X_tr, y_tr)
    ds_va = SeqDataset(X_va, np.zeros(len(X_va)))
    pin = (device.type == 'cuda')
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    model = LSTMModel(feature_dim=X_tr.shape[-1], hidden_dim=args.lstm_hidden_dim, num_layers=1, dropout=0.0, mode='cls_bottom', num_classes=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lstm_lr)
    pos = float((y_tr == 1).sum()); neg = float((y_tr == 0).sum())
    pos_weight_value = max(1.0, (neg / max(1.0, pos)))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device))
    try:
        scaler = amp.GradScaler('cuda') if (amp is not None and args.amp and device.type == 'cuda') else None  # type: ignore[attr-defined]
    except Exception:
        scaler = None
    for _ in range(args.lstm_epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = to_device(xb, device), to_device(yb, device)
            opt.zero_grad(set_to_none=True)
            try:
                ctx = amp.autocast(device_type='cuda', enabled=(args.amp and device.type == 'cuda'))  # type: ignore[attr-defined]
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
    prob = 1.0 / (1.0 + np.exp(-vlogit))
    return prob


def _train_tf_base(X_tr, y_tr, X_va, device, args):
    ds_tr = SeqDataset(X_tr, y_tr)
    ds_va = SeqDataset(X_va, np.zeros(len(X_va)))
    pin = (device.type == 'cuda')
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    model = TransformerModel(feature_dim=X_tr.shape[-1], model_dim=args.tf_model_dim, nhead=4, num_layers=1, dropout=0.1, mode='cls_bottom').to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.tf_lr)
    pos = float((y_tr == 1).sum()); neg = float((y_tr == 0).sum())
    pos_weight_value = max(1.0, (neg / max(1.0, pos)))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device))
    try:
        scaler = amp.GradScaler('cuda') if (amp is not None and args.amp and device.type == 'cuda') else None  # type: ignore[attr-defined]
    except Exception:
        scaler = None
    for _ in range(args.tf_epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = to_device(xb, device), to_device(yb, device)
            opt.zero_grad(set_to_none=True)
            try:
                ctx = amp.autocast(device_type='cuda', enabled=(args.amp and device.type == 'cuda'))  # type: ignore[attr-defined]
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
    prob = 1.0 / (1.0 + np.exp(-vlogit))
    return prob


def _train_xgb_base(X_tr, y_tr, X_va, args):
    if xgb is None:
        raise RuntimeError("xgboost is not available")
    pos = float((y_tr == 1).sum()); neg = float((y_tr == 0).sum())
    scale_pos_weight = (neg / max(1.0, pos))
    clf = xgb.XGBClassifier(
        n_estimators=args.xgb_estimators,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        random_state=args.seed,
        scale_pos_weight=scale_pos_weight,
    )
    n = len(X_tr)
    clf.fit(X_tr.reshape(n, -1), y_tr)
    prob = clf.predict_proba(X_va.reshape(len(X_va), -1))[:, 1]
    return prob


def _bag_average(fn, bag_n: int, seed_base: int, *fn_args):
    if bag_n is None or bag_n <= 1:
        return fn(*fn_args)
    probs = []
    for i in range(bag_n):
        try:
            set_seed(seed_base + i)
        except Exception:
            pass
        p = fn(*fn_args)
        probs.append(p)
    return np.mean(np.stack(probs, axis=0), axis=0)


def _train_lstm_ordinal_base(X_tr, y_tr, X_va, device, args):
    """Train a minimal 3-class LSTM on ordinal labels and return two validation signals:
    weakplus = P(y>=1), strong = P(y==2).
    Ordinal labels are derived from the same sequences with an ordinal dataset builder.
    """
    # Build ordinal labels for the same windows: recompute on the fly for simplicity
    # We approximate by rebuilding from tabular via the sequence dataset builder
    # using the same args params.
    # Note: We only need training and validation segments aligned with provided X_tr/X_va lengths.
    # To keep alignment simple, we rebuild full ordinal sequences and slice to lengths.
    total = X_tr.shape[0] + X_va.shape[0]
    # Rebuild using identical parameters as main dataset
    Xo, yo = build_bottom_ordinal_sequence_dataset(
        days=max(1, getattr(args, 'days', 14)),
        seq_len=getattr(args, 'seq_len', 16),
        interval=getattr(args, 'interval', '1m'),
        past_window=getattr(args, 'past_window', 15),
        future_window=getattr(args, 'future_window', 60),
        min_gap=getattr(args, 'min_gap', 20),
        tolerance_pct=getattr(args, 'tolerance_pct', 0.004),
        strong_mult=getattr(args, 'strong_mult', 2.0),
        feature_subset=None,
        data_source=getattr(args, 'data_source', 'synthetic'),
    )
    Xo = Xo[-total:]
    yo = yo[-total:]
    Xo_tr, yo_tr = Xo[:X_tr.shape[0]], yo[:X_tr.shape[0]]
    Xo_va = Xo[X_tr.shape[0]:]

    ds_tr = SeqDataset(Xo_tr, yo_tr)
    ds_va = SeqDataset(Xo_va, np.zeros(len(Xo_va)))
    pin = (device.type == 'cuda')
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    model = LSTMModel(feature_dim=Xo_tr.shape[-1], hidden_dim=args.lstm_hidden_dim, num_layers=1, dropout=0.0, mode='cls_bottom_ord', num_classes=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lstm_lr)
    c0 = float((yo_tr == 0).sum()); c1 = float((yo_tr == 1).sum()); c2 = float((yo_tr == 2).sum())
    tot = max(1.0, c0 + c1 + c2)
    weights = torch.tensor([tot/max(1.0,c0), tot/max(1.0,c1), tot/max(1.0,c2)], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    try:
        scaler = amp.GradScaler('cuda') if (amp is not None and args.amp and device.type == 'cuda') else None  # type: ignore[attr-defined]
    except Exception:
        scaler = None
    for _ in range(args.lstm_epochs):
        model.train()
        for xb, yb in dl_tr:
            xb = to_device(xb, device)
            yb = to_device(yb.long(), device)
            opt.zero_grad(set_to_none=True)
            try:
                ctx = amp.autocast(device_type='cuda', enabled=(args.amp and device.type == 'cuda'))  # type: ignore[attr-defined]
            except Exception:
                class _Noop:
                    def __enter__(self): return None
                    def __exit__(self, *a): return False
                ctx = _Noop()
            with ctx:
                logits = model(xb)
                loss = loss_fn(logits, yb)
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
    if not val_logits:
        return np.zeros((len(X_va),), dtype=np.float32), np.zeros((len(X_va),), dtype=np.float32)
    logit = np.concatenate(val_logits, axis=0)
    e = np.exp(logit - logit.max(axis=1, keepdims=True))
    ps = e / e.sum(axis=1, keepdims=True)
    p_strong = ps[:, 2]
    p_weakplus = 1.0 - ps[:, 0]
    return p_weakplus.astype(np.float32), p_strong.astype(np.float32)


def train(args):
    set_seed(args.seed)
    # TensorBoard (optional)
    writer = None
    if getattr(args, 'tb', False):
        if SummaryWriter is None:
            log.warning("TensorBoard not available; install 'tensorboard' or disable --tb")
        else:
            run_name = f"stacking_{args.interval}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            log_dir = os.path.join(getattr(args, 'log_dir', 'runs'), run_name)
            try:
                os.makedirs(log_dir, exist_ok=True)
                writer = SummaryWriter(log_dir=log_dir)
                writer.add_text('run/config', json.dumps({k: v for k, v in vars(args).items()}, ensure_ascii=False), 0)
            except Exception as e:
                log.warning("Failed to initialize SummaryWriter: %s", e)
    # Build bottom classification dataset once
    # Resolve optional feature subset for consistency with base model training
    feature_subset = None
    preset = getattr(args, 'use_feature_set', '')
    raw_subset = getattr(args, 'feature_subset', '')
    if preset:
        try:
            from .sequence_features import SEQUENCE_FEATURES_16, FULL_FEATURE_SET_V1
        except Exception:
            sys.path.append('.')
            from backend.app.training.sequence_features import SEQUENCE_FEATURES_16, FULL_FEATURE_SET_V1  # type: ignore
        if preset == '16':
            feature_subset = SEQUENCE_FEATURES_16
        elif preset == 'full_v1':
            feature_subset = FULL_FEATURE_SET_V1
    elif raw_subset:
        feature_subset = [f.strip() for f in raw_subset.split(',') if f.strip()]

    # Quick refit mode: restrict days to quick_refit_days for faster calibration
    effective_days = args.days
    if getattr(args, 'quick_refit', False):
        effective_days = getattr(args, 'quick_refit_days', 30)
        log.info(f"[stacking][quick-refit] enabled days={effective_days} (original {args.days})")
    X, y = build_bottom_sequence_dataset(days=effective_days, seq_len=args.seq_len, interval=args.interval,
                                         past_window=args.past_window, future_window=args.future_window,
                                         min_gap=args.min_gap, tolerance_pct=args.tolerance_pct,
                                         feature_subset=feature_subset, data_source=getattr(args, 'data_source', 'synthetic'))
    n = len(X)
    holdout_start = max(1, int(n * (1 - args.val_ratio)))
    X_holdout = X[holdout_start:]
    y_holdout = y[holdout_start:]
    X_fulltrain = X[:holdout_start]
    y_fulltrain = y[:holdout_start]

    device = get_device(args.device)

    # Optional time-ordered OOF folds for meta training
    oof_folds = int(getattr(args, 'oof_folds', 0))
    oof_probs = None
    oof_y = None
    keys: List[str] = []
    if oof_folds and oof_folds > 1 and len(X_fulltrain) > 10:
        k = max(2, oof_folds)
        fold_edges = [int(round(len(X_fulltrain) * i / k)) for i in range(0, k + 1)]
        parts = []
        y_parts = []
        for i in range(1, len(fold_edges)):
            tr_end = fold_edges[i - 1]
            va_end = fold_edges[i]
            if tr_end <= 0 or va_end - tr_end <= 0:
                continue
            X_tr, y_tr = X_fulltrain[:tr_end], y_fulltrain[:tr_end]
            X_va, y_va = X_fulltrain[tr_end:va_end], y_fulltrain[tr_end:va_end]
            local_probs = {}
            if 'lstm' in args.models:
                local_probs['lstm'] = _bag_average(lambda a,b,c,d,e: _train_lstm_base(a,b,c,d,e), getattr(args,'bagging_n',1), args.seed+i, X_tr, y_tr, X_va, device, args)
                if getattr(args, 'use_ordinal', False):
                    p_weakplus, p_strong = _bag_average(lambda a,b,c,d,e: _train_lstm_ordinal_base(a,b,c,d,e), getattr(args,'bagging_n',1), args.seed+1000+i, X_tr, y_tr, X_va, device, args)
                    local_probs['lstm_ord_weakplus'] = p_weakplus
                    local_probs['lstm_ord_strong'] = p_strong
            if 'tf' in args.models:
                local_probs['tf'] = _bag_average(lambda a,b,c,d,e: _train_tf_base(a,b,c,d,e), getattr(args,'bagging_n',1), args.seed+2000+i, X_tr, y_tr, X_va, device, args)
            if 'xgb' in args.models:
                local_probs['xgb'] = _bag_average(lambda a,b,c,d: _train_xgb_base(a,b,c,d), getattr(args,'bagging_n',1), args.seed+3000+i, X_tr, y_tr, X_va, args)
            if not keys:
                keys = sorted(local_probs.keys())
            # stack in key order
            parts.append(np.stack([local_probs[k_] for k_ in sorted(local_probs.keys())], axis=1))
            y_parts.append(y_va)
        if parts:
            oof_probs = np.concatenate(parts, axis=0)
            oof_y = np.concatenate(y_parts, axis=0)
    # If no OOF, use single split like before for meta training
    if oof_probs is None or oof_y is None:
        split = max(1, int(n * (1 - args.val_ratio)))
        X_tr, y_tr = X[:split], y[:split]
        X_va = X[split:]
        base_probs = {}
        if 'lstm' in args.models:
            base_probs['lstm'] = _bag_average(lambda a,b,c,d,e: _train_lstm_base(a,b,c,d,e), getattr(args,'bagging_n',1), args.seed, X_tr, y_tr, X_va, device, args)
            if getattr(args, 'use_ordinal', False):
                p_weakplus, p_strong = _bag_average(lambda a,b,c,d,e: _train_lstm_ordinal_base(a,b,c,d,e), getattr(args,'bagging_n',1), args.seed+1000, X_tr, y_tr, X_va, device, args)
                base_probs['lstm_ord_weakplus'] = p_weakplus
                base_probs['lstm_ord_strong'] = p_strong
        if 'tf' in args.models:
            base_probs['tf'] = _bag_average(lambda a,b,c,d,e: _train_tf_base(a,b,c,d,e), getattr(args,'bagging_n',1), args.seed+2000, X_tr, y_tr, X_va, device, args)
        if 'xgb' in args.models:
            base_probs['xgb'] = _bag_average(lambda a,b,c,d: _train_xgb_base(a,b,c,d), getattr(args,'bagging_n',1), args.seed+3000, X_tr, y_tr, X_va, args)
        keys = sorted(base_probs.keys())
        oof_probs = np.stack([base_probs[k] for k in keys], axis=1)
        oof_y = y[split:]

    # Determine ensemble (allow env override)
    env_ens = os.getenv('STACKING_META_ENSEMBLE')
    ensemble = env_ens if env_ens in {'logistic','dynamic','mlp','lgbm','bayes'} else getattr(args, 'ensemble', 'logistic')
    if LogisticRegression is None and ensemble == 'logistic':
        raise RuntimeError("scikit-learn not available for meta model")
    dynamic_weights = None
    bayes_weights = None
    if ensemble == 'dynamic':
        # recent-window metric based weights
        from backend.app.training.utils import logit as _logit, sigmoid as _sigmoid
        recent_frac = float(getattr(args, 'recent_frac', 0.2))
        start = max(0, int(len(y_va) * (1 - recent_frac)))
        y_recent = y_va[start:]
        X_recent = X_meta[start:]
        # use average precision (AUPRC) if available else precision@top1%
        weights = []
        try:
            from sklearn.metrics import average_precision_score
        except Exception:
            average_precision_score = None  # type: ignore
        for i in range(X_meta.shape[1]):
            p = X_recent[:, i]
            score = 0.0
            if average_precision_score is not None and len(np.unique(y_recent)) > 1:
                try:
                    score = float(average_precision_score(y_recent, p))
                except Exception:
                    score = 0.0
            if score == 0.0:
                # fallback precision@1%
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
        dynamic_weights = {k: float(wi) for k, wi in zip(keys, w.tolist())}
        z = np.sum(w.reshape(1, -1) * _logit(X_meta), axis=1)
        meta_prob = _sigmoid(z)
    elif ensemble == 'mlp':
        if MLPClassifier is None:
            log.warning("MLPClassifier not available; falling back to logistic meta")
            if LogisticRegression is None:
                raise RuntimeError("Neither MLPClassifier nor LogisticRegression available for meta model")
            meta = LogisticRegression(class_weight='balanced', max_iter=1000)
        else:
            meta = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', solver='adam', max_iter=500, random_state=args.seed)
        meta.fit(oof_probs, oof_y)
        # Build holdout base predictions using full pre-holdout training
        base_holdout = {}
        if len(X_holdout) > 0:
            if 'lstm' in args.models:
                base_holdout['lstm'] = _train_lstm_base(X_fulltrain, y_fulltrain, X_holdout, device, args)
                if getattr(args, 'use_ordinal', False):
                    p_weakplus, p_strong = _train_lstm_ordinal_base(X_fulltrain, y_fulltrain, X_holdout, device, args)
                    base_holdout['lstm_ord_weakplus'] = p_weakplus
                    base_holdout['lstm_ord_strong'] = p_strong
            if 'tf' in args.models:
                base_holdout['tf'] = _train_tf_base(X_fulltrain, y_fulltrain, X_holdout, device, args)
            if 'xgb' in args.models:
                base_holdout['xgb'] = _train_xgb_base(X_fulltrain, y_fulltrain, X_holdout, args)
        X_meta = np.stack([base_holdout[k] for k in keys], axis=1) if len(X_holdout) > 0 else oof_probs
        meta_prob = meta.predict_proba(X_meta)[:, 1]
    elif ensemble == 'lgbm':
        if lgb is None:
            log.warning("lightgbm not available; falling back to logistic meta")
            if LogisticRegression is None:
                raise RuntimeError("Neither lightgbm nor LogisticRegression available for meta model")
            meta = LogisticRegression(class_weight='balanced', max_iter=1000)
            meta.fit(oof_probs, oof_y)
            base_holdout = {}
            if len(X_holdout) > 0:
                if 'lstm' in args.models:
                    base_holdout['lstm'] = _train_lstm_base(X_fulltrain, y_fulltrain, X_holdout, device, args)
                    if getattr(args, 'use_ordinal', False):
                        p_weakplus, p_strong = _train_lstm_ordinal_base(X_fulltrain, y_fulltrain, X_holdout, device, args)
                        base_holdout['lstm_ord_weakplus'] = p_weakplus
                        base_holdout['lstm_ord_strong'] = p_strong
                if 'tf' in args.models:
                    base_holdout['tf'] = _train_tf_base(X_fulltrain, y_fulltrain, X_holdout, device, args)
                if 'xgb' in args.models:
                    base_holdout['xgb'] = _train_xgb_base(X_fulltrain, y_fulltrain, X_holdout, args)
            X_meta = np.stack([base_holdout[k] for k in keys], axis=1) if len(X_holdout) > 0 else oof_probs
            meta_prob = meta.predict_proba(X_meta)[:, 1]
        else:
            meta = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=15,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=args.seed,
                class_weight='balanced'
            )
            meta.fit(oof_probs, oof_y)
            base_holdout = {}
            if len(X_holdout) > 0:
                if 'lstm' in args.models:
                    base_holdout['lstm'] = _train_lstm_base(X_fulltrain, y_fulltrain, X_holdout, device, args)
                if 'tf' in args.models:
                    base_holdout['tf'] = _train_tf_base(X_fulltrain, y_fulltrain, X_holdout, device, args)
                if 'xgb' in args.models:
                    base_holdout['xgb'] = _train_xgb_base(X_fulltrain, y_fulltrain, X_holdout, args)
            X_meta = np.stack([base_holdout[k] for k in sorted(base_holdout.keys())], axis=1) if len(X_holdout) > 0 else oof_probs
            meta_prob = meta.predict_proba(X_meta)[:, 1]
    elif ensemble == 'bayes':
        # Build holdout base predictions using full pre-holdout training (align with keys)
        base_holdout = {}
        if len(X_holdout) > 0:
            if 'lstm' in args.models:
                base_holdout['lstm'] = _bag_average(lambda a,b,c,d,e: _train_lstm_base(a,b,c,d,e), getattr(args,'bagging_n',1), args.seed, X_fulltrain, y_fulltrain, X_holdout, device, args)
                if getattr(args, 'use_ordinal', False):
                    p_weakplus, p_strong = _bag_average(lambda a,b,c,d,e: _train_lstm_ordinal_base(a,b,c,d,e), getattr(args,'bagging_n',1), args.seed+1000, X_fulltrain, y_fulltrain, X_holdout, device, args)
                    base_holdout['lstm_ord_weakplus'] = p_weakplus
                    base_holdout['lstm_ord_strong'] = p_strong
            if 'tf' in args.models:
                base_holdout['tf'] = _bag_average(lambda a,b,c,d,e: _train_tf_base(a,b,c,d,e), getattr(args,'bagging_n',1), args.seed+2000, X_fulltrain, y_fulltrain, X_holdout, device, args)
            if 'xgb' in args.models:
                base_holdout['xgb'] = _bag_average(lambda a,b,c,d: _train_xgb_base(a,b,c,d), getattr(args,'bagging_n',1), args.seed+3000, X_fulltrain, y_fulltrain, X_holdout, args)
        X_meta = np.stack([base_holdout[k] for k in keys], axis=1) if len(X_holdout) > 0 else oof_probs
        eval_y = y_holdout if len(X_holdout) > 0 else oof_y
        # Bayesian-smoothed weights from AUPRC (fallback precision@1%)
        try:
            from sklearn.metrics import average_precision_score
        except Exception:
            average_precision_score = None  # type: ignore
        scores = []
        for i in range(X_meta.shape[1]):
            pcol = X_meta[:, i]
            score = 0.0
            if average_precision_score is not None and len(np.unique(eval_y)) > 1:
                try:
                    score = float(average_precision_score(eval_y, pcol))
                except Exception:
                    score = 0.0
            if score == 0.0:
                k_top = max(1, int(round(len(eval_y) * 0.01)))
                order = np.argsort(-pcol)
                idx = order[:k_top]
                tp = int((eval_y[idx] == 1).sum())
                score = tp / k_top
            scores.append(score)
        alpha = float(getattr(args, 'bayes_alpha', 1.0))
        w = np.array([s + alpha for s in scores], dtype=np.float32)
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        try:
            bayes_weights = {k: float(wi) for k, wi in zip(keys, w.tolist())}
        except Exception:
            bayes_weights = None
        # Weighted logit blend
        eps = 1e-6
        probs_clip = np.clip(X_meta, eps, 1 - eps)
        z = np.sum(w.reshape(1, -1) * np.log(probs_clip / (1 - probs_clip)), axis=1)
        meta_prob = 1.0 / (1.0 + np.exp(-z))
    else:
        meta = LogisticRegression(class_weight='balanced', max_iter=1000)
        meta.fit(oof_probs, oof_y)
        base_holdout = {}
        if len(X_holdout) > 0:
            if 'lstm' in args.models:
                base_holdout['lstm'] = _train_lstm_base(X_fulltrain, y_fulltrain, X_holdout, device, args)
                if getattr(args, 'use_ordinal', False):
                    p_weakplus, p_strong = _train_lstm_ordinal_base(X_fulltrain, y_fulltrain, X_holdout, device, args)
                    base_holdout['lstm_ord_weakplus'] = p_weakplus
                    base_holdout['lstm_ord_strong'] = p_strong
            if 'tf' in args.models:
                base_holdout['tf'] = _train_tf_base(X_fulltrain, y_fulltrain, X_holdout, device, args)
            if 'xgb' in args.models:
                base_holdout['xgb'] = _train_xgb_base(X_fulltrain, y_fulltrain, X_holdout, args)
        X_meta = np.stack([base_holdout[k] for k in keys], axis=1) if len(X_holdout) > 0 else oof_probs
        meta_prob = meta.predict_proba(X_meta)[:, 1]

    # Metrics & threshold for precision
    # Metrics computed on holdout if available, else on OOF segment
    eval_y = y_holdout if len(X_holdout) > 0 else oof_y
    report = compute_classification_report(eval_y, meta_prob)
    from backend.app.training.utils import precision_recall_at_k as _prk
    prk_metrics = _prk(eval_y, meta_prob)
    # Regime: low volatility on validation sequences
    metrics_lowvol = {}
    prk_lowvol = {}
    try:
        from backend.app.training.utils import volatility_mask
        # apply to evaluation segment if shapes align
        mask = volatility_mask(X_holdout if len(X_holdout) > 0 else X_fulltrain, percentile=getattr(args, 'regime_percentile', 0.5), feature_index=0) if getattr(args, 'regime_filter', 'none') == 'low_vol' else None
        if mask is not None and mask.shape[0] == len(eval_y) and mask.any():
            report_low = compute_classification_report(eval_y[mask], meta_prob[mask])
            metrics_lowvol.update(report_low)
            prk_lowvol = _prk(eval_y[mask], meta_prob[mask])
    except Exception as e:
        log.warning(f"[regime][stack] low_vol failed: {e}")
    thresholds = np.linspace(getattr(args, 't_low', 0.50), getattr(args, 't_high', 0.995), 60)
    best = {'t': 0.5, 'precision': 0.0, 'coverage': 0.0}
    n_va = len(eval_y)
    for t in thresholds:
        pred = (meta_prob >= t).astype(int)
        pos_c = int(pred.sum())
        if pos_c == 0:
            continue
        tp = int(((pred == 1) & (eval_y == 1)).sum())
        precision = tp / pos_c
        coverage = pos_c / max(1, n_va)
        if coverage >= args.min_coverage and precision > best['precision']:
            best = {'t': float(t), 'precision': float(precision), 'coverage': float(coverage)}

    # double threshold quick stats
    t_low = getattr(args, 't_low', 0.50); t_high = getattr(args, 't_high', 0.995)
    pred_low = (meta_prob >= t_low).astype(int)
    pred_high = (meta_prob >= t_high).astype(int)
    high_pos = int(pred_high.sum()); low_pos = int(pred_low.sum())
    high_tp = int(((pred_high == 1) & (eval_y == 1)).sum()) if high_pos > 0 else 0
    low_tp = int(((pred_low == 1) & (eval_y == 1)).sum()) if low_pos > 0 else 0
    high_precision = high_tp / max(1, high_pos)
    low_precision = low_tp / max(1, low_pos)
    high_coverage = high_pos / max(1, n_va)
    low_coverage = low_pos / max(1, n_va)

    payload = {
        'timestamp_utc': datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'models': keys,
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
        'split': {'n_total': int(n), 'n_pre_holdout': int(len(X_fulltrain)), 'n_holdout': int(len(X_holdout))},
        'oof_folds': int(oof_folds),
        'quick_refit': bool(getattr(args, 'quick_refit', False)),
        'effective_days': int(effective_days),
        'label_schema': {
            'version': getattr(settings, 'LABEL_SCHEMA_VERSION', 'bottom_v1') if 'settings' in locals() and settings is not None else 'bottom_v1',
            'past_window': getattr(args, 'past_window', None),
            'future_window': getattr(args, 'future_window', None),
            'min_gap': getattr(args, 'min_gap', None),
            'tolerance_pct': getattr(args, 'tolerance_pct', None),
        },
    }

    # --- Probability distribution summary for drift baseline (validation meta probabilities) ---
    try:
        q_levels = [0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]
        quantiles = {f'q{int(q*100):02d}': float(np.quantile(meta_prob, q)) for q in q_levels}
        prob_dist = {
            'n': int(len(meta_prob)),
            'mean': float(np.mean(meta_prob)) if len(meta_prob) else 0.0,
            'std': float(np.std(meta_prob)) if len(meta_prob) else 0.0,
            'min': float(np.min(meta_prob)) if len(meta_prob) else 0.0,
            'max': float(np.max(meta_prob)) if len(meta_prob) else 0.0,
            'quantiles': quantiles,
        }
        # Store a downsampled deterministic sample for KS/Wasserstein baseline
        if len(meta_prob):
            step = max(1, len(meta_prob)//200)
            sample = meta_prob[::step][:200]
            prob_dist['sample'] = [float(x) for x in sample]
        payload['prob_dist'] = prob_dist
    except Exception as e:
        log.warning(f"[stacking] prob_dist failed: {e}")

    # TensorBoard summary scalars
    if writer is not None:
        try:
            for k, v in payload.get('metrics', {}).items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f'metrics/{k}', float(v), 0)
            for k, v in payload.get('metrics_prk', {}).items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f'prk/{k}', float(v), 0)
            writer.add_scalar('thresholds/best_t_precision', payload['best_threshold_precision'], 0)
            writer.add_scalar('thresholds/precision_high', payload['precision_high'], 0)
            writer.add_scalar('thresholds/coverage_high', payload['coverage_high'], 0)
            writer.add_scalar('thresholds/precision_low', payload['precision_low'], 0)
            writer.add_scalar('thresholds/coverage_low', payload['coverage_low'], 0)
        except Exception:
            pass

    # Save meta payload (JSON) and optionally persist sklearn object for reproducible inference
    out_dir = os.path.dirname(args.meta_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    meta_conf = {
        'version': '1.0',
        'model_order': keys,
        'ensemble': ensemble,
        'created_at_utc': datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'quick_refit': bool(getattr(args, 'quick_refit', False)),
        'effective_days': int(effective_days),
        'label_schema_version': getattr(settings, 'LABEL_SCHEMA_VERSION', 'bottom_v1') if 'settings' in locals() and settings is not None else 'bottom_v1',
        # rollout/versioning context
        'oof_folds': int(oof_folds),
        'bagging_n': int(getattr(args, 'bagging_n', 1)),
        'bayes_alpha': float(getattr(args, 'bayes_alpha', 1.0)),
        'use_ordinal': bool(getattr(args, 'use_ordinal', False)),
        'interval': str(args.interval),
        'seq_len': int(args.seq_len),
        'val_ratio': float(args.val_ratio),
        'data_source': str(getattr(args, 'data_source', 'synthetic')),
        'feature_set_alias': preset or '',
        'feature_subset_len': int(len(feature_subset) if feature_subset else 0),
    }
    if ensemble == 'logistic':
        try:
            meta_conf['coef'] = meta.coef_.ravel().tolist()  # type: ignore[name-defined]
            meta_conf['intercept'] = float(meta.intercept_.ravel()[0])  # type: ignore[name-defined]
            # persist sklearn model
            import joblib  # type: ignore
            joblib.dump(meta, os.path.splitext(args.meta_out)[0] + '.joblib')  # type: ignore[name-defined]
        except Exception:
            pass
    elif ensemble in ('mlp','lgbm'):
        try:
            import joblib  # type: ignore
            joblib.dump(meta, os.path.splitext(args.meta_out)[0] + f'_{ensemble}.joblib')  # type: ignore[name-defined]
        except Exception:
            pass
    if dynamic_weights is not None:
        meta_conf['dynamic_weights'] = dynamic_weights
    if bayes_weights is not None:
        meta_conf['bayes_weights'] = bayes_weights
    with open(args.meta_out, 'w') as f:
        json.dump(meta_conf, f, ensure_ascii=False, indent=2)
    _save_metrics_sidecar(args.meta_out, payload)
    log.info("Stacking complete -> %s", args.meta_out)
    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--days', type=int, default=DEFAULT_BOTTOM_TRAIN_DAYS)
    p.add_argument('--seq-len', type=int, default=16)
    p.add_argument('--val-ratio', type=float, default=0.2)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--interval', type=str, default='1m')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    # bottom labeling params
    p.add_argument('--past-window', type=int, default=15)
    p.add_argument('--future-window', type=int, default=60)
    p.add_argument('--min-gap', type=int, default=20)
    p.add_argument('--tolerance-pct', type=float, default=0.004)

    # base models and quick hyperparams
    p.add_argument('--models', nargs='+', default=['lstm', 'tf', 'xgb'], choices=['lstm','tf','xgb'])
    p.add_argument('--lstm-epochs', type=int, default=1)
    p.add_argument('--lstm-hidden-dim', type=int, default=32)
    p.add_argument('--lstm-lr', type=float, default=1e-3)
    p.add_argument('--tf-epochs', type=int, default=1)
    p.add_argument('--tf-model-dim', type=int, default=32)
    p.add_argument('--tf-lr', type=float, default=1e-3)
    p.add_argument('--xgb-estimators', type=int, default=50)

    # meta
    p.add_argument('--min-coverage', type=float, default=0.005)
    p.add_argument('--meta-out', type=str, default='backend/app/training/models/stacking_meta.json')
    p.add_argument('--ensemble', type=str, default='logistic', choices=['logistic','dynamic','mlp','lgbm','bayes'], help='Meta ensemble method')
    p.add_argument('--bayes-alpha', type=float, default=float(os.getenv('STACKING_BAYES_ALPHA', '1.0')), help='Additive smoothing for Bayes weighting')
    p.add_argument('--bagging-n', type=int, default=int(os.getenv('STACKING_BAGGING_N', '1')), help='Bagging runs per base model (>=1)')
    p.add_argument('--recent-frac', type=float, default=0.2, help='Recent fraction of validation for dynamic weighting')
    p.add_argument('--oof-folds', type=int, default=int(os.getenv('STACKING_META_OOF_FOLDS', '0')), help='Time-ordered expanding OOF folds for meta training (0/1 to disable)')
    p.add_argument('--regime-filter', type=str, default='none', choices=['none','low_vol'])
    p.add_argument('--regime-percentile', type=float, default=0.5)
    p.add_argument('--t-low', type=float, default=0.50)
    p.add_argument('--t-high', type=float, default=0.995)
    p.add_argument('--use-ordinal', action='store_true', help='Include LSTM ordinal signals (weakplus/strong) as meta features')
    # feature selection
    p.add_argument('--feature-subset', type=str, default='', help='Comma-separated explicit feature list to use')
    p.add_argument('--use-feature-set', type=str, default='', choices=['','16','full_v1'], help='Predefined feature set alias')
    # data source
    p.add_argument('--data-source', type=str, default='synthetic', choices=['synthetic','real'], help='Use real Candle-derived features from DB or synthetic dataset')
    # logging
    p.add_argument('--tb', action='store_true', help='Enable TensorBoard logging')
    p.add_argument('--log-dir', type=str, default='runs', help='TensorBoard log directory')
    p.add_argument('--quick-refit', action='store_true', help='Use a shorter recent window for rapid logistic recalibration')
    p.add_argument('--quick-refit-days', type=int, default=30, help='Days of recent data for --quick-refit')
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
