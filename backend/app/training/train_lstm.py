import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import amp  # type: ignore
from typing import Optional
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None  # type: ignore

# Optional calibration imports
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
except Exception:
    LogisticRegression = None  # type: ignore
    IsotonicRegression = None  # type: ignore

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

try:
    from .utils import set_seed, EarlyStopping, to_device, get_device, compute_classification_report
except Exception:
    if '.' not in sys.path:
        sys.path.append('.')
    from backend.app.training.utils import set_seed, EarlyStopping, to_device, get_device, compute_classification_report

try:
    from .sequence_dataset import build_sequence_dataset
except Exception:
    sys.path.append('.')
    from backend.app.training.sequence_dataset import build_sequence_dataset
try:
    from .dataset import train_val_test_split_time_order
except Exception:
    sys.path.append('.')
    from backend.app.training.dataset import train_val_test_split_time_order
try:
    from .sequence_features import SEQUENCE_FEATURES_16, FULL_FEATURE_SET_V1
except Exception:
    sys.path.append('.')
    from backend.app.training.sequence_features import SEQUENCE_FEATURES_16, FULL_FEATURE_SET_V1

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
log = logging.getLogger('train_lstm')

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array(self.y[idx]))

class LSTMModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int, dropout: float, mode: str, num_classes: int = 1):
        super().__init__()
        self.mode = mode
        self.num_classes = int(num_classes)
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        out_dim = 1 if self.num_classes <= 1 else self.num_classes
        self.head = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        out = self.head(last)
        if self.num_classes <= 1:
            out = out.squeeze(-1)
        return out


def _save_metrics_sidecar(model_out: str, payload: dict) -> None:
    try:
        sidecar = os.path.splitext(model_out)[0] + '.metrics.json'
        with open(sidecar, 'w') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        log.info("Metrics saved -> %s", sidecar)
    except Exception as e:
        log.warning("Failed to write metrics sidecar: %s", e)


def train(args):
    set_seed(args.seed)
    # TensorBoard writer (optional)
    writer = None
    if getattr(args, 'tb', False):
        if SummaryWriter is None:
            log.warning("TensorBoard not available; install 'tensorboard' or disable --tb")
        else:
            run_name = f"lstm_{args.mode}_{args.interval}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            log_dir = os.path.join(getattr(args, 'log_dir', 'runs'), run_name)
            try:
                os.makedirs(log_dir, exist_ok=True)
                writer = SummaryWriter(log_dir=log_dir)
                # run metadata
                writer.add_text('run/config', json.dumps({k: v for k, v in vars(args).items()}, ensure_ascii=False), 0)
            except Exception as e:
                log.warning("Failed to initialize SummaryWriter: %s", e)
    # Resolve feature subset from args (backwards compatible: default None)
    feature_subset = None
    preset = getattr(args, 'use_feature_set', '')
    raw_subset = getattr(args, 'feature_subset', '')
    if preset:
        if preset == '16':
            feature_subset = SEQUENCE_FEATURES_16
        elif preset == 'full_v1':
            feature_subset = FULL_FEATURE_SET_V1
    elif raw_subset:
        feature_subset = [f.strip() for f in raw_subset.split(',') if f.strip()]

    # Effective days for dataset (fine-tune may use shorter window)
    effective_days = args.days
    if getattr(args, 'fine_tune', False):
        effective_days = getattr(args, 'ft_days', args.days)
        log.info(f"[LSTM][fine-tune] enabled days={effective_days} (original {args.days})")

    if args.mode == 'reg_next_ret':
        X, y = build_sequence_dataset(days=effective_days, seq_len=args.seq_len, interval=args.interval,
                                      feature_subset=feature_subset, data_source=getattr(args, 'data_source', 'synthetic'))
        is_classification = False
    elif args.mode == 'cls_bottom':
        try:
            from .sequence_dataset import build_bottom_sequence_dataset
        except Exception:
            sys.path.append('.')
            from backend.app.training.sequence_dataset import build_bottom_sequence_dataset
        X, y = build_bottom_sequence_dataset(days=effective_days, seq_len=args.seq_len, interval=args.interval,
                                             past_window=args.past_window, future_window=args.future_window,
                                             min_gap=args.min_gap, tolerance_pct=args.tolerance_pct,
                                             feature_subset=feature_subset, data_source=getattr(args, 'data_source', 'synthetic'))
        is_classification = True
    else:
        raise ValueError(f"Unsupported mode {args.mode}")
    # Time ordered split (supports optional test holdout)
    if getattr(args, 'test_ratio', 0.0) and args.test_ratio > 0:
        X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_test_split_time_order(X, y, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    else:
        n = len(X)
        split = max(1, int(n * (1 - args.val_ratio)))
        X_tr, X_va = X[:split], X[split:]
        y_tr, y_va = y[:split], y[split:]
        X_te = X[0:0]; y_te = y[0:0]

    # Dataset usage summary logging
    if args.mode == 'cls_bottom':
        pos_total = int((y == 1).sum()); neg_total = int((y == 0).sum())
        pos_tr = int((y_tr == 1).sum()); neg_tr = int((y_tr == 0).sum())
        pos_va = int((y_va == 1).sum()); neg_va = int((y_va == 0).sum())
        log.info(
            "[dataset][seq_split] mode=%s interval=%s total_seq=%d train_seq=%d val_seq=%d class_counts_total(pos=%d,neg=%d) train(pos=%d,neg=%d) val(pos=%d,neg=%d)",
            args.mode, args.interval, n, len(X_tr), len(X_va), pos_total, neg_total, pos_tr, neg_tr, pos_va, neg_va
        )
    else:
        log.info(
            "[dataset][seq_split] mode=%s interval=%s total_seq=%d train_seq=%d val_seq=%d",
            args.mode, args.interval, n, len(X_tr), len(X_va)
        )

    ds_tr = SeqDataset(X_tr, y_tr)
    ds_va = SeqDataset(X_va, y_va)
    ds_te = SeqDataset(X_te, y_te) if len(X_te) > 0 else None
    # Optional regime filter (low volatility) applied only in validation metrics, not training
    regime_mask = None
    if getattr(args, 'regime_filter', 'none') == 'low_vol':
        try:
            from backend.app.training.utils import volatility_mask
            regime_mask = volatility_mask(X_va, percentile=getattr(args, 'regime_percentile', 0.5), feature_index=0)
            log.info(f"[regime] low_vol applied val sequences mask_true={int(regime_mask.sum())}/{len(regime_mask)}")
        except Exception as e:
            log.warning(f"[regime] low_vol failed: {e}")
    device = get_device(args.device)
    pin = (device.type == 'cuda')
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin) if ds_te is not None else None
    model = LSTMModel(feature_dim=X.shape[-1], hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout, mode=args.mode).to(device)
    # Load from checkpoint for fine-tune if requested
    if getattr(args, 'fine_tune', False):
        load_candidates = []
        if getattr(args, 'load_from', ''):
            load_candidates.append(args.load_from)
        load_candidates.append(args.model_out)
        out_dir = os.path.dirname(args.model_out)
        if out_dir:
            load_candidates.append(os.path.join(out_dir, 'last_lstm.pt'))
        loaded = False
        for pth in load_candidates:
            try:
                if pth and os.path.exists(pth):
                    ck = torch.load(pth, map_location='cpu')
                    sd = ck.get('state_dict', ck)
                    missing, unexpected = model.load_state_dict(sd, strict=False)
                    log.info(f"[LSTM][fine-tune] loaded checkpoint from {pth} missing={len(missing)} unexpected={len(unexpected)}")
                    loaded = True
                    break
            except Exception as e:
                log.warning(f"[LSTM][fine-tune] failed to load {pth}: {e}")
        if not loaded:
            log.warning("[LSTM][fine-tune] no checkpoint loaded; training from scratch on recent window")
    # Optimizer with possible fine-tune LR override
    lr_use = getattr(args, 'ft_lr', None) if getattr(args, 'fine_tune', False) else None
    lr_use = float(lr_use) if lr_use is not None and lr_use > 0 else args.lr
    opt = torch.optim.Adam(model.parameters(), lr=lr_use)
    scheduler = None
    if args.scheduler.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    elif args.scheduler.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 3), gamma=0.5)
    # Compute pos_weight for class imbalance (neg/pos) if classification
    if is_classification:
        pos = float((y_tr == 1).sum())
        neg = float((y_tr == 0).sum())
        if args.pos_weight is not None and args.pos_weight >= 0:
            pos_weight_value = args.pos_weight if args.pos_weight > 0 else (max(1.0, (neg / pos)) if pos > 0 else 1.0)
        else:
            pos_weight_value = max(1.0, (neg / pos)) if pos > 0 else 1.0
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        log.info(f"[LSTM][cls] class_counts pos={int(pos)} neg={int(neg)} pos_weight={pos_weight_value:.4f}")
    else:
        loss_fn = nn.MSELoss()

    # Torch 2.x exposes autocast/GradScaler under torch.amp; fallback safe if unavailable
    try:
        scaler = amp.GradScaler('cuda') if (args.amp and device.type == 'cuda') else None  # type: ignore[attr-defined]
    except Exception:
        scaler = None
    stopper = EarlyStopping(patience=args.patience, mode='max' if is_classification and args.early_stop_metric != 'val_loss' else 'min')
    history = []

    total_epochs = getattr(args, 'ft_epochs', args.epochs) if getattr(args, 'fine_tune', False) else args.epochs
    for epoch in range(1, total_epochs + 1):
        model.train()
        train_losses = []
        # For probability calibration, collect train logits/targets (classification only)
        train_logits_epoch = []
        train_targets_epoch = []
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
                if is_classification:
                    loss = loss_fn(logits, yb.float())
                else:
                    loss = loss_fn(logits, yb)
            # collect logits for calibration (classification only)
            if is_classification and getattr(args, 'calibration', 'none') != 'none':
                try:
                    train_logits_epoch.append(logits.detach().cpu().numpy())
                    train_targets_epoch.append(yb.detach().cpu().numpy())
                except Exception:
                    pass
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
            train_losses.append(loss.item())
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = []
            val_targets = []
            for xb, yb in dl_va:
                xb = to_device(xb, device)
                logits = model(xb)
                val_logits.append(logits.detach().cpu().numpy())
                val_targets.append(yb.numpy())
        metrics = {}
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        if val_logits:
            vlogit = np.concatenate(val_logits)
            vt = np.concatenate(val_targets)
            if is_classification:
                # default sigmoid prob
                prob = 1.0 / (1.0 + np.exp(-vlogit))
                # apply optional probability calibration (fit on train epoch, evaluate on val)
                calib_used: Optional[str] = None
                if getattr(args, 'calibration', 'none') != 'none':
                    try:
                        tr_log = np.concatenate(train_logits_epoch) if train_logits_epoch else None
                        tr_y = np.concatenate(train_targets_epoch) if train_targets_epoch else None
                        if tr_log is not None and tr_y is not None and len(np.unique(tr_y)) >= 2:
                            if getattr(args, 'calibration', 'none') == 'platt' and LogisticRegression is not None:
                                lr_cal = LogisticRegression(max_iter=1000, class_weight='balanced')
                                lr_cal.fit(tr_log.reshape(-1, 1), tr_y)
                                prob = lr_cal.predict_proba(vlogit.reshape(-1, 1))[:, 1]
                                calib_used = 'platt'
                            elif getattr(args, 'calibration', 'none') == 'isotonic' and IsotonicRegression is not None:
                                iso = IsotonicRegression(out_of_bounds='clip')
                                prob = iso.fit_transform(tr_log, tr_y)
                                prob = IsotonicRegression(out_of_bounds='clip').fit(tr_log, tr_y).transform(vlogit)
                                calib_used = 'isotonic'
                        else:
                            log.info("[calibration] skipped (insufficient class variety or empty train logits)")
                    except Exception as e:
                        log.warning("[calibration] failed (%s), fallback to sigmoid", e)
                # Apply regime mask for metrics if requested
                vt_eval = vt
                prob_eval = prob
                if regime_mask is not None and regime_mask.shape[0] == len(vt):
                    vt_eval = vt[regime_mask]
                    prob_eval = prob[regime_mask]
                report = compute_classification_report(vt_eval, prob_eval)
                from backend.app.training.utils import precision_recall_at_k as _prk
                prk_metrics = _prk(vt_eval, prob_eval)
                metrics.update(report)
                metrics.update(prk_metrics)
                # precision-oriented threshold sweep under minimum coverage constraint
                thresholds = np.linspace(getattr(args, 't_low', 0.50), getattr(args, 't_high', 0.995), 60)
                best = {'t': 0.5, 'precision': 0.0, 'coverage': 0.0}
                n_va = len(vt)
                for t in thresholds:
                    pred = (prob >= t).astype(int)
                    pos = int(pred.sum())
                    if pos == 0:
                        continue
                    tp = int(((pred == 1) & (vt == 1)).sum())
                    precision = tp / pos
                    coverage = pos / max(1, n_va)
                    if coverage >= getattr(args, 'min_coverage', 0.005) and precision > best['precision']:
                        best = {'t': float(t), 'precision': float(precision), 'coverage': float(coverage)}
                # Double threshold decision stats (t_low, t_high)
                t_low = getattr(args, 't_low', 0.50)
                t_high = getattr(args, 't_high', 0.995)
                pred_low = (prob >= t_low).astype(int)
                pred_high = (prob >= t_high).astype(int)
                high_pos = int(pred_high.sum())
                low_pos = int(pred_low.sum())
                high_tp = int(((pred_high == 1) & (vt == 1)).sum()) if high_pos > 0 else 0
                low_tp = int(((pred_low == 1) & (vt == 1)).sum()) if low_pos > 0 else 0
                high_precision = high_tp / max(1, high_pos)
                low_precision = low_tp / max(1, low_pos)
                high_coverage = high_pos / max(1, n_va)
                low_coverage = low_pos / max(1, n_va)
                metrics.update({
                    'best_threshold_precision': best['t'],
                    'precision_at_best_t': best['precision'],
                    'coverage_at_best_t': best['coverage'],
                    'double_t_low': t_low,
                    'double_t_high': t_high,
                    'precision_high': high_precision,
                    'precision_low': low_precision,
                    'coverage_high': high_coverage,
                    'coverage_low': low_coverage,
                })
                monitor = report.get('f1', 0.0) if args.early_stop_metric == 'f1' else (
                          -report.get('logloss', 0.0) if args.early_stop_metric == 'neg_logloss' else report.get('f1', 0.0))
                log.info(
                    f"[LSTM][cls] Epoch {epoch}/{args.epochs} train_loss={train_loss:.6f} "
                    f"P={report['precision']:.4f} R={report['recall']:.4f} F1={report['f1']:.4f} "
                    f"AUROC={report.get('auroc','nan')} AUPRC={report.get('auprc','nan')} "
                    f"best_t(P)={metrics['best_threshold_precision']:.3f} P@t={metrics['precision_at_best_t']:.3f} cov={metrics['coverage_at_best_t']:.3f} "
                    f"double_t_high={t_high:.3f} P_high={high_precision:.3f} cov_high={high_coverage:.3f}"
                )
            else:
                pred = vlogit
                rmse = float(np.sqrt(np.mean((pred - vt) ** 2)))
                mae = float(np.mean(np.abs(pred - vt)))
                dir_acc = float(np.mean(np.sign(pred) == np.sign(vt)))
                metrics.update({'rmse': rmse, 'mae': mae, 'direction_acc': dir_acc})
                monitor = rmse if args.early_stop_metric in ('rmse', 'val_loss') else rmse
                log.info(f"[LSTM][reg] Epoch {epoch}/{args.epochs} train_loss={train_loss:.6f} rmse={rmse:.6f} mae={mae:.6f} dir_acc={dir_acc:.3f}")
        else:
            monitor = float('inf') if not is_classification else 0.0
            log.info(f"[LSTM] Epoch {epoch}/{args.epochs} train_loss={train_loss:.6f} (no val samples)")

        if scheduler is not None:
            scheduler.step()

        history.append({'epoch': epoch, 'train_loss': train_loss, **metrics})

        # TensorBoard logging per epoch
        if writer is not None:
            try:
                writer.add_scalar('train/loss', train_loss, epoch)
                if is_classification and metrics:
                    for key in ['precision','recall','f1','auroc','auprc','logloss','precision_at_best_t','coverage_at_best_t','precision_high','coverage_high','precision_low','coverage_low']:
                        if key in metrics:
                            writer.add_scalar(f'val/{key}', metrics[key], epoch)
                    if 'best_threshold_precision' in metrics:
                        writer.add_scalar('val/best_threshold_precision', metrics['best_threshold_precision'], epoch)
                elif (not is_classification) and metrics:
                    for key in ['rmse','mae','direction_acc']:
                        if key in metrics:
                            writer.add_scalar(f'val/{key}', metrics[key], epoch)
                # learning rate (first param group)
                try:
                    lr0 = opt.param_groups[0]['lr']
                    writer.add_scalar('optim/lr', lr0, epoch)
                except Exception:
                    pass
            except Exception as e:
                log.debug("TB write failed: %s", e)

        improved = stopper.step(monitor)
        # Save checkpoints
        out_dir = os.path.dirname(args.model_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        ckpt = {'state_dict': model.state_dict(), 'config': vars(args), 'epoch': epoch}
        torch.save(ckpt, os.path.join(out_dir, 'last_lstm.pt') if out_dir else args.model_out)
        if improved:
            torch.save(ckpt, args.model_out)
            payload = {
                'best_epoch': epoch,
                'history': history[-5:],  # last few epochs
                'metrics': metrics,
                'calibration': getattr(args, 'calibration', 'none'),
                'timestamp_utc': datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'mode': args.mode,
                'interval': args.interval,
                'seq_len': args.seq_len,
                'feature_list': feature_subset if feature_subset else [],
                'feature_version': preset if preset else ('explicit' if raw_subset else ''),
                'data_source': getattr(args, 'data_source', 'synthetic'),
                'fine_tune': bool(getattr(args, 'fine_tune', False)),
                'effective_days': int(effective_days),
                'label_schema': {
                    'version': getattr(settings, 'LABEL_SCHEMA_VERSION', 'bottom_v1') if 'settings' in locals() and settings is not None else 'bottom_v1',
                    'past_window': args.past_window if args.mode == 'cls_bottom' else None,
                    'future_window': args.future_window if args.mode == 'cls_bottom' else None,
                    'min_gap': args.min_gap if args.mode == 'cls_bottom' else None,
                    'tolerance_pct': args.tolerance_pct if args.mode == 'cls_bottom' else None,
                },
            }
            _save_metrics_sidecar(args.model_out, payload)
        if stopper.should_stop:
            log.info("Early stopping triggered at epoch %d", epoch)
            break

    # Ensure best model exists (if never improved, save current)
    out_dir = os.path.dirname(args.model_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(args.model_out):
        torch.save({'state_dict': model.state_dict(), 'config': vars(args)}, args.model_out)
        log.info("Model saved -> %s", args.model_out)
    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass

    # Final test evaluation (if test split present)
    test_metrics = {}
    if dl_te is not None:
        model.eval()
        with torch.no_grad():
            val_logits = []
            val_targets = []
            for xb, yb in dl_te:
                xb = to_device(xb, device)
                logits = model(xb)
                val_logits.append(logits.detach().cpu().numpy())
                val_targets.append(yb.numpy())
        if val_logits:
            vlogit = np.concatenate(val_logits)
            vt = np.concatenate(val_targets)
            if is_classification:
                prob = 1.0 / (1.0 + np.exp(-vlogit))
                from backend.app.training.utils import compute_classification_report as _report
                test_metrics = _report(vt, prob)
                log.info("[LSTM][cls][test] P=%.4f R=%.4f F1=%.4f AUROC=%s AUPRC=%s",
                         float(test_metrics.get('precision', 0.0)), float(test_metrics.get('recall', 0.0)), float(test_metrics.get('f1', 0.0)),
                         str(test_metrics.get('auroc', 'nan')), str(test_metrics.get('auprc', 'nan')))
            else:
                pred = vlogit
                rmse = float(np.sqrt(np.mean((pred - vt) ** 2)))
                mae = float(np.mean(np.abs(pred - vt)))
                dir_acc = float(np.mean(np.sign(pred) == np.sign(vt)))
                test_metrics = {'test_rmse': rmse, 'test_mae': mae, 'test_direction_acc': dir_acc}
                log.info("[LSTM][reg][test] RMSE=%.6f MAE=%.6f DirAcc=%.3f", rmse, mae, dir_acc)
    # Merge test metrics into sidecar if present
    if test_metrics:
        try:
            sidecar_path = os.path.splitext(args.model_out)[0] + '.metrics.json'
            payload = {}
            try:
                with open(sidecar_path, 'r') as f:
                    payload = json.load(f)
            except Exception:
                payload = {}
            payload = payload or {}
            payload['test_metrics'] = test_metrics
            payload['timestamp_utc_test'] = datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            with open(sidecar_path, 'w') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            log.info("Appended test metrics -> %s", sidecar_path)
        except Exception as e:
            log.warning("Failed to append test metrics: %s", e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--days', type=int, default=DEFAULT_BOTTOM_TRAIN_DAYS)
    p.add_argument('--seq-len', type=int, default=30)
    p.add_argument('--val-ratio', type=float, default=0.2)
    p.add_argument('--test-ratio', type=float, default=0.0, help='Time-ordered test holdout ratio from the most recent segment')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--num-layers', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--model-out', type=str, default='backend/app/training/models/lstm_model.pt')
    p.add_argument('--mode', type=str, default='reg_next_ret', choices=['reg_next_ret','cls_bottom'])
    p.add_argument('--interval', type=str, default='1m')
    p.add_argument('--device', type=str, default='auto', help="'cpu', 'cuda', or 'auto'")
    p.add_argument('--amp', action='store_true', help='Enable mixed precision if CUDA available')
    p.add_argument('--scheduler', type=str, default='none', choices=['none','cosine','step'])
    p.add_argument('--grad-clip', type=float, default=0.0)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--early-stop-metric', type=str, default='f1', choices=['f1','neg_logloss','rmse','val_loss'])
    p.add_argument('--seed', type=int, default=42)
    # bottom labeling params
    p.add_argument('--past-window', type=int, default=15)
    p.add_argument('--future-window', type=int, default=60)
    p.add_argument('--min-gap', type=int, default=20)
    p.add_argument('--tolerance-pct', type=float, default=0.004)
    p.add_argument('--pos-weight', type=float, default=-1.0, help='BCEWithLogitsLoss pos_weight; -1 for auto (neg/pos)')
    p.add_argument('--calibration', type=str, default='none', choices=['none','platt','isotonic'], help='Probability calibration method for classification')
    p.add_argument('--min-coverage', type=float, default=0.005, help='Min positive rate constraint for precision-oriented threshold sweep')
    p.add_argument('--regime-filter', type=str, default='none', choices=['none','low_vol'])
    p.add_argument('--regime-percentile', type=float, default=0.5)
    p.add_argument('--t-low', type=float, default=0.50)
    p.add_argument('--t-high', type=float, default=0.995)
    # feature selection
    p.add_argument('--feature-subset', type=str, default='', help='Comma-separated explicit feature list to use')
    p.add_argument('--use-feature-set', type=str, default='', choices=['','16','full_v1'], help='Predefined feature set alias')
    # data source
    p.add_argument('--data-source', type=str, default='synthetic', choices=['synthetic','real'], help='Use real Candle-derived features from DB or synthetic dataset')
    # logging
    p.add_argument('--tb', action='store_true', help='Enable TensorBoard logging')
    p.add_argument('--log-dir', type=str, default='runs', help='TensorBoard log directory')
    # fine-tune options
    p.add_argument('--fine-tune', action='store_true', help='Load existing checkpoint and train few epochs on recent window')
    p.add_argument('--ft-days', type=int, default=int(os.getenv('SEQ_FT_DAYS', '14')))
    p.add_argument('--ft-epochs', type=int, default=int(os.getenv('SEQ_FT_EPOCHS', '2')))
    p.add_argument('--ft-lr', type=float, default=float(os.getenv('SEQ_FT_LR', '5e-4')))
    p.add_argument('--load-from', type=str, default=os.getenv('SEQ_FT_LOAD_FROM', ''), help='Optional checkpoint path; defaults to model-out or last_lstm.pt')
    args = p.parse_args()
    train(args)

if __name__ == '__main__':
    main()
