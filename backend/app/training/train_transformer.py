import argparse
import json
import logging
import os
import sys
import math
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import amp
from typing import Optional
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None  # type: ignore

# Optional calibration dependencies
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
except Exception:
    LogisticRegression = None  # type: ignore
    IsotonicRegression = None  # type: ignore

try:
    from .utils import set_seed, EarlyStopping, to_device, get_device, compute_classification_report
except Exception:
    sys.path.append('.')
    from backend.app.training.utils import set_seed, EarlyStopping, to_device, get_device, compute_classification_report

try:
    from .sequence_dataset import build_sequence_dataset
except Exception:
    sys.path.append('.')
    from backend.app.training.sequence_dataset import build_sequence_dataset

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
log = logging.getLogger('train_transformer')

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array(self.y[idx]))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerModel(nn.Module):
    def __init__(self, feature_dim: int, model_dim: int, nhead: int, num_layers: int, dropout: float, mode: str):
        super().__init__()
        self.mode = mode
        self.in_proj = nn.Linear(feature_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=model_dim*4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_enc = PositionalEncoding(model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, 1)
    def forward(self, x):
        x = self.in_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.norm(x)
        pooled = x[:, -1, :]
        out = self.head(pooled).squeeze(-1)
        # return logits for classification; raw value for regression
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
            run_name = f"transformer_{args.mode}_{args.interval}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            log_dir = os.path.join(getattr(args, 'log_dir', 'runs'), run_name)
            try:
                os.makedirs(log_dir, exist_ok=True)
                writer = SummaryWriter(log_dir=log_dir)
                writer.add_text('run/config', json.dumps({k: v for k, v in vars(args).items()}, ensure_ascii=False), 0)
            except Exception as e:
                log.warning("Failed to initialize SummaryWriter: %s", e)
    if args.mode == 'reg_next_ret':
        X, y = build_sequence_dataset(days=args.days, seq_len=args.seq_len, interval=args.interval)
    else:
        try:
            from .sequence_dataset import build_bottom_sequence_dataset
        except Exception:
            sys.path.append('.')
            from backend.app.training.sequence_dataset import build_bottom_sequence_dataset
        X, y = build_bottom_sequence_dataset(days=args.days, seq_len=args.seq_len, interval=args.interval,
                                             past_window=args.past_window, future_window=args.future_window,
                                             min_gap=args.min_gap, tolerance_pct=args.tolerance_pct)
    n = len(X)
    split = max(1, int(n * (1 - args.val_ratio)))
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

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
    # Optional regime filter for validation metrics
    regime_mask = None
    if getattr(args, 'regime_filter', 'none') == 'low_vol':
        try:
            from backend.app.training.utils import volatility_mask
            regime_mask = volatility_mask(X_va, percentile=getattr(args, 'regime_percentile', 0.5), feature_index=0)
            log.info(f"[regime][TF] low_vol applied val sequences mask_true={int(regime_mask.sum())}/{len(regime_mask)}")
        except Exception as e:
            log.warning(f"[regime][TF] low_vol failed: {e}")
    device = get_device(args.device)
    pin = (device.type == 'cuda')
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    model = TransformerModel(feature_dim=X.shape[-1], model_dim=args.model_dim, nhead=args.nhead, num_layers=args.num_layers, dropout=args.dropout, mode=args.mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.scheduler.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    elif args.scheduler.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 3), gamma=0.5)
    if args.mode == 'cls_bottom':
        pos = float((y_tr == 1).sum()); neg = float((y_tr == 0).sum())
        if args.pos_weight is not None and args.pos_weight >= 0:
            pos_weight_value = args.pos_weight if args.pos_weight > 0 else (max(1.0, (neg / pos)) if pos > 0 else 1.0)
        else:
            pos_weight_value = max(1.0, (neg / pos)) if pos > 0 else 1.0
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device))
        log.info(f"[TF][cls] class_counts pos={int(pos)} neg={int(neg)} pos_weight={pos_weight_value:.4f}")
    else:
        loss_fn = nn.MSELoss()

    scaler = amp.GradScaler('cuda') if (args.amp and device.type == 'cuda') else None
    stopper = EarlyStopping(patience=args.patience, mode='max' if args.mode == 'cls_bottom' and args.early_stop_metric != 'val_loss' else 'min')
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train(); train_losses = []
        train_logits_epoch = []
        train_targets_epoch = []
        for xb, yb in dl_tr:
            xb, yb = to_device(xb, device), to_device(yb, device)
            opt.zero_grad(set_to_none=True)
            with amp.autocast(device_type='cuda', enabled=(args.amp and device.type == 'cuda')):
                logits = model(xb)
                if args.mode == 'cls_bottom':
                    loss = loss_fn(logits, yb.float())
                else:
                    loss = loss_fn(logits, yb)
            if args.mode == 'cls_bottom' and getattr(args, 'calibration', 'none') != 'none':
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
        model.eval(); val_logits=[]; val_targets=[]
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = to_device(xb, device)
                logits = model(xb)
                val_logits.append(logits.detach().cpu().numpy()); val_targets.append(yb.numpy())
        metrics = {}
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        if val_logits:
            vlogit = np.concatenate(val_logits); vt = np.concatenate(val_targets)
            if args.mode == 'cls_bottom':
                prob = 1.0 / (1.0 + np.exp(-vlogit))
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
                            log.info("[calibration][TF] skipped (insufficient class variety or empty train logits)")
                    except Exception as e:
                        log.warning("[calibration][TF] failed (%s), fallback to sigmoid", e)
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
                # precision-oriented threshold sweep
                thresholds = np.linspace(getattr(args, 't_low', 0.50), getattr(args, 't_high', 0.995), 60)
                best = {'t': 0.5, 'precision': 0.0, 'coverage': 0.0}
                n_va = len(vt)
                for t in thresholds:
                    pred = (prob >= t).astype(int)
                    pos_c = int(pred.sum())
                    if pos_c == 0:
                        continue
                    tp = int(((pred == 1) & (vt == 1)).sum())
                    precision = tp / pos_c
                    coverage = pos_c / max(1, n_va)
                    if coverage >= getattr(args, 'min_coverage', 0.005) and precision > best['precision']:
                        best = {'t': float(t), 'precision': float(precision), 'coverage': float(coverage)}
                # Double threshold report
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
                    f"[TF][cls] Epoch {epoch}/{args.epochs} train_loss={train_loss:.6f} "
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
                monitor = rmse if args.early_stop_metric in ('rmse','val_loss') else rmse
                log.info(f"[TF][reg] Epoch {epoch}/{args.epochs} train_loss={train_loss:.6f} rmse={rmse:.6f} mae={mae:.6f} dir_acc={dir_acc:.3f}")
        else:
            monitor = float('inf') if args.mode != 'cls_bottom' else 0.0
            log.info(f"[TF] Epoch {epoch}/{args.epochs} train_loss={train_loss:.6f} (no val samples)")

        if scheduler is not None:
            scheduler.step()

        history.append({'epoch': epoch, 'train_loss': train_loss, **metrics})

        # TensorBoard logging per epoch
        if writer is not None:
            try:
                writer.add_scalar('train/loss', train_loss, epoch)
                if args.mode == 'cls_bottom' and metrics:
                    for key in ['precision','recall','f1','auroc','auprc','logloss','precision_at_best_t','coverage_at_best_t','precision_high','coverage_high','precision_low','coverage_low']:
                        if key in metrics:
                            writer.add_scalar(f'val/{key}', metrics[key], epoch)
                    if 'best_threshold_precision' in metrics:
                        writer.add_scalar('val/best_threshold_precision', metrics['best_threshold_precision'], epoch)
                elif args.mode != 'cls_bottom' and metrics:
                    for key in ['rmse','mae','direction_acc']:
                        if key in metrics:
                            writer.add_scalar(f'val/{key}', metrics[key], epoch)
                try:
                    lr0 = opt.param_groups[0]['lr']
                    writer.add_scalar('optim/lr', lr0, epoch)
                except Exception:
                    pass
            except Exception as e:
                log.debug("TB write failed: %s", e)

        improved = stopper.step(monitor)
        out_dir = os.path.dirname(args.model_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        ckpt = {'state_dict': model.state_dict(), 'config': vars(args), 'epoch': epoch}
        torch.save(ckpt, os.path.join(out_dir, 'last_tf.pt') if out_dir else args.model_out)
        if improved:
            torch.save(ckpt, args.model_out)
            _save_metrics_sidecar(args.model_out, {
                'best_epoch': epoch,
                'history': history[-5:],
                'metrics': metrics,
                'calibration': getattr(args, 'calibration', 'none'),
                'timestamp_utc': datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'mode': args.mode,
                'interval': args.interval,
                'seq_len': args.seq_len,
            })
        if stopper.should_stop:
            log.info("Early stopping triggered at epoch %d", epoch)
            break

    out_dir = os.path.dirname(args.model_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'config': vars(args)}, args.model_out)
    log.info("Model saved -> %s", args.model_out)
    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--days', type=int, default=14)
    p.add_argument('--seq-len', type=int, default=30)
    p.add_argument('--val-ratio', type=float, default=0.2)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--model-dim', type=int, default=64)
    p.add_argument('--nhead', type=int, default=4)
    p.add_argument('--num-layers', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--mode', type=str, default='reg_next_ret', choices=['reg_next_ret','cls_bottom'])
    p.add_argument('--interval', type=str, default='1m')
    p.add_argument('--model-out', type=str, default='backend/app/training/models/transformer_model.pt')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--amp', action='store_true')
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
    # logging
    p.add_argument('--tb', action='store_true', help='Enable TensorBoard logging')
    p.add_argument('--log-dir', type=str, default='runs', help='TensorBoard log directory')
    args = p.parse_args()
    train(args)

if __name__ == '__main__':
    main()
