import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from .sequence_dataset import build_sequence_dataset
except Exception:
    sys.path.append('.')
    from backend.app.training.sequence_dataset import build_sequence_dataset

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
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int, dropout: float, mode: str):
        super().__init__()
        self.mode = mode
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        out = self.head(last).squeeze(-1)
        # return logits for classification; raw regression value otherwise
        return out


def train(args):
    if args.mode == 'reg_next_ret':
        X, y = build_sequence_dataset(days=args.days, seq_len=args.seq_len, interval=args.interval)
        is_classification = False
    elif args.mode == 'cls_bottom':
        # Build proper sliding window bottom sequence dataset
        try:
            from .sequence_dataset import build_bottom_sequence_dataset
        except Exception:
            sys.path.append('.')
            from backend.app.training.sequence_dataset import build_bottom_sequence_dataset
        X, y = build_bottom_sequence_dataset(days=args.days, seq_len=args.seq_len, interval=args.interval,
                                             past_window=args.past_window, future_window=args.future_window,
                                             min_gap=args.min_gap, tolerance_pct=args.tolerance_pct)
        is_classification = True
    else:
        raise ValueError(f"Unsupported mode {args.mode}")
    # Time ordered split
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
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size)

    device = torch.device('cpu')
    model = LSTMModel(feature_dim=X.shape[-1], hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout, mode=args.mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
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

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            if is_classification:
                loss = loss_fn(logits, yb.float())
            else:
                loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            train_losses.append(loss.item())
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = []
            val_targets = []
            for xb, yb in dl_va:
                xb = xb.to(device)
                logits = model(xb)
                p = torch.sigmoid(logits)
                val_preds.append(p.cpu().numpy())
                val_targets.append(yb.numpy())
        if val_preds:
            vp = np.concatenate(val_preds)
            vt = np.concatenate(val_targets)
            if is_classification:
                cls = (vp >= 0.5).astype(np.int32)
                tp = int(((cls == 1) & (vt == 1)).sum())
                fp = int(((cls == 1) & (vt == 0)).sum())
                fn = int(((cls == 0) & (vt == 1)).sum())
                precision = tp / (tp + fp) if tp + fp > 0 else 0.0
                recall = tp / (tp + fn) if tp + fn > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
                log.info(f"[LSTM][cls] Epoch {epoch}/{args.epochs} train_loss={np.mean(train_losses):.6f} TP={tp} FP={fp} FN={fn} P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
            else:
                rmse = float(np.sqrt(np.mean((vp - vt)**2)))
                mae = float(np.mean(np.abs(vp - vt)))
                dir_acc = float(np.mean(np.sign(vp) == np.sign(vt)))
                log.info(f"[LSTM][reg] Epoch {epoch}/{args.epochs} train_loss={np.mean(train_losses):.6f} rmse={rmse:.6f} mae={mae:.6f} dir_acc={dir_acc:.3f}")
        else:
            log.info(f"[LSTM] Epoch {epoch}/{args.epochs} train_loss={np.mean(train_losses):.6f} (no val samples)")

    # Save
    out_dir = os.path.dirname(args.model_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'config': vars(args)}, args.model_out)
    log.info("Model saved -> %s", args.model_out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--days', type=int, default=14)
    p.add_argument('--seq-len', type=int, default=30)
    p.add_argument('--val-ratio', type=float, default=0.2)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--num-layers', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--model-out', type=str, default='backend/app/training/models/lstm_model.pt')
    p.add_argument('--mode', type=str, default='reg_next_ret', choices=['reg_next_ret','cls_bottom'])
    p.add_argument('--interval', type=str, default='1m')
    # bottom labeling params
    p.add_argument('--past-window', type=int, default=15)
    p.add_argument('--future-window', type=int, default=60)
    p.add_argument('--min-gap', type=int, default=20)
    p.add_argument('--tolerance-pct', type=float, default=0.004)
    p.add_argument('--pos-weight', type=float, default=-1.0, help='BCEWithLogitsLoss pos_weight; -1 for auto (neg/pos)')
    args = p.parse_args()
    train(args)

if __name__ == '__main__':
    main()
