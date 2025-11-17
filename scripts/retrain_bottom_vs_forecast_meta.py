#!/usr/bin/env python
"""Retrain bottom-vs-forecast logistic meta coefficients.

Reads an evaluation CSV containing:
  timestamp, close, bottom_label (0/1), bottom_prob, fwd_pred_1..H
Generates forecast-derived features (room_to_forecast_min, rel_to_forecast_mean, forecast_expected_return)
Fits a logistic regression over feature vector [bottom_prob, room, rel, exp_ret]
Outputs updated meta JSON if improvement criteria met.

Usage:
  python scripts/retrain_bottom_vs_forecast_meta.py \
      --csv data/bottom_eval_sample.csv \
      --current-meta data/bottom_vs_forecast_meta.json \
      --out-meta data/bottom_vs_forecast_meta.json \
      --min-rel-brier-improve 0.01 \
      --max-iters 5000

Notes:
- Does NOT require scikit-learn (pure numpy gradient descent with L2 regularization).
- If --current-meta omitted, always writes new meta.
- Improvement is relative Brier score reduction ( (old - new) / old ).
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    fwd_cols = [c for c in df.columns if c.startswith('fwd_pred_')]
    if not fwd_cols:
        raise SystemExit('No fwd_pred_* columns present.')
    f_min = df[fwd_cols].min(axis=1)
    f_mean = df[fwd_cols].mean(axis=1)
    close = df['close'].astype(float)
    room_to_min = (f_min - close) / close.replace(0, pd.NA)
    rel_to_mean = (close - f_mean) / f_mean.replace(0, pd.NA)
    exp_ret = f_mean / close.replace(0, pd.NA) - 1.0
    return pd.DataFrame({
        'room_to_forecast_min': room_to_min.fillna(0.0),
        'rel_to_forecast_mean': rel_to_mean.fillna(0.0),
        'forecast_expected_return': exp_ret.fillna(0.0),
    })


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def fit_logistic(X: np.ndarray, y: np.ndarray, max_iters: int = 3000, lr: float = 0.05, l2: float = 0.01) -> tuple[np.ndarray, float]:
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for i in range(max_iters):
        z = X @ w + b
        p = sigmoid(z)
        # gradient
        err = p - y
        grad_w = (X.T @ err) / n + l2 * w
        grad_b = float(np.sum(err) / n)
        w -= lr * grad_w
        b -= lr * grad_b
        if i % 500 == 0 and i > 0:
            # simple early stop based on small gradient norm
            if np.linalg.norm(grad_w) < 1e-5 and abs(grad_b) < 1e-5:
                break
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='Evaluation CSV with bottom_prob, bottom_label, forward preds')
    ap.add_argument('--current-meta', help='Existing meta JSON to compare performance')
    ap.add_argument('--out-meta', required=True, help='Output path for meta JSON (overwrite if improved)')
    ap.add_argument('--min-rel-brier-improve', type=float, default=0.0, help='Minimum relative Brier improvement required to overwrite (0=always)')
    ap.add_argument('--max-iters', type=int, default=3000, help='Max iterations for GD')
    ap.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    ap.add_argument('--l2', type=float, default=0.01, help='L2 regularization strength')
    args = ap.parse_args()

    path_csv = Path(args.csv)
    if not path_csv.exists():
        raise SystemExit(f'CSV not found: {path_csv}')
    df = pd.read_csv(path_csv)
    needed = {'bottom_prob', 'bottom_label', 'close'}
    miss = needed - set(df.columns)
    if miss:
        raise SystemExit(f'Missing required columns: {miss}')

    feat_df = compute_features(df)
    X_df = pd.concat([
        df['bottom_prob'].astype(float),
        feat_df['room_to_forecast_min'],
        feat_df['rel_to_forecast_mean'],
        feat_df['forecast_expected_return'],
    ], axis=1)
    X = X_df.to_numpy()
    y = df['bottom_label'].astype(float).to_numpy()

    w, b = fit_logistic(X, y, max_iters=args.max_iters, lr=args.lr, l2=args.l2)
    p_new = sigmoid(X @ w + b)
    brier_new = brier(p_new, y)

    # If existing meta provided, evaluate its Brier
    overwrite = True
    prev_metrics = None
    if args.current_meta:
        cm_path = Path(args.current_meta)
        if cm_path.exists():
            try:
                meta_old = json.loads(cm_path.read_text(encoding='utf-8'))
                coefs_old = meta_old.get('coef') or meta_old.get('coefs') or []
                intercept_old = meta_old.get('intercept')
                if len(coefs_old) == X.shape[1] and intercept_old is not None:
                    p_old = sigmoid(X @ np.array(coefs_old) + float(intercept_old))
                    brier_old = brier(p_old, y)
                    rel_improve = (brier_old - brier_new) / brier_old if brier_old > 0 else 0.0
                    prev_metrics = {'brier_old': brier_old, 'rel_improve': rel_improve}
                    overwrite = rel_improve >= args.min_rel_brier_improve
                else:
                    prev_metrics = {'error': 'old meta shape mismatch'}
            except Exception as e:
                prev_metrics = {'error': f'failed to load old meta: {e}'}

    out_meta = {
        'coef': [float(v) for v in w.tolist()],
        'intercept': float(b),
        'feature_order': ['bottom_prob','room_to_forecast_min','rel_to_forecast_mean','forecast_expected_return'],
        'brier_new': brier_new,
        'retrain_samples': int(len(df)),
        'prev': prev_metrics,
    }

    if overwrite:
        Path(args.out_meta).write_text(json.dumps(out_meta, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'[meta] wrote new meta -> {args.out_meta}')
    else:
        print('[meta] NOT overwritten (relative improvement insufficient)')
    print(f'Brier new={brier_new:.6f}')
    if prev_metrics:
        print('Previous meta:', prev_metrics)

if __name__ == '__main__':
    main()
