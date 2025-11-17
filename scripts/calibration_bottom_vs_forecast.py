#!/usr/bin/env python
"""Calibration & reliability evaluation for bottom vs forecast adjusted probability.

Usage:
  python scripts/calibration_bottom_vs_forecast.py data/bottom_eval_sample.csv \
      --meta-json data/bottom_vs_forecast_meta.json \
      --out-bins data/bvf_calibration_bins.csv \
      --bins 12

Input CSV must contain:
  timestamp, close, bottom_label (0/1), bottom_prob, fwd_pred_1..H (same H used in meta fitting)

Outputs:
  - Prints Brier score (base & adjusted), bin summaries, ECE estimate.
  - Writes bin CSV (if --out-bins provided) with columns:
      bin_lower, bin_upper, count, prob_mean_base, label_rate_base, prob_mean_adj, label_rate_adj

"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import List
import pandas as pd


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


def compute_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    # Forward path columns assumed fwd_pred_1..H
    fwd_cols = [c for c in df.columns if c.startswith('fwd_pred_')]
    if not fwd_cols:
        raise ValueError('No forward prediction columns (fwd_pred_*) found.')
    # Use per-row min & mean of forward predictions
    fwd_min = df[fwd_cols].min(axis=1)
    fwd_mean = df[fwd_cols].mean(axis=1)
    close = df['close']
    room_to_min = (fwd_min - close) / close.replace(0, pd.NA)
    rel_to_mean = (close - fwd_mean) / fwd_mean.replace(0, pd.NA)
    exp_ret = fwd_mean / close.replace(0, pd.NA) - 1.0
    out = pd.DataFrame({
        'room_to_forecast_min': room_to_min.fillna(0.0),
        'rel_to_forecast_mean': rel_to_mean.fillna(0.0),
        'forecast_expected_return': exp_ret.fillna(0.0),
    })
    return out


def brier_score(p: pd.Series, y: pd.Series) -> float:
    return float(((p - y) ** 2).mean())


def reliability_bins(prob: pd.Series, label: pd.Series, bins: int) -> pd.DataFrame:
    edges = [i / bins for i in range(bins + 1)]
    cat = pd.cut(prob, edges, include_lowest=True)
    rows = []
    for interval, grp in prob.groupby(cat):
        if grp.empty:
            continue
        idx = grp.index
        p_mean = prob.loc[idx].mean()
        l_rate = label.loc[idx].mean()
        rows.append({
            'bin': str(interval),
            'bin_lower': interval.left,
            'bin_upper': interval.right,
            'count': len(idx),
            'prob_mean': p_mean,
            'label_rate': l_rate,
        })
    return pd.DataFrame(rows)


def expected_calibration_error(df_bins: pd.DataFrame) -> float:
    if df_bins.empty:
        return float('nan')
    w = df_bins['count']
    total = w.sum()
    ece = ((w / total) * (df_bins['prob_mean'] - df_bins['label_rate']).abs()).sum()
    return float(ece)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', help='Evaluation CSV with bottom_prob, bottom_label, forward price predictions')
    ap.add_argument('--meta-json', required=True, help='Meta JSON with coef & intercept for adjusted probability')
    ap.add_argument('--out-bins', help='Optional output CSV path for bin summary (base & adjusted)')
    ap.add_argument('--bins', type=int, default=10, help='Number of reliability bins (default 10)')
    args = ap.parse_args()

    path_csv = Path(args.csv)
    path_meta = Path(args.meta_json)
    if not path_csv.exists():
        raise SystemExit(f'CSV not found: {path_csv}')
    if not path_meta.exists():
        raise SystemExit(f'Meta JSON not found: {path_meta}')

    df = pd.read_csv(path_csv)
    required = {'bottom_prob', 'bottom_label', 'close'}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'Missing columns: {missing}')

    with path_meta.open('r', encoding='utf-8') as fh:
        meta = json.load(fh)
    coef: List[float] = meta.get('coef') or meta.get('coefs') or []
    intercept = meta.get('intercept')
    if not coef or intercept is None:
        raise SystemExit('Meta JSON missing coef/intercept')
    # Align features
    feat_df = compute_forecast_features(df)
    # Compose feature vector columns
    # Order: bottom_prob, room_to_forecast_min, rel_to_forecast_mean, forecast_expected_return
    X = pd.concat([df['bottom_prob'], feat_df['room_to_forecast_min'], feat_df['rel_to_forecast_mean'], feat_df['forecast_expected_return']], axis=1)
    if len(coef) != X.shape[1]:
        raise SystemExit(f'Coefficient length {len(coef)} does not match feature count {X.shape[1]}')
    z = intercept + (X * coef).sum(axis=1)
    p_adj = z.apply(sigmoid)

    # Metrics
    y = df['bottom_label'].astype(float)
    p_base = df['bottom_prob'].astype(float)
    brier_base = brier_score(p_base, y)
    brier_adj = brier_score(p_adj, y)

    bins_base = reliability_bins(p_base, y, args.bins)
    bins_adj = reliability_bins(p_adj, y, args.bins)
    # Merge base/adj bins on interval string for unified view
    merged = bins_base.merge(bins_adj, on='bin', suffixes=('_base', '_adj'))
    # ECE (base & adjusted)
    ece_base = expected_calibration_error(bins_base)
    ece_adj = expected_calibration_error(bins_adj)

    print('--- Calibration Summary ---')
    print(f'Rows: {len(df)}')
    print(f'Brier Base     : {brier_base:.6f}')
    print(f'Brier Adjusted : {brier_adj:.6f}')
    print(f'ECE Base       : {ece_base:.6f}')
    print(f'ECE Adjusted   : {ece_adj:.6f}')
    print('\nReliability Bins (merged):')
    print(merged.head(len(merged)).to_string(index=False))

    if args.out_bins:
        merged.to_csv(args.out_bins, index=False)
        print(f'Wrote bins CSV: {args.out_bins}')


if __name__ == '__main__':
    main()
