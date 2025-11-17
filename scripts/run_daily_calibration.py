#!/usr/bin/env python
"""Daily calibration automation for bottom-vs-forecast adjusted probability.

Intended to be scheduled (e.g. Windows Task Scheduler or cron).
Aggregates calibration metrics, writes timestamped CSV + JSON summary.

Usage:
  python scripts/run_daily_calibration.py \
      --eval-csv data/bottom_eval_sample.csv \
      --meta-json data/bottom_vs_forecast_meta.json \
      --out-dir data/calibration_logs

Creates:
  data/calibration_logs/
    YYYYMMDD_calibration_bins.csv
    YYYYMMDD_calibration_summary.json

"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import datetime as dt
import pandas as pd
import math


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1/(1+z)
    else:
        z = math.exp(x)
        return z/(1+z)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    fwd_cols = [c for c in df.columns if c.startswith('fwd_pred_')]
    if not fwd_cols:
        raise SystemExit('No fwd_pred_* columns present for feature derivation.')
    f_min = df[fwd_cols].min(axis=1)
    f_mean = df[fwd_cols].mean(axis=1)
    close = df['close']
    room_to_min = (f_min - close) / close.replace(0, pd.NA)
    rel_to_mean = (close - f_mean) / f_mean.replace(0, pd.NA)
    exp_ret = f_mean / close.replace(0, pd.NA) - 1.0
    return pd.DataFrame({
        'room_to_forecast_min': room_to_min.fillna(0.0),
        'rel_to_forecast_mean': rel_to_mean.fillna(0.0),
        'forecast_expected_return': exp_ret.fillna(0.0),
    })


def reliability_bins(prob: pd.Series, label: pd.Series, bins: int):
    edges = [i/bins for i in range(bins+1)]
    cat = pd.cut(prob, edges, include_lowest=True)
    rows = []
    for interval, grp in prob.groupby(cat):
        idx = grp.index
        rows.append({
            'bin': str(interval),
            'bin_lower': interval.left,
            'bin_upper': interval.right,
            'count': len(idx),
            'prob_mean': prob.loc[idx].mean(),
            'label_rate': label.loc[idx].mean(),
        })
    return pd.DataFrame(rows)


def brier(p: pd.Series, y: pd.Series) -> float:
    return float(((p - y)**2).mean())


def ece(df_bins: pd.DataFrame) -> float:
    if df_bins.empty:
        return float('nan')
    w = df_bins['count']
    total = w.sum()
    return float(((w/total) * (df_bins['prob_mean'] - df_bins['label_rate']).abs()).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-csv', required=True)
    ap.add_argument('--meta-json', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--bins', type=int, default=12)
    args = ap.parse_args()

    eval_path = Path(args.eval_csv)
    meta_path = Path(args.meta_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not eval_path.exists():
        raise SystemExit(f'eval csv missing: {eval_path}')
    if not meta_path.exists():
        raise SystemExit(f'meta json missing: {meta_path}')

    df = pd.read_csv(eval_path)
    for col in ['bottom_prob','bottom_label','close']:
        if col not in df.columns:
            raise SystemExit(f'missing column {col}')
    with meta_path.open('r', encoding='utf-8') as fh:
        meta = json.load(fh)
    coef = meta.get('coef') or meta.get('coefs') or []
    intercept = meta.get('intercept')
    if not coef or intercept is None:
        raise SystemExit('meta missing coef/intercept')

    feat_df = compute_features(df)
    X = pd.concat([
        df['bottom_prob'],
        feat_df['room_to_forecast_min'],
        feat_df['rel_to_forecast_mean'],
        feat_df['forecast_expected_return'],
    ], axis=1)
    if len(coef) != X.shape[1]:
        raise SystemExit(f'coef length {len(coef)} != feature count {X.shape[1]}')
    z = intercept + (X * coef).sum(axis=1)
    p_adj = z.apply(sigmoid)

    y = df['bottom_label'].astype(float)
    p_base = df['bottom_prob'].astype(float)
    brier_base = brier(p_base, y)
    brier_adj = brier(p_adj, y)
    bins_base = reliability_bins(p_base, y, args.bins)
    bins_adj = reliability_bins(p_adj, y, args.bins)
    merged = bins_base.merge(bins_adj, on='bin', suffixes=('_base','_adj'))
    ece_base = ece(bins_base)
    ece_adj = ece(bins_adj)

    today = dt.datetime.utcnow().strftime('%Y%m%d')
    bins_csv = out_dir / f'{today}_calibration_bins.csv'
    summary_json = out_dir / f'{today}_calibration_summary.json'
    merged.to_csv(bins_csv, index=False)
    with summary_json.open('w', encoding='utf-8') as fh:
        json.dump({
            'date': today,
            'rows': len(df),
            'brier_base': brier_base,
            'brier_adjusted': brier_adj,
            'ece_base': ece_base,
            'ece_adjusted': ece_adj,
            'bins': args.bins,
        }, fh, ensure_ascii=False, indent=2)

    print('Daily calibration written:')
    print('  bins ->', bins_csv)
    print('  summary ->', summary_json)
    print(f'Brier base={brier_base:.6f} adjusted={brier_adj:.6f} | ECE base={ece_base:.6f} adjusted={ece_adj:.6f}')

if __name__ == '__main__':
    main()
