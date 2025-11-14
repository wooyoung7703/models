"""Analyze stacking signal persistence after each entry over a lookahead window.

Scans last N days, simulates entries (without adds) and records how many of the next K candles
keep prob above soft thresholds: threshold, 0.98*threshold, 0.95*threshold.

Run:
  python backend/scripts/analyze_signal_persistence.py --days 30 --lookahead 30
"""
from __future__ import annotations
from pathlib import Path
import sys
import argparse
from datetime import timedelta
from typing import List, Dict, Any
from sqlmodel import Session, select

_THIS = Path(__file__).resolve(); _ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.app.core.config import settings
from backend.app.db import engine as main_engine
from backend.app.models import Candle
from backend.app.predictor import RealTimePredictor


def analyze(days: int, lookahead: int) -> Dict[str, Any]:
    symbol = settings.SYMBOL; ex = settings.EXCHANGE_TYPE; itv = settings.INTERVAL
    with Session(main_engine) as s:
        last = s.exec(select(Candle).where((Candle.symbol==symbol)&(Candle.exchange_type==ex)&(Candle.interval==itv)).order_by(Candle.open_time.desc()).limit(1)).first()  # type: ignore[attr-defined]
        if not last:
            raise RuntimeError('No candles found')
        end_dt = last.open_time; start_dt = end_dt - timedelta(days=days)
        rows: List[Candle] = s.exec(select(Candle).where((Candle.symbol==symbol)&(Candle.exchange_type==ex)&(Candle.interval==itv)&(Candle.open_time>=start_dt)&(Candle.open_time<=end_dt)).order_by(Candle.open_time.asc())).all()  # type: ignore[attr-defined]
    predictor = RealTimePredictor(symbol=symbol, interval=itv)
    entries = []
    for i, row in enumerate(rows):
        nc = predictor.predict_from_row(row, price_override=row.close, price_source='closed').to_dict()
        stk = (nc.get('stacking') or {})
        if not stk.get('ready'): continue
        if stk.get('decision'):
            entries.append((i, stk))
    # Evaluate persistence
    stats = []
    for idx, sblk in entries:
        th = sblk.get('threshold'); prob0 = float(sblk.get('prob') or 0.0)
        if th is None: continue
        w_full = w_98 = w_95 = 0
        horizon = min(len(rows)-1, idx + lookahead)
        for j in range(idx+1, horizon+1):
            nc2 = predictor.predict_from_row(rows[j], price_override=rows[j].close, price_source='closed').to_dict()
            s2 = (nc2.get('stacking') or {})
            p2 = float(s2.get('prob') or 0.0)
            if p2 >= th: w_full += 1
            if p2 >= th*0.98: w_98 += 1
            if p2 >= th*0.95: w_95 += 1
        stats.append({'entry_index': idx, 'prob0': prob0, 'above_t': w_full, 'above_0p98t': w_98, 'above_0p95t': w_95})
    # Aggregate
    def _avg(k):
        return (sum(d[k] for d in stats) / len(stats)) if stats else 0.0
    summary = {
        'entries': len(stats),
        'lookahead': lookahead,
        'avg_above_t': _avg('above_t'),
        'avg_above_0p98t': _avg('above_0p98t'),
        'avg_above_0p95t': _avg('above_0p95t'),
        'stats': stats[:20],  # sample first 20
    }
    print('Signal persistence summary')
    print(summary)
    return summary

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=30)
    ap.add_argument('--lookahead', type=int, default=30)
    args = ap.parse_args()
    analyze(args.days, args.lookahead)
