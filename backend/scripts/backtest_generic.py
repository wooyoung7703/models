"""Generic backtest script allowing --days parameter.

Usage:
  python backend/scripts/backtest_generic.py --days 90

Reuses logic from backtest_last_month but exposes CLI arg.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from typing import Any, Dict
from datetime import datetime, timedelta

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sqlmodel import Session, select, SQLModel, create_engine
from backend.app.core.config import settings
from backend.app.db import engine as main_engine
from backend.app.models import Candle, Trade, TradeFill
from backend.app.predictor import RealTimePredictor
from backend.app.trade_manager import TradeManager


def run_backtest(days: int) -> Dict[str, Any]:
    symbol = settings.SYMBOL
    ex = settings.EXCHANGE_TYPE
    itv = settings.INTERVAL
    bt_path = _ROOT / 'backend' / 'data' / 'backtest_tmp_generic.db'
    bt_url = f"sqlite:///{bt_path}"
    bt_engine = create_engine(bt_url, echo=False)
    SQLModel.metadata.create_all(bt_engine)

    # Query candles range
    with Session(main_engine) as s:
        last = s.exec(select(Candle).where((Candle.symbol==symbol)&(Candle.exchange_type==ex)&(Candle.interval==itv)).order_by(Candle.open_time.desc()).limit(1)).first()  # type: ignore[attr-defined]
        if not last:
            raise RuntimeError("No candles found for symbol")
        end_dt = last.open_time
        start_dt = end_dt - timedelta(days=days)
        rows = s.exec(select(Candle).where((Candle.symbol==symbol)&(Candle.exchange_type==ex)&(Candle.interval==itv)&(Candle.open_time>=start_dt)&(Candle.open_time<=end_dt)).order_by(Candle.open_time.asc())).all()  # type: ignore[attr-defined]

    predictor = RealTimePredictor(symbol=symbol, interval=itv)
    tm = TradeManager(lambda: Session(bt_engine), leverage=10, add_cooldown_seconds=0)
    last_fill_time: Dict[int, datetime] = {}
    cooldown = settings.ADD_COOLDOWN_SECONDS

    for row in rows:
        nc = predictor.predict_from_row(row, price_override=row.close, price_source='closed')
        nc_d = nc.to_dict()
        # Cooldown gating for adds
        with Session(bt_engine) as ts:
            open_trade = ts.exec(select(Trade).where((Trade.symbol==symbol)&(Trade.status=='open')).order_by(Trade.created_at.desc())).first()  # type: ignore[attr-defined]
        modified_nc = nc_d
        if open_trade is not None:
            tid = int(open_trade.id)  # type: ignore[arg-type]
            last_ts = last_fill_time.get(tid)
            can_add_now = True
            if cooldown > 0 and last_ts is not None:
                elapsed = (row.close_time - last_ts).total_seconds()
                if elapsed < cooldown:
                    can_add_now = False
            if not can_add_now:
                sblk = dict(modified_nc.get('stacking') or {})
                sblk['decision'] = False
                modified_nc = {**modified_nc, 'stacking': sblk}
        action = tm.process(symbol=symbol, interval=itv, exchange_type=ex, price=row.close, nowcast=modified_nc, features_health=None)
        if action.get('action') in {'enter','add'}:
            with Session(bt_engine) as ts:
                t = ts.exec(select(Trade).where((Trade.symbol==symbol)&(Trade.status=='open')).order_by(Trade.created_at.desc())).first()  # type: ignore[attr-defined]
                if t is not None and t.id is not None:
                    last_fill_time[int(t.id)] = row.close_time
                    f = ts.exec(select(TradeFill).where(TradeFill.trade_id==t.id).order_by(TradeFill.timestamp.desc()).limit(1)).first()  # type: ignore[attr-defined]
                    if f is not None:
                        f.timestamp = row.close_time
                        ts.add(f); ts.commit()

    summary: Dict[str, Any] = {
        'symbol': symbol,
        'interval': itv,
        'exchange_type': ex,
        'start': rows[0].open_time.isoformat() if rows else None,
        'end': rows[-1].open_time.isoformat() if rows else None,
        'cooldown_seconds': cooldown,
        'stats': {}
    }
    with Session(bt_engine) as ts:
        closed = ts.exec(select(Trade).where((Trade.symbol==symbol)&(Trade.status=='closed')).order_by(Trade.created_at.asc())).all()  # type: ignore[attr-defined]
        open_tr = ts.exec(select(Trade).where((Trade.symbol==symbol)&(Trade.status=='open')).order_by(Trade.created_at.desc())).first()  # type: ignore[attr-defined]
    n = len(closed)
    wins = sum(1 for t in closed if (t.pnl_pct_snapshot or 0.0) >= 0)
    total_pnl = float(sum((t.pnl_pct_snapshot or 0.0) for t in closed))
    avg_pnl = (total_pnl / n) if n else 0.0
    floating_pnl = None
    if open_tr is not None and open_tr.avg_price:
        last_price = rows[-1].close if rows else open_tr.avg_price
        floating_pnl = (last_price / open_tr.avg_price) - 1.0
    summary['stats'] = {
        'n_closed_trades': n,
        'win_rate': (wins / n) if n else 0.0,
        'total_pnl_pct': total_pnl,
        'avg_pnl_pct': avg_pnl,
        'floating_pnl_pct': floating_pnl,
    }
    print(f"Backtest summary (last {days} days)")
    print(summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=30, help='Number of past days to backtest')
    args = ap.parse_args()
    run_backtest(args.days)

if __name__ == '__main__':
    main()
