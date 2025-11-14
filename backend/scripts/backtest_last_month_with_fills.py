"""Backtest last ~30 days and also report per-trade add counts and max add chain length.

Run:
  python backend/scripts/backtest_last_month_with_fills.py
"""
from __future__ import annotations
from pathlib import Path
import sys
from datetime import timedelta
from typing import Dict, Any
from sqlmodel import Session, select, SQLModel, create_engine

_THIS = Path(__file__).resolve(); _ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.app.core.config import settings
from backend.app.db import engine as main_engine
from backend.app.models import Candle, Trade, TradeFill
from backend.app.predictor import RealTimePredictor
from backend.app.trade_manager import TradeManager


def backtest(days: int = 30) -> Dict[str, Any]:
    symbol = settings.SYMBOL; ex = settings.EXCHANGE_TYPE; itv = settings.INTERVAL
    bt_path = _ROOT / 'backend' / 'data' / 'backtest_tmp_adds.db'
    bt_engine = create_engine(f'sqlite:///{bt_path}', echo=False)
    SQLModel.metadata.create_all(bt_engine)
    with Session(main_engine) as s:
        last = s.exec(select(Candle).where((Candle.symbol==symbol)&(Candle.exchange_type==ex)&(Candle.interval==itv)).order_by(Candle.open_time.desc()).limit(1)).first()  # type: ignore[attr-defined]
        if not last:
            raise RuntimeError('No candles found')
        end_dt = last.open_time; start_dt = end_dt - timedelta(days=days)
        rows = s.exec(select(Candle).where((Candle.symbol==symbol)&(Candle.exchange_type==ex)&(Candle.interval==itv)&(Candle.open_time>=start_dt)&(Candle.open_time<=end_dt)).order_by(Candle.open_time.asc())).all()  # type: ignore[attr-defined]
    predictor = RealTimePredictor(symbol=symbol, interval=itv)
    tm = TradeManager(lambda: Session(bt_engine), leverage=10, add_cooldown_seconds=0)
    last_fill_time: Dict[int, Any] = {}; cooldown = settings.ADD_COOLDOWN_SECONDS
    for row in rows:
        nc = predictor.predict_from_row(row, price_override=row.close, price_source='closed').to_dict()
        with Session(bt_engine) as ts:
            open_trade = ts.exec(select(Trade).where((Trade.symbol==symbol)&(Trade.status=='open')).order_by(Trade.created_at.desc())).first()  # type: ignore[attr-defined]
        modified = nc
        if open_trade is not None:
            tid = int(open_trade.id)  # type: ignore[arg-type]
            last_ts = last_fill_time.get(tid)
            can_add_now = True
            if cooldown > 0 and last_ts is not None:
                elapsed = (row.close_time - last_ts).total_seconds()
                if elapsed < cooldown:
                    can_add_now = False
            if not can_add_now:
                sblk = dict(modified.get('stacking') or {})
                sblk['decision'] = False
                modified = {**modified, 'stacking': sblk}
        act = tm.process(symbol=symbol, interval=itv, exchange_type=ex, price=row.close, nowcast=modified, features_health=None)
        if act.get('action') in {'enter','add'}:
            with Session(bt_engine) as ts:
                t = ts.exec(select(Trade).where((Trade.symbol==symbol)&(Trade.status=='open')).order_by(Trade.created_at.desc())).first()  # type: ignore[attr-defined]
                if t is not None and t.id is not None:
                    last_fill_time[int(t.id)] = row.close_time
                    f = ts.exec(select(TradeFill).where(TradeFill.trade_id==t.id).order_by(TradeFill.timestamp.desc()).limit(1)).first()  # type: ignore[attr-defined]
                    if f is not None:
                        f.timestamp = row.close_time
                        ts.add(f); ts.commit()
    with Session(bt_engine) as ts:
        closed = ts.exec(select(Trade).where((Trade.symbol==symbol)&(Trade.status=='closed')).order_by(Trade.created_at.asc())).all()  # type: ignore[attr-defined]
        open_tr = ts.exec(select(Trade).where((Trade.symbol==symbol)&(Trade.status=='open')).order_by(Trade.created_at.desc())).first()  # type: ignore[attr-defined]
        fills = ts.exec(select(TradeFill).order_by(TradeFill.timestamp.asc())).all()  # type: ignore[attr-defined]
    # Aggregate adds per trade
    adds_map: Dict[int, int] = {}
    for f in fills:
        adds_map[f.trade_id] = adds_map.get(f.trade_id, 0) + 1
    stats_trades = []
    for t in closed:
        stats_trades.append({
            'id': t.id,
            'adds_done': t.adds_done,
            'fills_total': adds_map.get(int(t.id), 0),
            'pnl_pct': t.pnl_pct_snapshot,
        })
    max_adds = max((t.adds_done for t in closed), default=0)
    summary = {
        'symbol': symbol,
        'interval': itv,
        'days': days,
        'n_closed_trades': len(closed),
        'max_adds_done': max_adds,
        'trades': stats_trades,
    }
    print('Backtest adds summary')
    print(summary)
    return summary

if __name__ == '__main__':
    backtest()
