"""Backtest last N days via TradeManager (as in fills script), then compute capitalized PnL with fees.

- Reuses the exact TM-driven entry/add/exit logic to generate trades/fills into a temp SQLite.
- After simulation, maps each closed trade to notional sizing and computes PnL in USDT with fees.

Run example (PowerShell):
  $env:EXCHANGE_TYPE='futures'; $env:STACKING_THRESHOLD='-1'; \
  python -m backend.scripts.backtest_capitalize_from_tm --days 90 \
      --initial-notional 2000 --add-notional 1000 --taker-fee 0.0004
"""
from __future__ import annotations
from pathlib import Path
import sys
import argparse
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


def simulate(days: int) -> Dict[str, Any]:
    symbol = settings.SYMBOL; ex = settings.EXCHANGE_TYPE; itv = settings.INTERVAL
    bt_path = _ROOT / 'backend' / 'data' / 'bt_capital_tmp.db'
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
    last_fill_time = {}
    cooldown = settings.ADD_COOLDOWN_SECONDS
    for row in rows:
        nc = predictor.predict_from_row(row, price_override=row.close, price_source='closed').to_dict()
        # cooldown gating for adds as in fills script
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
        fills = ts.exec(select(TradeFill).order_by(TradeFill.timestamp.asc())).all()  # type: ignore[attr-defined]
    # Aggregate adds per trade
    adds_map: Dict[int, int] = {}
    for f in fills:
        adds_map[f.trade_id] = adds_map.get(f.trade_id, 0) + 1
    stats_trades = []
    for t in closed:
        tid = int(t.id) if t.id is not None else -1
        stats_trades.append({
            'id': tid,
            'adds_done': int(t.adds_done or 0),
            'fills_total': adds_map.get(tid, 0),
            'pnl_pct': float(t.pnl_pct_snapshot or 0.0),
        })
    return {
        'symbol': symbol,
        'interval': itv,
        'days': days,
        'trades': stats_trades,
    }


def compute_capitalized(trades: list[dict], initial_notional: float, add_notional: float, fee_rate: float) -> Dict[str, Any]:
    total_net = 0.0
    total_fees = 0.0
    wins = 0
    losses = 0
    adds_hist = []
    for t in trades:
        fills_total = int(t['fills_total'])
        adds = max(0, fills_total - 1)
        total_notional = initial_notional + add_notional * adds
        pnl_pct = float(t['pnl_pct'])
        gross = total_notional * pnl_pct
        # fees: entry (initial_notional) + adds (add_notional*adds) + exit (total_notional)
        fees = initial_notional * fee_rate + add_notional * adds * fee_rate + total_notional * fee_rate
        net = gross - fees
        total_net += net
        total_fees += fees
        adds_hist.append(adds)
        if pnl_pct > 0:
            wins += 1
        else:
            losses += 1
    return {
        'n_trades_closed': len(trades),
        'wins': wins,
        'losses': losses,
        'avg_adds_per_trade': (sum(adds_hist)/len(adds_hist)) if adds_hist else 0.0,
        'net_pnl_usdt': round(total_net, 2),
        'fees_usdt': round(total_fees, 2),
        'net_return_pct_on_1k': round(100.0 * (total_net / 1000.0), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=90)
    ap.add_argument('--initial-notional', type=float, default=2000.0)
    ap.add_argument('--add-notional', type=float, default=1000.0)
    ap.add_argument('--taker-fee', type=float, default=0.0004)
    args = ap.parse_args()

    sim = simulate(days=args.days)
    cap = compute_capitalized(sim['trades'], initial_notional=args.initial_notional, add_notional=args.add_notional, fee_rate=args.taker_fee)
    out = {**sim, **cap, 'threshold_env': getattr(settings, 'STACKING_THRESHOLD', None)}
    print(out)


if __name__ == '__main__':
    main()
