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
from collections import deque

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
    # Use same cooldown logic as live (internal to TradeManager)
    cooldown = int(settings.ADD_COOLDOWN_SECONDS)
    tm = TradeManager(lambda: Session(bt_engine), leverage=10, add_cooldown_seconds=cooldown)

    # In backtest, assume feature pipeline is healthy; ignore historical gaps to match live conditions
    missing24 = 0
    prev_time: datetime | None = None

    for row in rows:
        # Predict nowcast for this candle
        nc = predictor.predict_from_row(row, price_override=row.close, price_source='closed')
        nc_d = nc.to_dict()

        prev_time = row.close_time

        # Build features_health similar to live checks
        features_health = {
            'data_fresh_seconds': 0,  # in backtest we evaluate at candle time
            'missing_minutes_24h': int(missing24),
            '5m_latest_open_time': True,   # 5m/15m aggregates assumed derivable from 1m during backtest
            '15m_latest_open_time': True,
        }

        action = tm.process(symbol=symbol, interval=itv, exchange_type=ex, price=row.close, nowcast=nc_d, features_health=features_health)
        act = action.get('action')
        if act in {'enter','add','close'}:
            with Session(bt_engine) as ts:
                # On enter/add, update last fill timestamp to candle close_time
                if act in {'enter','add'}:
                    # fetch latest open trade
                    t = ts.exec(select(Trade).where((Trade.symbol==symbol)&(Trade.status=='open')).order_by(Trade.created_at.desc())).first()  # type: ignore[attr-defined]
                    if t is not None and t.id is not None:
                        f = ts.exec(select(TradeFill).where(TradeFill.trade_id==t.id).order_by(TradeFill.timestamp.desc()).limit(1)).first()  # type: ignore[attr-defined]
                        if f is not None:
                            f.timestamp = row.close_time
                            ts.add(f)
                        if act == 'enter':
                            # align created_at to candle time
                            t.created_at = row.close_time
                            ts.add(t)
                    ts.commit()
                elif act == 'close':
                    # Align closed trade's closed_at to candle time
                    tid = action.get('trade_id')
                    if tid is not None:
                        t = ts.exec(select(Trade).where((Trade.id==tid))).first()  # type: ignore[attr-defined]
                        if t is not None:
                            t.closed_at = row.close_time
                            ts.add(t)
                            # also align the last fill timestamp if exists
                            f = ts.exec(select(TradeFill).where(TradeFill.trade_id==t.id).order_by(TradeFill.timestamp.desc()).limit(1)).first()  # type: ignore[attr-defined]
                            if f is not None:
                                f.timestamp = row.close_time
                                ts.add(f)
                            ts.commit()

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
    ap.add_argument('--stacking-th', type=float, default=None, help='Override stacking threshold (>=0 to force env, <0 to prefer sidecar/adaptive)')
    ap.add_argument('--entry-meta-th', type=float, default=None, help='Override entry-meta threshold')
    ap.add_argument('--adaptive-q', type=float, default=None, help='Override adaptive threshold quantile (e.g., 0.9)')
    ap.add_argument('--gate', type=int, default=None, help='Override ENTRY_META_GATE_ENABLED (1/0)')
    ap.add_argument('--no-sl', action='store_true', help='Disable stop loss checks (no SL)')
    ap.add_argument('--tp-net-pct', type=float, default=None, help='Set TAKE_PROFIT_PCT and evaluate on net-of-fees (e.g., 0.01 for +1% net)')
    ap.add_argument('--tp-mode', type=str, default=None, choices=['fixed','trailing'], help='Override TP_MODE')
    ap.add_argument('--reset', action='store_true', help='Reset temp backtest DB before running')
    ap.add_argument('--enable-entry-meta', type=int, default=None, help='Enable Entry Meta computation (1/0)')
    ap.add_argument('--entry-meta-path', type=str, default=None, help='Path to entry_meta.json sidecar')
    args = ap.parse_args()
    # Runtime overrides to mirror requested test conditions
    from backend.app.core.config import settings
    if args.stacking_th is not None:
        settings.STACKING_THRESHOLD = float(args.stacking_th)
    if args.entry_meta_th is not None:
        settings.ENTRY_META_THRESHOLD = float(args.entry_meta_th)
    if args.adaptive_q is not None:
        settings.ENABLE_ADAPTIVE_THRESHOLD = True
        settings.ADAPTIVE_THRESHOLD_QUANTILE = float(args.adaptive_q)
    if args.gate is not None:
        settings.ENTRY_META_GATE_ENABLED = bool(int(args.gate))
    if args.no_sl:
        settings.DISABLE_STOP_LOSS = True
    if args.tp_net_pct is not None:
        settings.TAKE_PROFIT_PCT = float(args.tp_net_pct)
        settings.TP_DECISION_ON_NET = True
    if args.tp_mode is not None:
        settings.TP_MODE = str(args.tp_mode)
    if args.enable_entry_meta is not None:
        settings.ENABLE_ENTRY_META = bool(int(args.enable_entry_meta))
    if args.entry_meta_path is not None:
        settings.ENTRY_META_PATH = str(args.entry_meta_path)
    # Reset temp DB (delete rows) if requested
    if args.reset:
        bt_path = _ROOT / 'backend' / 'data' / 'backtest_tmp_generic.db'
        bt_url = f"sqlite:///{bt_path}"
        bt_engine = create_engine(bt_url, echo=False)
        from sqlalchemy import text
        with Session(bt_engine) as ts:
            ts.exec(text("DELETE FROM trade_fills"))
            ts.exec(text("DELETE FROM trades"))
            ts.commit()
    run_backtest(args.days)

if __name__ == '__main__':
    main()
