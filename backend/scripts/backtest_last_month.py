"""Backtest the current trading rules over the last ~30 days using DB candles.

This script:
- Reads candles for settings.SYMBOL / settings.EXCHANGE_TYPE / settings.INTERVAL
- Runs the RealTimePredictor to produce nowcast per candle
- Applies TradeManager logic (entry/TP/SL/add) with a custom cooldown gate based on candle time
- Stores simulated trades into a temporary SQLite file (to avoid polluting main DB)
- Prints summary stats (trades, win rate, avg pnl, total pnl) and a brief trade list

Run from repo root:
  python backend/scripts/backtest_last_month.py
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Ensure repo root on path
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sqlmodel import Session, select, SQLModel, create_engine
from backend.app.core.config import settings
from backend.app.db import engine as main_engine
from backend.app.models import Candle, Trade, TradeFill
from backend.app.predictor import RealTimePredictor
from backend.app.model_adapters import registry
from backend.app.trade_manager import TradeManager


def backtest(days: int = 30) -> Dict[str, Any]:
    symbol = settings.SYMBOL
    ex = settings.EXCHANGE_TYPE
    itv = settings.INTERVAL
    # Create temp backtest DB for simulated trades
    bt_path = _ROOT / 'backend' / 'data' / 'backtest_tmp.db'
    bt_url = f"sqlite:///{bt_path}"
    bt_engine = create_engine(bt_url, echo=False)
    # Create only trades tables in temp DB
    # For simplicity, create full metadata
    SQLModel.metadata.create_all(bt_engine)

    # Load models
    try:
        if not registry.adapters and not registry.stacking:
            registry.load_from_settings()
    except Exception:
        pass

    # Query candles for last N days
    end_dt = None
    results = []
    with Session(main_engine) as s:
        # get max open_time
        last = s.exec(
            select(Candle).where((Candle.symbol==symbol)&(Candle.exchange_type==ex)&(Candle.interval==itv))  # type: ignore[attr-defined]
            .order_by(Candle.open_time.desc()).limit(1)
        ).first()
        if not last:
            raise RuntimeError("No candles found for symbol. Fill DB first.")
        end_dt = last.open_time
        start_dt = end_dt - timedelta(days=days)
        rows = s.exec(
            select(Candle).where((Candle.symbol==symbol)&(Candle.exchange_type==ex)&(Candle.interval==itv)&(Candle.open_time>=start_dt)&(Candle.open_time<=end_dt))  # type: ignore[attr-defined]
            .order_by(Candle.open_time.asc())
        ).all()

    predictor = RealTimePredictor(symbol=symbol, interval=itv)
    # Trade manager with cooldown disabled (we'll gate adds ourselves by candle time)
    tm = TradeManager(lambda: Session(bt_engine), leverage=10, add_cooldown_seconds=0)

    # Track last fill candle time per trade id for cooldown simulation
    last_fill_time: Dict[int, datetime] = {}
    cooldown = settings.ADD_COOLDOWN_SECONDS

    for row in rows:
        nc = predictor.predict_from_row(row, price_override=row.close, price_source='closed')
        nc_d = nc.to_dict()
        # Decide if we need to block adds due to cooldown
        with Session(bt_engine) as ts:
            open_trade = ts.exec(
                select(Trade).where((Trade.symbol==symbol)&(Trade.status=='open')).order_by(Trade.created_at.desc())  # type: ignore[attr-defined]
            ).first()
        modified_nc = nc_d
        if open_trade is not None:
            # Compute add cooldown eligibility from last_fill_time map
            tid = int(open_trade.id)  # type: ignore[arg-type]
            last_ts = last_fill_time.get(tid)
            can_add_now = True
            if cooldown > 0 and last_ts is not None:
                elapsed = (row.close_time - last_ts).total_seconds()
                if elapsed < cooldown:
                    can_add_now = False
            if not can_add_now:
                # Disable stacking decision for this tick to prevent TM from adding
                sblk = dict(modified_nc.get('stacking') or {})
                sblk['decision'] = False
                modified_nc = {**modified_nc, 'stacking': sblk}

        action = tm.process(symbol=symbol, interval=itv, exchange_type=ex, price=row.close, nowcast=modified_nc, features_health=None)
        # If a fill happened, update the last fill time to candle timestamp
        if action.get('action') in {'enter','add'}:
            with Session(bt_engine) as ts:
                t = ts.exec(
                    select(Trade).where((Trade.symbol==symbol)&(Trade.status=='open')).order_by(Trade.created_at.desc())  # type: ignore[attr-defined]
                ).first()
                if t is not None and t.id is not None:
                    # Update last fill time map
                    last_fill_time[int(t.id)] = row.close_time
                    # Optionally align last TradeFill timestamp to candle time (best effort)
                    f = ts.exec(
                        select(TradeFill).where(TradeFill.trade_id==t.id).order_by(TradeFill.timestamp.desc()).limit(1)  # type: ignore[attr-defined]
                    ).first()
                    if f is not None:
                        f.timestamp = row.close_time
                        ts.add(f)
                        ts.commit()

    # Summarize results
    summary: Dict[str, Any] = {
        'symbol': symbol,
        'interval': itv,
        'exchange_type': ex,
        'start': rows[0].open_time.isoformat() if rows else None,
        'end': rows[-1].open_time.isoformat() if rows else None,
        'cooldown_seconds': cooldown,
        'stats': {}
    }
    closed = []
    open_tr = None
    with Session(bt_engine) as ts:
        closed = ts.exec(
            select(Trade).where((Trade.symbol==symbol)&(Trade.status=='closed')).order_by(Trade.created_at.asc())  # type: ignore[attr-defined]
        ).all()
        open_tr = ts.exec(
            select(Trade).where((Trade.symbol==symbol)&(Trade.status=='open')).order_by(Trade.created_at.desc())  # type: ignore[attr-defined]
        ).first()

    n = len(closed)
    wins = sum(1 for t in closed if (t.pnl_pct_snapshot or 0.0) >= 0)
    total_pnl = float(sum((t.pnl_pct_snapshot or 0.0) for t in closed))
    avg_pnl = (total_pnl / n) if n else 0.0
    # Optional: include floating pnl for open trade at last price
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

    # Print concise report
    print("Backtest summary (last %d days)" % days)
    print(summary)
    return summary


if __name__ == '__main__':
    backtest(30)
