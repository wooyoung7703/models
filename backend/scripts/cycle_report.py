from __future__ import annotations

from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List

from sqlmodel import Session, select, create_engine

# Ensure repo root in sys.path for direct execution
import sys
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.app.models import Trade, TradeFill


def load_cycles(db_path: Path) -> List[Dict[str, Any]]:
    engine = create_engine(f"sqlite:///{db_path}")
    out: List[Dict[str, Any]] = []
    with Session(engine) as s:
        trades = s.exec(
            select(Trade).where(Trade.status == 'closed').order_by(Trade.created_at.asc())  # type: ignore[attr-defined]
        ).all()
        for t in trades:
            fills = s.exec(
                select(TradeFill).where(TradeFill.trade_id == t.id).order_by(TradeFill.timestamp.asc())  # type: ignore[attr-defined]
            ).all()
            fills_count = len(fills)
            qty = float(t.quantity or 0.0)
            avg = float(t.avg_price or t.entry_price or 0.0)
            leverage = int(t.leverage or 1)
            total_notional = sum(float(f.price) * float(f.quantity or 0.0) for f in fills)
            pnl_pct_net = float(t.pnl_pct_snapshot or 0.0)
            roi_leveraged_pct = pnl_pct_net * leverage
            hold_minutes = None
            if getattr(t, 'created_at', None) and getattr(t, 'closed_at', None):
                hold_minutes = int((t.closed_at - t.created_at).total_seconds() // 60)
            out.append({
                'trade_id': int(t.id),
                'created_at': t.created_at,
                'closed_at': t.closed_at,
                'fills': fills_count,
                'quantity': qty,
                'avg_entry_price': avg,
                'total_buy_notional': total_notional,
                'pnl_pct_net': pnl_pct_net,
                'roi_leveraged_pct': roi_leveraged_pct,
                'hold_minutes': hold_minutes,
            })
    return out


def summarize(cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not cycles:
        return {}
    pnls = [c['pnl_pct_net'] for c in cycles]
    fills = [c['fills'] for c in cycles]
    notionals = [c['total_buy_notional'] for c in cycles]
    max_fills = max(fills)
    max_fills_ids = [c['trade_id'] for c in cycles if c['fills'] == max_fills]
    return {
        'n_cycles': len(cycles),
        'wr': sum(1 for p in pnls if p >= 0) / len(pnls),
        'total_pnl_pct': sum(pnls),
        'avg_pnl_pct': mean(pnls),
        'median_pnl_pct': median(pnls),
        'avg_fills': mean(fills),
        'median_fills': median(fills),
        'max_fills': max_fills,
        'max_fills_trade_ids': max_fills_ids,
        'avg_total_notional': mean(notionals),
        'median_total_notional': median(notionals),
    }


def main():
    db_path = Path('backend/data/backtest_tmp_generic.db').resolve()
    if not db_path.exists():
        print(f"Backtest DB not found: {db_path}")
        return
    cycles = load_cycles(db_path)
    summ = summarize(cycles)
    print("Cycle summary:", summ)
    # Print first 10 cycles as a sample
    print("trade_id,created_at,closed_at,fills,quantity,avg_entry,tot_buy_notional,pnl_pct_net,roi_leveraged_pct,hold_min")
    for c in cycles[:10]:
        print(
            f"{c['trade_id']},{c['created_at']},{c['closed_at']},{c['fills']},{c['quantity']:.0f},{c['avg_entry_price']:.6f},{c['total_buy_notional']:.6f},{c['pnl_pct_net']:.6f},{c['roi_leveraged_pct']:.6f},{c['hold_minutes']}"
        )


if __name__ == '__main__':
    main()
