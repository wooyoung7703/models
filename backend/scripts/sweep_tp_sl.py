"""Grid-search TP/SL thresholds for capital-aware backtest (fees-inclusive TP supported).

Example (PowerShell):
  $env:EXCHANGE_TYPE='futures'; $env:STACKING_THRESHOLD='0.78'; \
  python -m backend.scripts.sweep_tp_sl --days 90 --initial-notional 1500 --taker-fee 0.0004 \
      --tp-grid 0.008,0.01,0.012 --sl-grid -0.004,-0.005,-0.006 --equal-add

Notes:
- Respects current environment-driven predictor settings (symbol, interval, threshold mode).
- Uses the same backtest() as backtest_capitalized to ensure identical fee/exit logic.
- Sorts results by net_pnl_usdt desc and prints the top N.
"""
from __future__ import annotations
import sys
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Any

_THIS = Path(__file__).resolve(); _ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.scripts.backtest_capitalized import backtest


def _parse_grid(val: str) -> List[float]:
    parts = [p.strip() for p in val.split(',') if p.strip()]
    out: List[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid float in grid: '{p}'")
    if not out:
        raise argparse.ArgumentTypeError("Grid cannot be empty")
    return out


essential_columns = (
    'tp', 'sl', 'n_trades_closed', 'wins', 'losses',
    'avg_adds_per_trade', 'max_adds_per_trade', 'net_pnl_usdt', 'net_return_pct_on_1k'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=90)
    ap.add_argument('--initial-notional', type=float, default=2000.0)
    ap.add_argument('--add-notional', type=float, default=1000.0)
    ap.add_argument('--leverage', type=int, default=10)
    ap.add_argument('--taker-fee', type=float, default=0.0004)
    ap.add_argument('--maker-fee', type=float, default=0.0002)
    ap.add_argument('--use-maker', action='store_true')
    ap.add_argument('--tp-includes-fees', dest='tp_includes_fees', action='store_true', help='Evaluate TP as net-of-fees (default: on)')
    ap.add_argument('--no-tp-includes-fees', dest='tp_includes_fees', action='store_false')
    ap.set_defaults(tp_includes_fees=True)
    ap.add_argument('--equal-add', action='store_true', help='Use equal add sizing (add_notional = initial_notional)')
    ap.add_argument('--tp-grid', type=_parse_grid, required=True, help='Comma-separated list of TP fractions, e.g., 0.008,0.01,0.012')
    ap.add_argument('--sl-grid', type=_parse_grid, required=True, help='Comma-separated list of SL fractions, e.g., -0.004,-0.005,-0.006')
    ap.add_argument('--topk', type=int, default=10, help='Show top-K results by net_pnl_usdt')
    args = ap.parse_args()

    results: List[Dict[str, Any]] = []
    for tp in args.tp_grid:
        for sl in args.sl_grid:
            out = backtest(days=args.days, initial_notional=args.initial_notional, add_notional=args.add_notional,
                           leverage=args.leverage, taker_fee=args.taker_fee, maker_fee=args.maker_fee, use_maker=args.use_maker,
                           tp=tp, sl=sl, tp_includes_fees=args.tp_includes_fees, equal_add=args.equal_add)
            out['tp'] = tp
            out['sl'] = sl
            results.append(out)

    # Sort by net pnl desc
    results.sort(key=lambda d: d.get('net_pnl_usdt', -1e18), reverse=True)

    # Pretty summary
    print("\n=== Top results (by net_pnl_usdt) ===")
    for i, r in enumerate(results[:args.topk], 1):
        row = {k: r.get(k) for k in essential_columns}
        print(f"[{i}] ", row)

    if results:
        best = results[0]
        print("\nBest setting:")
        print({
            'tp': best.get('tp'),
            'sl': best.get('sl'),
            'net_pnl_usdt': best.get('net_pnl_usdt'),
            'net_return_pct_on_1k': best.get('net_return_pct_on_1k'),
            'n_trades_closed': best.get('n_trades_closed'),
            'wins': best.get('wins'),
            'losses': best.get('losses'),
            'avg_adds_per_trade': best.get('avg_adds_per_trade'),
            'max_adds_per_trade': best.get('max_adds_per_trade'),
        })


if __name__ == '__main__':
    main()
