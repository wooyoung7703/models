"""Sweep adaptive quantiles and TP targets (no SL) to study entry count, PnL, and risk.

Example (PowerShell):
  $env:EXCHANGE_TYPE='futures'; $env:STACKING_THRESHOLD='-1'
  python -m backend.scripts.sweep_quantile_tp --days 90 --initial-notional 1500 \
    --taker-fee 0.0004 --equal-add --cooldown 60 \
    --quantiles 0.88,0.89,0.90,0.91,0.92 --tps 0.008,0.009,0.010,0.011,0.012

Notes:
- Requires adaptive mode (STACKING_THRESHOLD=-1) to utilize quantile override.
- TP is net-of-fees by default. SL is not used by backtest_capitalized.
- Outputs table per (quantile, tp) and reports: max by entries, max by PnL, and a balanced pick.
"""
from __future__ import annotations
import sys
from pathlib import Path
import argparse
from typing import List, Dict, Any, Tuple

_THIS = Path(__file__).resolve(); _ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.scripts.backtest_capitalized import backtest


def _parse_list_floats(v: str) -> List[float]:
    parts = [p.strip() for p in v.split(',') if p.strip()]
    out: List[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid float: {p}")
    if not out:
        raise argparse.ArgumentTypeError("Empty list")
    return out


cols = (
    'quantile', 'tp', 'n_trades_closed', 'wins', 'losses',
    'avg_adds_per_trade', 'net_pnl_usdt',
    'net_return_pct_on_1k', 'worst_unrealized_move_pct'
)


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmax - vmin < 1e-9:
        return [0.5 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=90)
    ap.add_argument('--initial-notional', type=float, default=2000.0)
    ap.add_argument('--add-notional', type=float, default=1000.0)
    ap.add_argument('--leverage', type=int, default=10)
    ap.add_argument('--taker-fee', type=float, default=0.0004)
    ap.add_argument('--maker-fee', type=float, default=0.0002)
    ap.add_argument('--use-maker', action='store_true')
    ap.add_argument('--tp-includes-fees', dest='tp_includes_fees', action='store_true')
    ap.add_argument('--no-tp-includes-fees', dest='tp_includes_fees', action='store_false')
    ap.set_defaults(tp_includes_fees=True)
    ap.add_argument('--equal-add', action='store_true')
    ap.add_argument('--cooldown', type=int, default=None)
    ap.add_argument('--quantiles', type=_parse_list_floats, required=True)
    ap.add_argument('--tps', type=_parse_list_floats, required=True)
    args = ap.parse_args()

    results: List[Dict[str, Any]] = []
    for q in args.quantiles:
        for tp in args.tps:
            out = backtest(days=args.days, initial_notional=args.initial_notional, add_notional=args.add_notional,
                           leverage=args.leverage, taker_fee=args.taker_fee, maker_fee=args.maker_fee, use_maker=args.use_maker,
                           tp=tp, tp_includes_fees=args.tp_includes_fees, equal_add=args.equal_add,
                           cooldown_override=args.cooldown, adaptive_quantile_override=q)
            out['quantile'] = q
            out['tp'] = tp
            results.append(out)

    # Print table
    print("\n=== Entry count / PnL / Risk (quantile x tp) ===")
    for r in results:
        row = {k: r.get(k) for k in cols}
        print(row)

    if not results:
        return

    # Picks: max entries, max PnL
    best_entries = max(results, key=lambda d: (d.get('n_trades_closed', 0), d.get('net_pnl_usdt', -1e18)))
    best_pnl = max(results, key=lambda d: d.get('net_pnl_usdt', -1e18))

    # Balanced pick (normalized multi-criteria):
    # maximize entries and PnL; minimize avg_adds and |worst_unrealized_move_pct|
    n_entries = [r.get('n_trades_closed', 0.0) for r in results]
    pnls = [r.get('net_pnl_usdt', 0.0) for r in results]
    avg_adds = [r.get('avg_adds_per_trade', 0.0) for r in results]
    worsts = [abs(r.get('worst_unrealized_move_pct', 0.0) or 0.0) for r in results]

    ne_norm = _normalize(n_entries)
    pnl_norm = _normalize(pnls)
    # For risk terms we want smaller is better; convert to 1 - normalized
    adds_norm = _normalize(avg_adds)
    worst_norm = _normalize(worsts)
    adds_inv = [1.0 - x for x in adds_norm]
    worst_inv = [1.0 - x for x in worst_norm]

    # weights: entries 0.35, pnl 0.35, adds 0.15, worst 0.15
    scores = [0.35*ne + 0.35*pp + 0.15*ad + 0.15*wm for ne, pp, ad, wm in zip(ne_norm, pnl_norm, adds_inv, worst_inv)]
    best_bal_idx = max(range(len(results)), key=lambda i: scores[i])
    best_bal = results[best_bal_idx]

    print("\nBest by entries:")
    print({k: best_entries.get(k) for k in cols})
    print("\nBest by net_pnl_usdt:")
    print({k: best_pnl.get(k) for k in cols})
    print("\nBest balanced (entries+PnL high, adds/worst low):")
    print({k: best_bal.get(k) for k in cols})


if __name__ == '__main__':
    main()
