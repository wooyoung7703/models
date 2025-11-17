"""Sweep ADAPTIVE_THRESHOLD_QUANTILE values and report trade count vs net PnL.

Example (PowerShell):
  $env:EXCHANGE_TYPE='futures'; $env:STACKING_THRESHOLD='-1'; \
  python -m backend.scripts.sweep_adaptive_quantile --days 90 --initial-notional 1500 \
    --taker-fee 0.0004 --equal-add --tp 0.01 --cooldown 600 \
    --quantiles 0.95,0.96,0.97,0.98

Notes:
- Requires STACKING_THRESHOLD=-1 to engage adaptive threshold.
- Uses backtest_capitalized.backtest() with adaptive_quantile_override for each run.
- No SL; TP is net-of-fees by default; equal-add supported.
"""
from __future__ import annotations
import sys
from pathlib import Path
import argparse
from typing import List, Dict, Any

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
    'quantile', 'n_trades_closed', 'wins', 'losses', 'avg_adds_per_trade',
    'net_pnl_usdt', 'net_return_pct_on_1k'
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
    ap.add_argument('--tp', type=float, default=0.01)
    ap.add_argument('--tp-includes-fees', dest='tp_includes_fees', action='store_true')
    ap.add_argument('--no-tp-includes-fees', dest='tp_includes_fees', action='store_false')
    ap.set_defaults(tp_includes_fees=True)
    ap.add_argument('--equal-add', action='store_true')
    ap.add_argument('--cooldown', type=int, default=None)
    ap.add_argument('--quantiles', type=_parse_list_floats, required=True,
                    help='Comma-separated list, e.g., 0.95,0.96,0.97,0.98')
    args = ap.parse_args()

    results: List[Dict[str, Any]] = []
    for q in args.quantiles:
        out = backtest(days=args.days, initial_notional=args.initial_notional, add_notional=args.add_notional,
                       leverage=args.leverage, taker_fee=args.taker_fee, maker_fee=args.maker_fee, use_maker=args.use_maker,
                       tp=args.tp, tp_includes_fees=args.tp_includes_fees, equal_add=args.equal_add,
                       cooldown_override=args.cooldown, adaptive_quantile_override=q)
        out['quantile'] = q
        results.append(out)

    print("\n=== Trade count vs PnL (adaptive quantile sweep) ===")
    for r in results:
        row = {k: r.get(k) for k in cols}
        print(row)

    # Simple best-by-net pnl
    best = max(results, key=lambda d: d.get('net_pnl_usdt', -1e18)) if results else None
    if best:
        print("\nBest by net_pnl_usdt:")
        print({k: best.get(k) for k in cols})


if __name__ == '__main__':
    main()
