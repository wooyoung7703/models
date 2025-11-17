"""Sweep adaptive quantiles and cooldowns to increase entry count and observe PnL.

Example (PowerShell):
  $env:EXCHANGE_TYPE='futures'; $env:STACKING_THRESHOLD='-1'
  python -m backend.scripts.sweep_quantile_cooldown --days 90 --initial-notional 1500 \
    --taker-fee 0.0004 --equal-add --tp 0.01 \
    --quantiles 0.90,0.92,0.94,0.96 --cooldowns 600,300,120,60
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


def _parse_list_ints(v: str) -> List[int]:
    parts = [p.strip() for p in v.split(',') if p.strip()]
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid int: {p}")
    if not out:
        raise argparse.ArgumentTypeError("Empty list")
    return out


cols = (
    'quantile', 'cooldown_seconds', 'n_trades_closed', 'wins', 'losses',
    'avg_adds_per_trade', 'net_pnl_usdt', 'net_return_pct_on_1k'
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
    ap.add_argument('--quantiles', type=_parse_list_floats, required=True)
    ap.add_argument('--cooldowns', type=_parse_list_ints, required=True)
    args = ap.parse_args()

    results: List[Dict[str, Any]] = []
    for q in args.quantiles:
        for cd in args.cooldowns:
            out = backtest(days=args.days, initial_notional=args.initial_notional, add_notional=args.add_notional,
                           leverage=args.leverage, taker_fee=args.taker_fee, maker_fee=args.maker_fee, use_maker=args.use_maker,
                           tp=args.tp, tp_includes_fees=args.tp_includes_fees, equal_add=args.equal_add,
                           cooldown_override=cd, adaptive_quantile_override=q)
            out['quantile'] = q
            results.append(out)

    print("\n=== Entry count vs PnL (quantile x cooldown sweep) ===")
    for r in results:
        row = {k: r.get(k) for k in cols}
        print(row)

    # Best by n_trades then by PnL (tie-breaker)
    results.sort(key=lambda d: (d.get('n_trades_closed', 0), d.get('net_pnl_usdt', -1e18)), reverse=True)
    if results:
        best = results[0]
        print('\nBest by entry count then PnL:')
        print({k: best.get(k) for k in cols})


if __name__ == '__main__':
    main()
