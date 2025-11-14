"""Capital-aware backtest over last N days using runtime nowcast decisions.

- Uses RealTimePredictor and stacking.decision as signal.
- Simulates a single long-only strategy with DCA adds and TP/SL exits.
- Applies trading fees per fill on notional value.
- Does NOT modify DB trades; runs a self-contained simulation in-memory.

Run example (PowerShell):
  $env:EXCHANGE_TYPE='futures'; $env:STACKING_THRESHOLD='-1'; \
  python -m backend.scripts.backtest_capitalized --days 90 \
      --initial-notional 2000 --add-notional 1000 --leverage 10 \
      --taker-fee 0.0004

Notes:
- Notional = price * base_qty. This simulation controls notional directly; leverage affects margin usage only, not PnL math.
- PnL in USDT = total_notional * (exit_price/avg_price - 1).
- Fees per fill = notional * fee_rate, assessed on entry, each add, and exit.
- Take-profit (tp) can be evaluated as NET of fees if --tp-includes-fees is enabled (default: on). In that case, TP +1% means "after deducting all fees including exit fee, net return >= +1%".
- Cooldown between fills uses settings.ADD_COOLDOWN_SECONDS.
"""
from __future__ import annotations
from pathlib import Path
import sys
import argparse
from datetime import timedelta
from typing import Dict, Any, Optional

from sqlmodel import Session, select

_THIS = Path(__file__).resolve(); _ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.app.core.config import settings
from backend.app.db import engine as main_engine
from backend.app.models import Candle
from backend.app.predictor import RealTimePredictor


class Position:
    def __init__(self):
        self.open: bool = False
        self.avg_price: float = 0.0
        self.total_notional: float = 0.0  # USDT
        self.fee_paid: float = 0.0        # USDT accumulated
        self.fills: int = 0
        self.last_fill_time = None        # datetime
        self.last_fill_price: Optional[float] = None
        # tracking intra-trade excursions (relative move vs avg_price)
        self.min_move: float = 0.0
        self.max_move: float = 0.0
        # trailing TP state (net-of-fees space)
        self.trail_active: bool = False
        self.trail_trigger: float = 0.0
        self.trail_step: float = 0.0
        self.trail_giveback: float = 0.0
        self.trail_high_net_return: float = 0.0

    def enter(self, price: float, notional: float, fee_rate: float, ts):
        if self.open:
            raise RuntimeError("enter called while position open")
        self.open = True
        self.avg_price = price
        self.total_notional = notional
        fee = notional * fee_rate
        self.fee_paid += fee
        self.fills += 1
        self.last_fill_time = ts
        self.last_fill_price = price
        self.min_move = 0.0
        self.max_move = 0.0
        # reset trailing
        self.trail_active = False
        self.trail_high_net_return = 0.0

    def maybe_add(self, price: float, notional: float, fee_rate: float, ts) -> bool:
        if not self.open:
            return False
        # Enforce add only when price is below both avg_price and the previous fill price
        if price >= self.avg_price:
            return False
        if self.last_fill_price is not None and price >= self.last_fill_price:
            return False
        new_notional = self.total_notional + notional
        # new avg price weighted by notional
        self.avg_price = (self.avg_price * self.total_notional + price * notional) / new_notional
        self.total_notional = new_notional
        fee = notional * fee_rate
        self.fee_paid += fee
        self.fills += 1
        self.last_fill_time = ts
        self.last_fill_price = price
        return True

    def check_exit(self, price: float, take_profit_pct: float, stop_loss_pct: float) -> Optional[Dict[str, Any]]:
        if not self.open or self.avg_price <= 0:
            return None
        move = price / self.avg_price - 1.0
        # Default TP/SL evaluation (may be overridden by fee-inclusive TP at call site)
        if move >= take_profit_pct or move <= stop_loss_pct:
            exit_notional = self.total_notional
            pnl = exit_notional * move
            return {
                "move": move,
                "exit_notional": exit_notional,
                "pnl": pnl,
                "reason": "tp" if move >= take_profit_pct else "sl",
            }
        return None

    def exit(self, exit_fee_rate: float) -> float:
        # apply exit fee on total notional
        fee = self.total_notional * exit_fee_rate
        self.fee_paid += fee
        self.open = False
        self.total_notional = 0.0
        self.avg_price = 0.0
        self.fills += 1
        return fee


def backtest(days: int, initial_notional: float, add_notional: float, leverage: int,
             taker_fee: float, maker_fee: float = 0.0002, use_maker: bool = False,
             tp: float = 0.01, sl: Optional[float] = None, tp_includes_fees: bool = True,
             tp_mode: str = 'fixed', tp_trigger: Optional[float] = None,
             tp_step: float = 0.001, tp_giveback: float = 0.0,
             equal_add: bool = False, cooldown_override: Optional[int] = None,
             adaptive_quantile_override: Optional[float] = None) -> Dict[str, Any]:
    symbol = settings.SYMBOL; ex = settings.EXCHANGE_TYPE; itv = settings.INTERVAL
    # Optional override for adaptive quantile (used by predictor)
    prev_quantile = None
    if adaptive_quantile_override is not None:
        try:
            prev_quantile = getattr(settings, 'ADAPTIVE_THRESHOLD_QUANTILE', None)
            setattr(settings, 'ADAPTIVE_THRESHOLD_QUANTILE', float(adaptive_quantile_override))
        except Exception:
            prev_quantile = None
    with Session(main_engine) as s:
        last = s.exec(select(Candle).where((Candle.symbol==symbol)&(Candle.exchange_type==ex)&(Candle.interval==itv)).order_by(Candle.open_time.desc()).limit(1)).first()  # type: ignore[attr-defined]
        if not last:
            raise RuntimeError('No candles found')
        end_dt = last.open_time; start_dt = end_dt - timedelta(days=days)
        rows = s.exec(select(Candle).where((Candle.symbol==symbol)&(Candle.exchange_type==ex)&(Candle.interval==itv)&(Candle.open_time>=start_dt)&(Candle.open_time<=end_dt)).order_by(Candle.open_time.asc())).all()  # type: ignore[attr-defined]
    predictor = RealTimePredictor(symbol=symbol, interval=itv)

    fee_rate = maker_fee if use_maker else taker_fee
    cooldown = int(cooldown_override) if (cooldown_override is not None) else int(getattr(settings, 'ADD_COOLDOWN_SECONDS', 600) or 0)

    pos = Position()
    equity_start = 1000.0  # reference capital for reporting (not directly enforced)
    equity = equity_start

    n_trades = 0
    wins = 0
    losses = 0
    adds_hist = []
    fees_total = 0.0
    realized_pnl_total = 0.0
    # per-trade net pnl tracking (realized only)
    trade_net_pnls: list[float] = []
    # worst intra-trade adverse move observed (unrealized) across all trades
    worst_unrealized_move_pct: float = 0.0

    for row in rows:
        nc = predictor.predict_from_row(row, price_override=row.close, price_source='closed').to_dict()
        ts = getattr(row, 'close_time', getattr(row, 'open_time', None))
        stacking = (nc.get('stacking') or {})
        decision = bool(stacking.get('decision'))

        if not pos.open:
            # consider entry
            if decision:
                # entry: pay fee immediately
                pos.enter(price=row.close, notional=initial_notional, fee_rate=fee_rate, ts=ts)
                n_trades += 1
            continue
        # already open -> update excursion stats and check exit first
        exit_check = None
        move = row.close / (pos.avg_price if pos.avg_price > 0 else 1.0) - 1.0
        # update intra-trade excursions
        pos.min_move = min(pos.min_move, move)
        pos.max_move = max(pos.max_move, move)
        if tp_includes_fees:
            # Compute net return if we were to exit now (including all fees: entry/add already in pos.fee_paid + exit fee)
            exit_notional_preview = pos.total_notional
            net_pnl_preview = exit_notional_preview * move - (pos.fee_paid + exit_notional_preview * fee_rate)
            net_return_preview = (net_pnl_preview / exit_notional_preview) if exit_notional_preview > 0 else -1e9
            # SL condition (gross move) if provided
            sl_hit = (sl is not None) and (move <= float(sl))
            if tp_mode == 'trailing':
                # initialize trigger default from tp if not provided
                trigger = tp if (tp_trigger is None) else tp_trigger
                # Activate trailing once trigger reached
                if not pos.trail_active and net_return_preview >= float(trigger):
                    pos.trail_active = True
                    pos.trail_high_net_return = net_return_preview
                    pos.trail_trigger = float(trigger)
                    pos.trail_step = float(tp_step)
                    pos.trail_giveback = float(tp_giveback)
                # Update trailing high if active
                if pos.trail_active:
                    if net_return_preview > pos.trail_high_net_return:
                        pos.trail_high_net_return = net_return_preview
                    # compute steps above trigger achieved by the high
                    steps_achieved = int(max(0.0, (pos.trail_high_net_return - pos.trail_trigger)) // max(1e-9, pos.trail_step))
                    # previous floor (one step below the latest level); if no full step yet, it's the trigger
                    prev_floor = pos.trail_trigger if steps_achieved == 0 else (pos.trail_trigger + (steps_achieved - 1) * pos.trail_step)
                    # Exit if we retrace below previous floor minus giveback
                    if net_return_preview <= (prev_floor - pos.trail_giveback) or sl_hit:
                        exit_check = {
                            "move": move,
                            "exit_notional": exit_notional_preview,
                            "pnl": exit_notional_preview * move,
                            "reason": "tp_trail" if not sl_hit else "sl",
                            "net_return": net_return_preview,
                        }
            else:
                if net_return_preview >= tp or sl_hit:
                    exit_check = {
                        "move": move,
                        "exit_notional": exit_notional_preview,
                        "pnl": exit_notional_preview * move,
                        "reason": "tp_net" if net_return_preview >= tp else "sl",
                        "net_return": net_return_preview,
                    }
        else:
            # Simple gross TP with optional gross SL
            if move >= tp or ((sl is not None) and (move <= float(sl))):
                exit_check = {
                    "move": move,
                    "exit_notional": pos.total_notional,
                    "pnl": pos.total_notional * move,
                    "reason": "tp" if move >= tp else "sl",
                }
        if exit_check is not None:
            # realize PnL
            pnl = exit_check['pnl']
            fees_before_exit = pos.fee_paid
            # charge exit fee and finalize fees for this position
            fee_exit = pos.exit(exit_fee_rate=fee_rate)
            realized_pnl_total += pnl
            fees_total += fee_exit
            # entry/add fees were tracked inside pos.fee_paid but not yet added to global; add them now excluding the just-added exit fee to avoid double count
            fees_total += (pos.fee_paid - fee_exit)
            # realized net pnl for this trade
            net_this_trade = pnl - (fees_before_exit + fee_exit)
            trade_net_pnls.append(net_this_trade)
            # classify win/loss
            if (exit_check.get('reason') in ("tp", "tp_net")) or (exit_check['move'] >= tp):
                wins += 1
            else:
                losses += 1
            adds_hist.append(max(0, pos.fills - 2))  # fills: entry(1)+exit(1)+adds
            # update global worst unrealized move
            worst_unrealized_move_pct = min(worst_unrealized_move_pct, pos.min_move)
            # reset fee bucket after accounting
            pos = Position()
            continue

        # no exit -> consider add (only if price < previous fill price)
        if decision and (pos.last_fill_price is not None) and (row.close < pos.last_fill_price):
            can_add = True
            if cooldown > 0 and pos.last_fill_time is not None and ts is not None:
                elapsed = (ts - pos.last_fill_time).total_seconds()
                if elapsed < cooldown:
                    can_add = False
            if can_add:
                add_amt = initial_notional if equal_add else add_notional
                pos.maybe_add(price=row.close, notional=add_amt, fee_rate=fee_rate, ts=ts)

    # Summary
    avg_adds = float(sum(adds_hist)/len(adds_hist)) if adds_hist else 0.0
    max_adds = max(adds_hist) if adds_hist else 0
    # Net = realized PnL - all fees
    net_pnl_usdt = realized_pnl_total - fees_total
    max_trade_net = max(trade_net_pnls) if trade_net_pnls else 0.0
    min_trade_net = min(trade_net_pnls) if trade_net_pnls else 0.0

    out = {
        'symbol': symbol,
        'interval': itv,
        'days': days,
        'threshold_env': getattr(settings, 'STACKING_THRESHOLD', None),
        'leverage': leverage,
        'initial_notional': initial_notional,
        'add_notional': (initial_notional if equal_add else add_notional),
        'taker_fee': taker_fee,
        'maker_fee': maker_fee,
        'use_maker': use_maker,
    'tp_includes_fees': tp_includes_fees,
    'tp_mode': tp_mode,
    'tp_trigger': (tp if (tp_trigger is None) else tp_trigger) if tp_mode == 'trailing' else None,
    'tp_step': tp_step if tp_mode == 'trailing' else None,
    'tp_giveback': tp_giveback if tp_mode == 'trailing' else None,
        'equal_add': equal_add,
    'cooldown_seconds': cooldown,
        'n_trades_closed': len(adds_hist),
        'wins': wins,
        'losses': losses,
        'avg_adds_per_trade': avg_adds,
        'max_adds_per_trade': max_adds,
        'net_pnl_usdt': round(net_pnl_usdt, 2),
        'net_return_pct_on_1k': round(100.0 * (net_pnl_usdt / equity_start), 2),
        'max_trade_net_pnl_usdt': round(max_trade_net, 2),
        'min_trade_net_pnl_usdt': round(min_trade_net, 2),
        'worst_unrealized_move_pct': round(100.0 * worst_unrealized_move_pct, 3),
    }
    print(out)
    # Restore settings
    if adaptive_quantile_override is not None and prev_quantile is not None:
        try:
            setattr(settings, 'ADAPTIVE_THRESHOLD_QUANTILE', prev_quantile)
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=90)
    ap.add_argument('--initial-notional', type=float, default=2000.0)
    ap.add_argument('--add-notional', type=float, default=1000.0)
    ap.add_argument('--leverage', type=int, default=10)
    ap.add_argument('--taker-fee', type=float, default=0.0004)
    ap.add_argument('--maker-fee', type=float, default=0.0002)
    ap.add_argument('--use-maker', action='store_true')
    ap.add_argument('--tp', type=float, default=0.01, help='Take profit threshold (fraction), default 0.01 = +1%')
    ap.add_argument('--sl', type=float, default=None, help='Optional stop loss threshold (fraction, negative). E.g., -0.005 = -0.5%')
    ap.add_argument('--tp-includes-fees', dest='tp_includes_fees', action='store_true', help='Evaluate TP as net-of-fees (default: on)')
    ap.add_argument('--no-tp-includes-fees', dest='tp_includes_fees', action='store_false')
    ap.set_defaults(tp_includes_fees=True)
    # Trailing TP options
    ap.add_argument('--tp-mode', type=str, choices=['fixed','trailing'], default='fixed', help='TP mode: fixed (default) or trailing')
    ap.add_argument('--tp-trigger', type=float, default=None, help='Trailing TP trigger net return (fraction). Defaults to --tp when not provided.')
    ap.add_argument('--tp-step', type=float, default=0.001, help='Trailing TP ratchet step (fraction), e.g., 0.001 = 0.1%')
    ap.add_argument('--tp-giveback', type=float, default=0.0, help='Trailing TP giveback below previous floor before exit (fraction). 0.0 = sell when below previous floor')
    ap.add_argument('--tp-optimal', action='store_true', help='Apply empirically optimal trailing TP profile for this strategy (trigger=0.5%, step=0.05%, giveback=0.10%).')
    ap.add_argument('--equal-add', action='store_true', help='Use equal add sizing (add_notional = initial_notional)')
    ap.add_argument('--cooldown', type=int, default=None, help='Override cooldown seconds between fills (default: use settings.ADD_COOLDOWN_SECONDS)')
    ap.add_argument('--adaptive-quantile', type=float, default=None, help='Override ADAPTIVE_THRESHOLD_QUANTILE (e.g., 0.96)')
    args = ap.parse_args()
    # Apply optimal profile if requested
    if args.tp_optimal:
        args.tp_mode = 'trailing'
        # Use provided tp as trigger if explicitly set; otherwise default to 0.005
        if args.tp is None or args.tp == 0.01:  # default tp value means not explicitly tuned by user
            args.tp = 0.005
        args.tp_trigger = args.tp
        args.tp_step = 0.0005
        args.tp_giveback = 0.001
    backtest(days=args.days, initial_notional=args.initial_notional, add_notional=args.add_notional,
             leverage=args.leverage, taker_fee=args.taker_fee, maker_fee=args.maker_fee, use_maker=args.use_maker,
             tp=args.tp, sl=args.sl, tp_includes_fees=args.tp_includes_fees, tp_mode=args.tp_mode,
             tp_trigger=args.tp_trigger, tp_step=args.tp_step, tp_giveback=args.tp_giveback,
             equal_add=args.equal_add, cooldown_override=args.cooldown, adaptive_quantile_override=args.adaptive_quantile)


if __name__ == '__main__':
    main()
