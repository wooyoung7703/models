from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Any

from sqlmodel import Session, select

from .models import Trade, TradeFill
from .core.config import settings
try:
    from .scheduler import trigger_trade_close_retrain
except Exception:
    trigger_trade_close_retrain = None  # type: ignore


@dataclass
class EntryConfig:
    # Relaxed thresholds to increase signal coverage after initial zero-trade backtest.
    # Adjust gradually if trade quality deteriorates.
    min_margin: float = 0.01      # >= +1% above threshold (was 0.02)
    min_conf: float = 0.08        # confidence (|margin|) minimum (was 0.10)
    min_bottom: float = 0.55      # heuristic bottom score (was 0.60)
    min_z: float = 1.2            # raw stacking logit intensity (was 1.5)
    require_fresh: bool = True
    require_no_gaps: bool = True
    require_tf_ok: bool = True


class TradeManager:
    """Simple long-only manager with unlimited DCA (cooldown + price drop).

    - Opens trade when entry conditions satisfied
    - Adds only if current price < last fill price and cooldown passed
    - Closes when price crosses TP/SL relative to avg_price
    - Leverage is informational here; TP/SL are on underlying move (e.g., +1%, -0.5%)
    """

    def __init__(self, session_factory, leverage: int = 10, add_cooldown_seconds: int = 600):
        self.session_factory = session_factory
        self.leverage = leverage
        # Minimum time between fills (including entry -> first add)
        self.add_cooldown_seconds = max(0, int(add_cooldown_seconds))
        # Trailing TP per-trade runtime state (not persisted). Keyed by trade.id
        self._trail_state: Dict[int, Dict[str, Any]] = {}

    # ---------- Entry heuristics using nowcast -----------
    def _passes_entry(self, nowcast: Dict[str, Any], features_health: Optional[Dict[str, Any]] = None) -> bool:
        # Simplified: only stacking threshold decision required.
        s = (nowcast or {}).get('stacking') or {}
        if not s.get('ready'):
            return False
        # Stacking decision is base requirement
        if not s.get('decision'):
            return False
        # Optional precision gate via entry_meta
        try:
            from .core.config import settings as _s
            if getattr(_s, 'ENTRY_META_GATE_ENABLED', True):
                em = s.get('entry_meta') if isinstance(s, dict) else None
                if em is None or not isinstance(em, dict):
                    return False  # require meta block present
                if not em.get('entry_decision'):
                    return False
        except Exception:
            # Fail safe: treat as not passing if config resolution fails
            return False
        # Optional: still enforce basic data freshness/gap checks if provided.
        cfg = EntryConfig()
        if features_health:
            fresh_sec = features_health.get('data_fresh_seconds')
            if cfg.require_fresh and (fresh_sec is None or float(fresh_sec) >= 300.0):
                return False
            miss = int(features_health.get('missing_minutes_24h') or 0)
            if cfg.require_no_gaps and miss >= 10:
                return False
            has5 = bool(features_health.get('5m_latest_open_time'))
            has15 = bool(features_health.get('15m_latest_open_time'))
            if cfg.require_tf_ok and not (has5 and has15):
                return False
        return True

    # ---------- Public API -----------
    def get_open_trade(self, session: Session, symbol: str) -> Optional[Trade]:
        return session.exec(
            select(Trade).where((Trade.symbol == symbol) & (Trade.status == 'open')).order_by(Trade.created_at.desc())  # type: ignore[attr-defined]
        ).first()

    def _create_trade(self, session: Session, symbol: str, price: float, interval: str, exchange_type: str, strategy_json: Optional[str] = None) -> Trade:
        t = Trade(
            symbol=symbol, exchange_type=exchange_type, interval=interval,
            side='long', leverage=self.leverage, status='open',
            entry_price=price, avg_price=price, quantity=1.0,
            strategy_json=strategy_json,
        )
        # Apply runtime defaults from settings
        try:
            # TP/SL defaults
            tp_pct = float(getattr(settings, 'TAKE_PROFIT_PCT', t.take_profit_pct))
            sl_pct = getattr(settings, 'STOP_LOSS_PCT', t.stop_loss_pct)
            t.take_profit_pct = tp_pct
            if sl_pct is not None:
                t.stop_loss_pct = float(sl_pct)
            # Initialize trailing state for this trade id
            # Config is read from settings during evaluation
        except Exception:
            pass
        session.add(t)
        session.flush()
        # t.id should be populated after flush; add guard for type checker
        trade_id = int(t.id)  # type: ignore[arg-type]
        f = TradeFill(trade_id=trade_id, price=price, quantity=1.0, symbol=symbol, exchange_type=exchange_type, interval=interval)
        session.add(f)
        # Initialize trail state container
        self._trail_state[trade_id] = {"active": False, "high": None}
        return t

    def _add_fill(self, session: Session, trade: Trade, price: float):
        # Unlimited adds; cooldown + last-fill price rule only
        # Removed avg_price check: allow add as long as below last fill
        new_qty = trade.quantity + 1.0
        trade.avg_price = (trade.avg_price * trade.quantity + price) / new_qty
        trade.quantity = new_qty
        trade.adds_done += 1
        trade_id = int(trade.id)  # type: ignore[arg-type]
        f = TradeFill(trade_id=trade_id, price=price, quantity=1.0, symbol=trade.symbol, exchange_type=trade.exchange_type, interval=trade.interval)
        session.add(f)

    def _last_fill_time(self, session: Session, trade: Trade) -> Optional[datetime]:
        row = session.exec(
            select(TradeFill.timestamp).where(TradeFill.trade_id == trade.id).order_by(TradeFill.timestamp.desc()).limit(1)  # type: ignore[attr-defined]
        ).first()
        return row if isinstance(row, datetime) else (row or None)

    def _maybe_close(self, session: Session, trade: Trade, price: float) -> bool:
        # Compute underlying move from avg_price
        if trade.avg_price <= 0:
            return False
        move = (price / trade.avg_price) - 1.0
        trade.last_price = price
        # For decision thresholds, default to gross move; net-of-fees computed below
        trade.pnl_pct_snapshot = move

        # Pre-compute net-of-fees move estimate
        try:
            fee_in = float(getattr(settings, 'FEE_ENTRY_PCT', 0.0))
            fee_out = float(getattr(settings, 'FEE_EXIT_PCT', 0.0))
            net_move = move - fee_in - fee_out * (price / trade.avg_price)
        except Exception:
            net_move = move

        # Stop loss (skip if disabled)
        if not bool(getattr(settings, 'DISABLE_STOP_LOSS', False)) and (move <= trade.stop_loss_pct):
            trade.status = 'closed'
            trade.closed_at = datetime.utcnow()
            # snapshot net-of-fees pnl
            trade.pnl_pct_snapshot = net_move
            # cleanup trail
            try:
                tid = int(trade.id)  # type: ignore[arg-type]
                self._trail_state.pop(tid, None)
            except Exception:
                pass
            return True
        # Take profit: support trailing mode
        if str(getattr(settings, 'TP_MODE', 'fixed')).lower() == 'trailing':
            trigger = float(getattr(settings, 'TP_TRIGGER', trade.take_profit_pct))
            step = max(1e-9, float(getattr(settings, 'TP_STEP', 0.0005)))
            giveback = float(getattr(settings, 'TP_GIVEBACK', 0.0))
            # Activate when reaching trigger
            tid = int(trade.id)  # type: ignore[arg-type]
            st = self._trail_state.setdefault(tid, {"active": False, "high": None})
            if (not st.get("active")) and (move >= trigger):
                st["active"] = True
                st["high"] = move
            # Update and check retrace below previous floor - giveback
            if st.get("active"):
                high = float(st.get("high") or move)
                if move > high:
                    high = move
                    st["high"] = high
                steps_achieved = int(max(0.0, (high - trigger)) // step)
                prev_floor = trigger if steps_achieved == 0 else (trigger + (steps_achieved - 1) * step)
                if move <= (prev_floor - giveback):
                    trade.status = 'closed'
                    trade.closed_at = datetime.utcnow()
                    trade.pnl_pct_snapshot = net_move
                    # cleanup trail
                    try:
                        self._trail_state.pop(tid, None)
                    except Exception:
                        pass
                    return True
        else:
            use_net = bool(getattr(settings, 'TP_DECISION_ON_NET', False))
            decision_move = net_move if use_net else move
            if decision_move >= trade.take_profit_pct:
                trade.status = 'closed'
                trade.closed_at = datetime.utcnow()
                trade.pnl_pct_snapshot = net_move
                try:
                    tid = int(trade.id)  # type: ignore[arg-type]
                    self._trail_state.pop(tid, None)
                except Exception:
                    pass
                return True
        return False

    def process(self, symbol: str, interval: str, exchange_type: str, price: float, nowcast: Dict[str, Any], features_health: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate and act on trading rules. Returns action summary."""
        action = {"symbol": symbol, "price": price, "action": "hold"}
        with self.session_factory() as session:
            trade = self.get_open_trade(session, symbol)
            if trade is None:
                # consider entry
                if self._passes_entry(nowcast, features_health):
                    trade = self._create_trade(session, symbol, price, interval, exchange_type, strategy_json=str(nowcast.get('stacking')))
                    action.update({"action": "enter", "trade_id": trade.id, "avg_price": trade.avg_price, "qty": trade.quantity})
                else:
                    session.commit()
                    return action
            else:
                # manage existing: TP/SL check first
                if self._maybe_close(session, trade, price):
                    action.update({"action": "close", "trade_id": trade.id, "pnl_pct": trade.pnl_pct_snapshot})
                    session.commit()
                    # Event-driven retrain trigger (if enabled & cooldown satisfied)
                    try:
                        if trigger_trade_close_retrain:
                            trigger_trade_close_retrain()
                    except Exception:
                        pass
                    return action
                # DCA add if price lower and adds remain and signal still ON and cooldown passed
                s = (nowcast or {}).get('stacking') or {}
                # Restored: require stacking.decision plus (if enabled) entry_meta.entry_decision for adds.
                can_add_signal = bool(s.get('decision'))
                try:
                    from .core.config import settings as _s
                    if getattr(_s, 'ENTRY_META_GATE_ENABLED', True):
                        em = s.get('entry_meta') if isinstance(s, dict) else None
                        if not (isinstance(em, dict) and em.get('entry_decision')):
                            can_add_signal = False
                except Exception:
                    can_add_signal = False
                # Removed avg_price gate; rely on last fill price strictly below check
                if can_add_signal:
                    can_add = True
                    if self.add_cooldown_seconds > 0:
                        last_fill = self._last_fill_time(session, trade)
                        if last_fill is not None:
                            dt_now = datetime.utcnow()
                            if (dt_now - last_fill).total_seconds() < self.add_cooldown_seconds:
                                can_add = False
                    # Enforce last-fill price rule: only add if price < previous fill price
                    if can_add:
                        last_fill_row = session.exec(
                            select(TradeFill).where(TradeFill.trade_id == trade.id).order_by(TradeFill.timestamp.desc()).limit(1)  # type: ignore[attr-defined]
                        ).first()
                        if last_fill_row is not None and price >= float(getattr(last_fill_row, 'price', price)):
                            can_add = False
                    if can_add:
                        self._add_fill(session, trade, price)
                        action.update({"action": "add", "trade_id": trade.id, "avg_price": trade.avg_price, "qty": trade.quantity, "adds_done": trade.adds_done})
            session.commit()
        return action
