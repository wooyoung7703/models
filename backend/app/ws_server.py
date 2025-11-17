import asyncio
import signal
import json
import logging
import os
from typing import Any, Dict, Optional, Set, TYPE_CHECKING

from websockets.server import WebSocketServerProtocol, serve

from sqlmodel import Session, select

from .db import init_db, engine
from .collector import BinanceCollector
from .core.config import settings
from .models import Candle, Trade, TradeFill
from .predictor import RealTimePredictor
from .model_adapters import registry
from .gap_fill import find_missing_ranges, seed_feature_calculator
from .snapshots import load_feature_state, save_feature_state
from pathlib import Path
from datetime import datetime
try:
    from . import scheduler as _scheduler_mod  # for training status broadcast
except Exception:
    _scheduler_mod = None


WS_HOST = os.getenv("WS_HOST", "0.0.0.0")
WS_PORT = int(os.getenv("WS_PORT", "8022"))  # reuse dev port previously used by uvicorn


class AppState:
    def __init__(self) -> None:
        self.collector: BinanceCollector = BinanceCollector()
        self.trade_manager: Optional["TradeManager"] = None
        self.latest_nowcast: Dict[str, Dict[str, Any]] = {}
        self.features_snapshot: Dict[str, Dict[str, Any]] = {}
        self.clients: Set[WebSocketServerProtocol] = set()
        self._model_mtimes: Dict[str, float] = {}
        self.trainer_meta: Dict[str, Any] = {}
        self.entry_metrics: Dict[str, Any] = {}
        self.entry_history_overall: list[Dict[str, Any]] = []
        self.entry_history_by_symbol: Dict[str, list[Dict[str, Any]]] = {}

if TYPE_CHECKING:
    from .trade_manager import TradeManager


state = AppState()
shutdown_event: asyncio.Event = asyncio.Event()

def _get_model_fields(model_cls) -> list[str]:
    mf = getattr(model_cls, "model_fields", None)
    if isinstance(mf, dict):
        return list(mf.keys())
    ff = getattr(model_cls, "__fields__", None)
    if isinstance(ff, dict):
        return list(ff.keys())
    ann = getattr(model_cls, "__annotations__", None)
    if isinstance(ann, dict):
        return list(ann.keys())
    return []


async def broadcast(message: Dict[str, Any]) -> None:
    if not state.clients:
        return
    data = json.dumps(message, ensure_ascii=False)
    dead: Set[WebSocketServerProtocol] = set()
    for ws in list(state.clients):
        try:
            await ws.send(data)
        except Exception:
            dead.add(ws)
    for ws in dead:
        try:
            state.clients.discard(ws)
            # 추가 진단: 클라이언트 종료 코드/이유 로깅
            try:
                code = getattr(ws, 'close_code', None)
                reason = getattr(ws, 'close_reason', None)
                logging.info("WS client removed dead=%d close_code=%s reason=%s remaining=%d", len(dead), code, reason, len(state.clients))
            except Exception:
                pass
        except Exception:
            pass


async def model_artifact_watcher() -> None:
    """주기적으로 모델 파일 mtime을 폴링해 변경 시 레지스트리를 리로드합니다."""
    if not settings.MODEL_WATCH_ENABLED:
        return
    paths = [
        getattr(settings, 'MODEL_XGB_PATH', None),
        getattr(settings, 'MODEL_LSTM_PATH', None),
        getattr(settings, 'MODEL_TRANSFORMER_PATH', None),
        getattr(settings, 'STACKING_META_PATH', None),
        getattr(settings, 'BOTTOM_VS_FORECAST_META_PATH', None),
    ]
    paths = [p for p in paths if isinstance(p, str) and p]
    interval = max(10, int(getattr(settings, 'MODEL_WATCH_INTERVAL_SECONDS', 60)))
    import time
    import os as _os
    logging.info("model watcher 시작 (interval=%ss, paths=%s)", interval, paths)
    while True:
        changed = False
        for p in paths:
            try:
                m = _os.path.getmtime(p)
            except Exception:
                m = -1.0
            prev = state._model_mtimes.get(p)
            if prev is None:
                state._model_mtimes[p] = m
            elif m > 0 and prev is not None and m > prev:
                changed = True
                state._model_mtimes[p] = m
        if changed:
            try:
                registry.load_from_settings()
                logging.info("모델 아티팩트 변경 감지 → 레지스트리 리로드 완료: %s", registry.status())
            except Exception as e:
                logging.exception("레지스트리 리로드 실패: %s", e)
        await asyncio.sleep(interval)


async def trainer_meta_watcher() -> None:
    """주기적으로 트레이너 컨테이너 heartbeat 및 마지막 학습 메타 파일을 읽어 브로드캐스트.

    파일 경로:
      /app/backend/data/trainer_heartbeat.txt (mtime 기반)
      /app/backend/data/trainer_last_run.json (start/end 기록)
    """
    hb_path = os.getenv("TRAIN_HEARTBEAT_PATH", "/app/backend/data/trainer_heartbeat.txt")
    lr_path = os.getenv("TRAIN_LAST_RUN_PATH", "/app/backend/data/trainer_last_run.json")
    interval = int(os.getenv("TRAINER_META_POLL_SECONDS", "30"))
    import time, json
    from datetime import datetime
    logging.info("trainer_meta_watcher 시작 (interval=%ds)", interval)
    while True:
        meta: Dict[str, Any] = {}
        now_ts = time.time()
        # Heartbeat age
        try:
            st = os.stat(hb_path)
            age = now_ts - st.st_mtime
            meta["heartbeat_age_seconds"] = int(age)
            meta["heartbeat_iso"] = datetime.utcfromtimestamp(st.st_mtime).isoformat() + "Z"
            # 건강 기준: age < 2 * TRAIN_INTERVAL_SECONDS (환경 변수 사용)
            train_interval = int(os.getenv("TRAIN_INTERVAL_SECONDS", "3600"))
            meta["heartbeat_healthy"] = age < max(600, 2 * train_interval)
        except Exception as e:
            meta["heartbeat_error"] = str(e)
        # Last run
        try:
            if os.path.exists(lr_path):
                with open(lr_path, "r", encoding="utf-8") as f:
                    lr = json.load(f)
                if isinstance(lr, dict):
                    meta["last_run"] = lr
                    # 종료 시각 기준 경과
                    end_iso = lr.get("end")
                    if end_iso:
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(end_iso.replace("Z", ""))
                            meta["last_run_age_seconds"] = int((datetime.utcnow() - dt).total_seconds())
                        except Exception:
                            pass
        except Exception as e:
            meta["last_run_error"] = str(e)
        state.trainer_meta = meta
        # Broadcast incremental trainer meta update
        try:
            await broadcast({"type": "trainer", "data": meta})
        except Exception:
            pass
        await asyncio.sleep(interval)


def _compute_features_health(session: Session, sym: str) -> Dict[str, Any]:
    from datetime import datetime, timezone, timedelta
    out: Dict[str, Any] = {}
    now = datetime.now(tz=timezone.utc)
    latest = session.exec(
        select(Candle)
        .where((Candle.symbol == sym) & (Candle.exchange_type == settings.EXCHANGE_TYPE) & (Candle.interval == settings.INTERVAL))  # type: ignore[attr-defined]
        .order_by(Candle.open_time.desc())  # type: ignore[attr-defined]
        .limit(1)
    ).first()
    if latest:
        out["latest_open_time"] = latest.open_time.isoformat()
        out["latest_close_time"] = getattr(latest, "close_time", None).isoformat() if getattr(latest, "close_time", None) else None  # type: ignore[union-attr]
        out["data_fresh_seconds"] = max(0.0, (now - latest.open_time.replace(tzinfo=latest.open_time.tzinfo or timezone.utc)).total_seconds())
        base_fields = {
            "id","symbol","exchange_type","interval","open_time","close_time","open","high","low","close","volume","trades"
        }
        feature_names = [f for f in _get_model_fields(Candle) if f not in base_fields]
        nulls = sum(1 for f in feature_names if getattr(latest, f) is None)
        out["latest_feature_null_ratio"] = float(nulls / len(feature_names)) if feature_names else 0.0
    else:
        out["data_fresh_seconds"] = None

    # Missing minutes in last 24h (only for 1m)
    lookback_start = now - timedelta(hours=24)
    missing_ranges = (
        find_missing_ranges(session, sym, settings.EXCHANGE_TYPE, settings.INTERVAL, lookback_start, now)
        if settings.INTERVAL == "1m"
        else []
    )
    def _range_minutes(a, b):
        return int((b - a).total_seconds() // 60) + 1 if b >= a else 0
    out["missing_minutes_24h"] = int(sum(_range_minutes(a, b) for a, b in missing_ranges))
    # Resampled TF presence
    for tf in ("5m", "15m"):
        r = session.exec(
            select(Candle.open_time)
            .where((Candle.symbol == sym) & (Candle.exchange_type == settings.EXCHANGE_TYPE) & (Candle.interval == tf))  # type: ignore[attr-defined]
            .order_by(Candle.open_time.desc())  # type: ignore[attr-defined]
            .limit(1)
        ).first()
        out[f"{tf}_latest_open_time"] = r.isoformat() if r else None
    return out


async def predictor_loop(sym: str) -> None:
    import math
    predictor = RealTimePredictor(symbol=sym, interval=settings.INTERVAL)
    interval = max(5, settings.PREDICT_INTERVAL_SECONDS)
    try:
        logging.info("predictor_loop started for %s interval=%ss", sym, interval)
    except Exception:
        pass
    while True:
        try:
            with Session(engine) as session:
                row = session.exec(
                    select(Candle)
                    .where((Candle.symbol == sym) & (Candle.exchange_type == settings.EXCHANGE_TYPE) & (Candle.interval == settings.INTERVAL))  # type: ignore[attr-defined]
                    .order_by(Candle.open_time.desc())  # type: ignore[attr-defined]
                    .limit(1)
                ).first()
                if not row:
                    logging.info("predictor_loop %s no row yet", sym)
                if row:
                    live_price = state.collector.live_prices.get(sym)
                    price_source = "live" if live_price is not None else "closed"
                    nc = predictor.predict_from_row(row, price_override=live_price, price_source=price_source)
                    state.latest_nowcast[sym] = nc.to_dict()
                    # Update features snapshot
                    feat = _compute_features_health(session, sym)
                    state.features_snapshot[sym] = feat
                    # Broadcast compact update for this symbol
                    await broadcast({
                        "type": "nowcast",
                        "symbol": sym,
                        "data": state.latest_nowcast[sym],
                    })
                    try:
                        base_probs = state.latest_nowcast[sym].get('base_probs')
                        base_info = state.latest_nowcast[sym].get('base_info')
                        st = (nc.stacking or {})
                        used = st.get('used_models')
                        method = st.get('method')
                        th_src = st.get('threshold_source') or st.get('threshold_source_final')
                        # Extract seq model readiness 스냅샷
                        seq_status = {}
                        for mdl in ('lstm','tf'):
                            if isinstance(base_info, dict) and mdl in base_info:
                                seq_status[mdl] = base_info.get(mdl)
                        logging.info(
                            "nowcast %s p=%.6f bottom=%.4f stack=%.4f used=%s method=%s th_src=%s base=%s seq=%s",
                            sym,
                            nc.price,
                            nc.bottom_score,
                            float(st.get('prob') or 0.0),
                            used,
                            method,
                            th_src,
                            base_probs,
                            seq_status,
                        )
                    except Exception:
                        pass
        except Exception as e:
            logging.exception("predictor_loop error [%s]: %s", sym, e)
        await asyncio.sleep(interval)


def _trades_snapshot(session: Session) -> Any:
    from datetime import datetime
    from sqlalchemy import text
    rows = session.exec(
        select(Trade)
        .where(Trade.symbol.in_(settings.SYMBOLS))  # type: ignore[attr-defined]
        .order_by(text('created_at DESC'))
        .limit(200)
    ).all()
    out = []
    # cooldown is global from TradeManager if present
    tm = getattr(state, 'trade_manager', None)
    cooldown = int(getattr(tm, 'add_cooldown_seconds', 0) or 0)
    for t in rows:
        fills = session.exec(
            select(TradeFill).where(TradeFill.trade_id == t.id).order_by(TradeFill.timestamp.asc())  # type: ignore[attr-defined]
        ).all()
        last_fill_at = fills[-1].timestamp if fills else t.created_at
        next_add_in_seconds = None
        if t.status == 'open' and cooldown > 0 and last_fill_at is not None:
            elapsed = (datetime.utcnow() - last_fill_at).total_seconds()
            next_add_in_seconds = int(max(0, cooldown - elapsed))
        out.append({
            "id": t.id,
            "symbol": t.symbol,
            "status": t.status,
            "side": t.side,
            "leverage": t.leverage,
            "created_at": t.created_at.isoformat(),
            "closed_at": t.closed_at.isoformat() if t.closed_at else None,
            "entry_price": t.entry_price,
            "avg_price": t.avg_price,
            "quantity": t.quantity,
            "adds_done": t.adds_done,
            "take_profit_pct": t.take_profit_pct,
            "stop_loss_pct": t.stop_loss_pct,
            "last_price": t.last_price,
            "pnl_pct_snapshot": t.pnl_pct_snapshot,
            "cooldown_seconds": cooldown,
            "last_fill_at": last_fill_at.isoformat() if last_fill_at else None,
            "fills": [
                {"t": f.timestamp.isoformat(), "price": f.price, "qty": f.quantity} for f in fills
            ],
        })
    return out


async def trades_broadcast_loop() -> None:
    while True:
        try:
            with Session(engine) as session:
                data = _trades_snapshot(session)
            await broadcast({"type": "trades", "data": data})
        except Exception as e:
            logging.exception("trades loop error: %s", e)
        await asyncio.sleep(45)


def _compute_entry_metrics(session: Session) -> Dict[str, Any]:
    """Compute rolling win-rate over last N closed trades (overall and per-symbol)."""
    from sqlalchemy import text
    N = int(getattr(settings, 'ENTRY_WINRATE_WINDOW', 30))
    min_samples = int(getattr(settings, 'ENTRY_WINRATE_MIN_SAMPLES', 10))
    target = float(getattr(settings, 'ENTRY_WINRATE_TARGET', 0.75))
    rows = session.exec(
        select(Trade)
        .where(Trade.status == 'closed')  # type: ignore[attr-defined]
        .order_by(text('closed_at DESC'))
        .limit(max(N * 5, N))
    ).all()
    # Group by symbol and overall take last N
    overall: list[Trade] = []
    by_symbol: Dict[str, list[Trade]] = {s: [] for s in settings.SYMBOLS}
    for t in rows:
        if len(overall) < N:
            overall.append(t)
        sym = str(getattr(t, 'symbol', '') or '').lower()
        if sym in by_symbol and len(by_symbol[sym]) < N:
            by_symbol[sym].append(t)
    def _wr(lst: list[Trade]) -> Dict[str, Any]:
        n = len(lst)
        if n == 0:
            return {"samples": 0, "win_rate": None}
        wins = 0
        for t in lst:
            try:
                p = float(getattr(t, 'pnl_pct_snapshot', 0.0) or 0.0)
                if p >= 0:
                    wins += 1
            except Exception:
                pass
        return {"samples": n, "win_rate": (wins / n) if n else None}
    overall_stats = _wr(overall)
    by_sym_stats: Dict[str, Any] = {sym: _wr(by_symbol.get(sym, [])) for sym in settings.SYMBOLS}
    # Determine effective thresholds (env/sidecar/dynamic)
    th_env = float(getattr(settings, 'ENTRY_META_THRESHOLD', 0.0))
    th_side = None
    th_dyn = None
    try:
        if registry.stacking is not None:
            th_side = getattr(registry.stacking, 'entry_threshold_sidecar', None)
            th_dyn = getattr(registry.stacking, 'entry_threshold_dynamic', None)
    except Exception:
        pass
    out = {
        "window": N,
        "min_samples": min_samples,
        "target": target,
        "overall": overall_stats,
        "by_symbol": by_sym_stats,
        "threshold": {
            "env": th_env,
            "sidecar": (float(th_side) if isinstance(th_side, (int, float)) else None),
            "dynamic": (float(th_dyn) if isinstance(th_dyn, (int, float)) else None),
        },
    }
    return out


async def entry_metrics_loop() -> None:
    """Periodically compute win-rate metrics and adapt dynamic entry threshold."""
    interval = 60
    step = float(getattr(settings, 'ENTRY_META_DYNAMIC_STEP', 0.01))
    th_min = float(getattr(settings, 'ENTRY_META_DYNAMIC_MIN', 0.85))
    th_max = float(getattr(settings, 'ENTRY_META_DYNAMIC_MAX', 0.98))
    base_target = float(getattr(settings, 'ENTRY_WINRATE_TARGET', 0.75))
    min_samples = int(getattr(settings, 'ENTRY_WINRATE_MIN_SAMPLES', 10))
    onboard_samples = int(getattr(settings, 'ENTRY_META_ONBOARD_SAMPLES', max(min_samples * 2, 20)))
    onboard_scale = float(getattr(settings, 'ENTRY_META_ONBOARD_STEP_SCALE', 0.5))
    state_path = str(getattr(settings, 'ENTRY_META_STATE_PATH', ''))
    def _ensure_parent(path: str) -> None:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    def _save_state(global_dyn: float, per_sym: Dict[str, float]) -> None:
        if not state_path:
            return
        try:
            _ensure_parent(state_path)
            payload = {
                'dynamic_global': float(global_dyn),
                'dynamic_by_symbol': {str(k): float(v) for k, v in (per_sym or {}).items()},
                'updated_at': datetime.utcnow().isoformat() + 'Z',
                'window': int(getattr(settings, 'ENTRY_WINRATE_WINDOW', 30)),
                'target': float(getattr(settings, 'ENTRY_WINRATE_TARGET', 0.75)),
            }
            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f)
        except Exception:
            pass
    # Initialize dynamic threshold from sidecar or env if not set
    def _init_dynamic_from_sources() -> float:
        base = None
        try:
            if registry.stacking is not None:
                if isinstance(getattr(registry.stacking, 'entry_threshold_dynamic', None), (int, float)):
                    return float(getattr(registry.stacking, 'entry_threshold_dynamic'))
                side = getattr(registry.stacking, 'entry_threshold_sidecar', None)
                if isinstance(side, (int, float)):
                    base = float(side)
        except Exception:
            base = None
        if base is None:
            try:
                base = float(getattr(settings, 'ENTRY_META_THRESHOLD', 0.9))
            except Exception:
                base = 0.9
        return max(th_min, min(th_max, float(base)))
    while True:
        try:
            with Session(engine) as session:
                metrics = _compute_entry_metrics(session)
            # Session-based target override (optional)
            target = float(base_target)
            try:
                if getattr(settings, 'ENTRY_SESSION_SPLIT_ENABLED', False):
                    from datetime import datetime
                    now_h = datetime.utcnow().hour
                    start_h = int(getattr(settings, 'ENTRY_SESSION_DAY_START_HOUR', 9))
                    end_h = int(getattr(settings, 'ENTRY_SESSION_DAY_END_HOUR', 21))
                    if start_h < end_h:
                        in_day = (now_h >= start_h and now_h < end_h)
                    else:
                        # overnight wrap (e.g., 21 -> 7)
                        in_day = (now_h >= start_h or now_h < end_h)
                    target = float(getattr(settings, 'ENTRY_WINRATE_TARGET_DAY', base_target)) if in_day else float(getattr(settings, 'ENTRY_WINRATE_TARGET_NIGHT', base_target))
            except Exception:
                target = float(base_target)
            # Adapt threshold if enough samples
            cur_dyn = None
            try:
                if registry.stacking is not None:
                    cur_dyn = getattr(registry.stacking, 'entry_threshold_dynamic', None)
                    if not isinstance(cur_dyn, (int, float)):
                        cur_dyn = None
            except Exception:
                cur_dyn = None
            if cur_dyn is None:
                cur_dyn = _init_dynamic_from_sources()
            wr = metrics.get('overall', {}).get('win_rate') if isinstance(metrics.get('overall'), dict) else None
            samples = metrics.get('overall', {}).get('samples') if isinstance(metrics.get('overall'), dict) else 0
            new_dyn = float(cur_dyn)
            if isinstance(wr, (int, float)) and int(samples) >= min_samples:
                step_eff = step
                if int(samples) < onboard_samples:
                    step_eff = max(1e-6, step * onboard_scale)
                if wr < target:
                    new_dyn = min(th_max, new_dyn + step_eff)
                elif wr > target + 0.02:
                    new_dyn = max(th_min, new_dyn - step_eff)
            # Set back to registry
            try:
                if registry.stacking is not None:
                    registry.stacking.entry_threshold_dynamic = float(new_dyn)
            except Exception:
                pass
            # Attach dynamic threshold into metrics
            metrics.setdefault('threshold', {})['dynamic'] = float(new_dyn)
            metrics['target'] = float(base_target)
            metrics['target_effective'] = float(target)
            # Per-symbol dynamic thresholds
            try:
                sym_dyn: Dict[str, float] = {}
                if registry.stacking is not None and isinstance(getattr(registry.stacking, 'entry_threshold_dynamic_by_symbol', None), dict):
                    sym_dyn = dict(getattr(registry.stacking, 'entry_threshold_dynamic_by_symbol'))
                for sym, st in (metrics.get('by_symbol') or {}).items():
                    try:
                        sym_wr = st.get('win_rate') if isinstance(st, dict) else None
                        sym_samp_v = st.get('samples') if isinstance(st, dict) else 0
                        try:
                            sym_samp = int(sym_samp_v) if isinstance(sym_samp_v, (int, float)) else 0
                        except Exception:
                            sym_samp = 0
                        base = sym_dyn.get(sym)
                        if not isinstance(base, (int, float)):
                            # seed from global dynamic or init
                            base = float(new_dyn)
                        nd = float(base)
                        if isinstance(sym_wr, (int, float)) and sym_samp >= min_samples:
                            step_eff = step
                            if sym_samp < onboard_samples:
                                step_eff = max(1e-6, step * onboard_scale)
                            if sym_wr < target:
                                nd = min(th_max, nd + step_eff)
                            elif sym_wr > target + 0.02:
                                nd = max(th_min, nd - step_eff)
                        sym_dyn[sym] = float(nd)
                    except Exception:
                        pass
                if registry.stacking is not None:
                    registry.stacking.entry_threshold_dynamic_by_symbol = sym_dyn
                metrics.setdefault('threshold', {})['dynamic_by_symbol'] = sym_dyn
            except Exception:
                pass
            # Persist state (best-effort)
            try:
                _save_state(float(new_dyn), dict(metrics.get('threshold', {}).get('dynamic_by_symbol') or {}))
            except Exception:
                pass
            # Append simple history (overall and per-symbol)
            try:
                from datetime import datetime
                ts = datetime.utcnow().isoformat() + 'Z'
                max_hist = 60
                # overall
                entry = {
                    'ts': ts,
                    'wr': metrics.get('overall', {}).get('win_rate') if isinstance(metrics.get('overall'), dict) else None,
                    'samples': metrics.get('overall', {}).get('samples') if isinstance(metrics.get('overall'), dict) else None,
                }
                state.entry_history_overall.append(entry)
                if len(state.entry_history_overall) > max_hist:
                    state.entry_history_overall = state.entry_history_overall[-max_hist:]
                # by symbol
                for sym, st in (metrics.get('by_symbol') or {}).items():
                    lst = state.entry_history_by_symbol.setdefault(sym, [])
                    lst.append({'ts': ts, 'wr': (st or {}).get('win_rate'), 'samples': (st or {}).get('samples')})
                    if len(lst) > max_hist:
                        state.entry_history_by_symbol[sym] = lst[-max_hist:]
                metrics['history_overall'] = state.entry_history_overall
                metrics['history_by_symbol'] = state.entry_history_by_symbol
            except Exception:
                pass
            state.entry_metrics = metrics
            await broadcast({"type": "entry_metrics", "data": metrics})
        except Exception as e:
            logging.debug("entry_metrics loop skipped: %s", e)
        await asyncio.sleep(interval)


async def training_status_broadcast_loop() -> None:
    """Periodic broadcast of scheduler training/meta-retrain status."""
    interval = 60
    while True:
        try:
            if _scheduler_mod is not None:
                meta = _scheduler_mod.get_last_retrain_meta()
                await broadcast({"type": "training_status", "data": meta})
        except Exception as e:
            logging.debug("training_status loop skipped: %s", e)
        await asyncio.sleep(interval)


def _admin_token_ok(payload: Dict[str, Any]) -> bool:
    token_cfg = os.getenv("WS_ADMIN_TOKEN", "").strip()
    if not token_cfg:
        return False  # disabled unless explicitly set
    if not isinstance(payload, dict):
        return False
    token = str(payload.get("token") or "").strip()
    return token == token_cfg


async def _handle_admin_command(ws: WebSocketServerProtocol, payload: Dict[str, Any]) -> None:
    action = str(payload.get("action") or "").strip()
    ok = False
    resp: Dict[str, Any] = {"type": "admin_ack", "action": action}
    try:
        if not _admin_token_ok(payload):
            resp["ok"] = False
            resp["error"] = "unauthorized"
            await ws.send(json.dumps(resp, ensure_ascii=False))
            return
        if action == "scheduler_start":
            tz_name = payload.get("tz")
            try:
                if _scheduler_mod is not None:
                    _scheduler_mod.start_scheduler(timezone_name=tz_name)
                    ok = True
            except Exception as e:
                resp["error"] = str(e)
        elif action == "scheduler_stop":
            try:
                if _scheduler_mod is not None:
                    _scheduler_mod.stop_scheduler()
                    ok = True
            except Exception as e:
                resp["error"] = str(e)
        elif action == "trigger_monthly":
            try:
                if _scheduler_mod is not None:
                    import asyncio as _aio
                    _aio.create_task(_scheduler_mod.trigger_now_background())
                    ok = True
            except Exception as e:
                resp["error"] = str(e)
        elif action == "trigger_meta":
            try:
                if _scheduler_mod is not None and hasattr(_scheduler_mod, 'run_meta_retrain_job'):
                    import asyncio as _aio
                    _aio.create_task(_scheduler_mod.run_meta_retrain_job())
                    ok = True
            except Exception as e:
                resp["error"] = str(e)
        elif action == "trigger_prob_drift":
            try:
                if _scheduler_mod is not None and hasattr(_scheduler_mod, 'trigger_prob_drift_retrain'):
                    _scheduler_mod.trigger_prob_drift_retrain()
                    ok = True
            except Exception as e:
                resp["error"] = str(e)
        elif action == "reload_models":
            try:
                registry.load_from_settings()
                ok = True
                resp["registry"] = registry.status()
            except Exception as e:
                resp["error"] = str(e)
        elif action == "get_status":
            try:
                data: Dict[str, Any] = {"trainer": state.trainer_meta}
                if _scheduler_mod is not None:
                    data["training_status"] = _scheduler_mod.get_last_retrain_meta()
                try:
                    data["registry"] = registry.status()
                except Exception:
                    pass
                if state.entry_metrics:
                    data["entry_metrics"] = state.entry_metrics
                resp["data"] = data
                ok = True
            except Exception as e:
                resp["error"] = str(e)
        else:
            resp["error"] = "unknown_action"
        resp["ok"] = ok
    except Exception as e:
        resp["ok"] = False
        resp["error"] = str(e)
    try:
        await ws.send(json.dumps(resp, ensure_ascii=False))
    except Exception:
        pass


async def ws_handler(websocket: WebSocketServerProtocol) -> None:
    # Register
    state.clients.add(websocket)
    try:
        # Initial snapshot
        # Attach stacking meta from registry for convenience under _stacking_meta
        snapshot = {**state.latest_nowcast}
        try:
            status = registry.status()
            stk = status.get('stacking') if isinstance(status, dict) else None
            if stk and stk.get('ready'):
                snapshot['_stacking_meta'] = {
                    'method': stk.get('method'),
                    'method_meta': stk.get('method_meta'),
                    'method_override': stk.get('method_override'),
                    'threshold': stk.get('threshold'),
                    'threshold_source': stk.get('threshold_source'),
                    'used_models': stk.get('models'),
                    'entry': stk.get('entry_meta'),
                }
            if state.trainer_meta:
                snapshot['_trainer_meta'] = state.trainer_meta
            if state.entry_metrics:
                snapshot['_entry_meta'] = state.entry_metrics
        except Exception:
            pass
        # Diagnostic log of snapshot sizes
        trades_list = []  # ensure defined even if snapshot retrieval fails
        try:
            with Session(engine) as _s:
                trades_list = _trades_snapshot(_s)
                trades_count = len(trades_list)
            logging.info(
                "WS client connected; sending snapshot nowcast_syms=%d features_syms=%d trades=%d", 
                len(snapshot), len(state.features_snapshot), trades_count
            )
        except Exception:
            pass
        # Attempt send; handle early disconnect quietly
        try:
            await websocket.send(json.dumps({
                "type": "snapshot",
                "nowcast": snapshot,
                "features": state.features_snapshot,
                "trades": trades_list,
            }, ensure_ascii=False, default=str))
        except Exception as e:
            try:
                logging.debug("WS initial snapshot send failed (client closed early): %s", e)
            except Exception:
                pass
            return
        # Handle optional admin messages from client; keep loop quiet on disconnects
        while not websocket.closed:
            try:
                # Use a short timeout to remain responsive; rely on built-in ping/pong
                msg = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                try:
                    obj = json.loads(msg)
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("type") in {"admin", "cmd"}:
                    await _handle_admin_command(websocket, obj)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break
    finally:
        try:
            state.clients.discard(websocket)
        except Exception:
            pass


async def startup() -> None:
    logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL, logging.INFO))
    init_db()
    light_mode = os.getenv("LIGHT_MODE", "0") in {"1","true","True"}
    # Setup TradeManager on state (reuse config from settings)
    try:
        from .trade_manager import TradeManager
        if not light_mode:
            state.trade_manager = TradeManager(lambda: Session(engine), leverage=10, add_cooldown_seconds=settings.ADD_COOLDOWN_SECONDS)
    except Exception:
        state.trade_manager = None
    # Load model registry (non-fatal)
    if not light_mode:
        try:
            registry.load_from_settings()
            logging.info("Model registry: %s", registry.status())
        except Exception as e:
            logging.exception("registry load failed: %s", e)
        # Load persisted dynamic thresholds if present
        try:
            spath = str(getattr(settings, 'ENTRY_META_STATE_PATH', ''))
            if spath and os.path.exists(spath) and registry.stacking is not None:
                with open(spath, 'r', encoding='utf-8') as f:
                    st = json.load(f)
                gv = st.get('dynamic_global')
                if isinstance(gv, (int, float)):
                    registry.stacking.entry_threshold_dynamic = float(gv)
                mp = st.get('dynamic_by_symbol')
                if isinstance(mp, dict):
                    # coerce values to float and lowercase keys
                    reg: Dict[str, float] = {}
                    for k, v in mp.items():
                        try:
                            if isinstance(v, (int, float)):
                                reg[str(k).lower()] = float(v)
                        except Exception:
                            pass
                    registry.stacking.entry_threshold_dynamic_by_symbol = reg
                logging.info("Loaded entry-meta dynamic thresholds from %s (global=%s, symbols=%d)", spath, str(registry.stacking.entry_threshold_dynamic), len(getattr(registry.stacking, 'entry_threshold_dynamic_by_symbol', {}) or {}))
        except Exception as _e:
            logging.warning("Entry-meta dynamic state load skipped: %s", _e)
    # Optional gap fill before starting realtime collector
    if settings.GAP_FILL_ENABLED:
        try:
            total_inserted = 0
            from .gap_fill import fill_gaps
            for sym in settings.SYMBOLS:
                inserted = fill_gaps(symbol=sym)
                total_inserted += inserted
                logging.info("Startup gap fill inserted %d candles for %s", inserted, sym)
            logging.info("Startup gap fill total inserted %d", total_inserted)
        except Exception as e:
            logging.exception("Gap fill failed: %s", e)

    # Feature calculator seed (similar to FastAPI startup)
    try:
        with Session(engine) as session:
            for sym in settings.SYMBOLS:
                fc = state.collector.features_by_symbol.get(sym)
                if fc is None:
                    from .features import FeatureCalculator
                    fc = FeatureCalculator()
                    state.collector.features_by_symbol[sym] = fc
                snap = load_feature_state(session, sym, settings.EXCHANGE_TYPE, settings.INTERVAL)
                if snap:
                    from .features import FeatureCalculator
                    state.collector.features_by_symbol[sym] = FeatureCalculator.from_snapshot(snap)
                else:
                    seed_feature_calculator(
                        fc,
                        session,
                        sym,
                        settings.EXCHANGE_TYPE,
                        settings.INTERVAL,
                        lookback_minutes=400,
                    )
            # Seed sequence buffers with recent historical candles (up to SEQ_LEN)
            try:
                from sqlalchemy import desc
                from .models import Candle as _C
                from .seq_buffer import get_buffer, extract_vector_from_candle
                for sym in settings.SYMBOLS:
                    rows = session.exec(
                        select(_C)
                        .where((_C.symbol == sym) & (_C.exchange_type == settings.EXCHANGE_TYPE) & (_C.interval == settings.INTERVAL))
                        .order_by(_C.open_time.desc())  # type: ignore[attr-defined]
                        .limit(settings.SEQ_LEN)
                    ).all()
                    if rows:
                        buf = get_buffer(sym)
                        # Prepend oldest first to preserve chronological order
                        for row in reversed(rows):
                            try:
                                buf.append(extract_vector_from_candle(row))
                            except Exception:
                                pass
                        logging.info("Seeded sequence buffer for %s len=%d", sym, len(buf))
            except Exception as _se:
                logging.warning("Sequence buffer seed failed: %s", _se)
    except Exception as e:
        logging.exception("Seed feature calculators failed: %s", e)

    # Start collector and periodic predictor tasks
    if not settings.DISABLE_BACKGROUND_LOOPS:
        await state.collector.start()
    if not light_mode and settings.PREDICT_ENABLED:
        for sym in settings.SYMBOLS:
            asyncio.create_task(predictor_loop(sym))
            try:
                logging.info("Scheduled predictor loop for %s", sym)
            except Exception:
                pass
        # Perform an immediate one-off prediction per symbol to populate initial snapshot
        try:
            from .predictor import RealTimePredictor as _RTP
            with Session(engine) as _session:
                for sym in settings.SYMBOLS:
                    # Use text order_by to avoid SQLModel attribute resolution issues in this context
                    from sqlalchemy import text as _text
                    row = _session.exec(
                        select(Candle)
                        .where((Candle.symbol == sym) & (Candle.exchange_type == settings.EXCHANGE_TYPE) & (Candle.interval == settings.INTERVAL))
                        .order_by(_text('open_time DESC'))
                        .limit(1)
                    ).first()
                    if not row:
                        continue
                    live_price = state.collector.live_prices.get(sym)
                    price_source = "live" if live_price is not None else "closed"
                    _pred = _RTP(symbol=sym, interval=settings.INTERVAL)
                    nc = _pred.predict_from_row(row, price_override=live_price, price_source=price_source)
                    state.latest_nowcast[sym] = nc.to_dict()
                    state.features_snapshot[sym] = _compute_features_health(_session, sym)
                logging.info("Initial nowcast snapshot populated for %d symbols", len(state.latest_nowcast))
        except Exception as _e:
            logging.warning("Initial prediction population failed: %s", _e)
    # Start periodic resampler for 5m/15m
    async def _run_resampler():
        while True:
            try:
                from .resampler import resample_incremental
                with Session(engine) as session:
                    total_written = 0
                    for sym in settings.SYMBOLS:
                        for tf in ("5m", "15m"):
                            written = resample_incremental(session, sym, settings.EXCHANGE_TYPE, tf)
                            total_written += written
                    if total_written:
                        logging.info("Incremental resampler tick wrote %d candles", total_written)
            except Exception as e:
                logging.exception("Resampler error: %s", e)
            await asyncio.sleep(getattr(settings, 'RESAMPLER_INTERVAL_SECONDS', 55))
    if not settings.DISABLE_BACKGROUND_LOOPS:
        asyncio.create_task(_run_resampler())
    # Trades snapshot loop
    asyncio.create_task(trades_broadcast_loop())
    # Training status loop
    asyncio.create_task(training_status_broadcast_loop())
    # Entry metrics + dynamic threshold loop
    asyncio.create_task(entry_metrics_loop())


async def main() -> None:
    await startup()
    # 모델 아티팩트 변경 감시 태스크 시작
    try:
        if settings.MODEL_WATCH_ENABLED:
            asyncio.create_task(model_artifact_watcher())
        # Trainer meta watcher always enabled (light overhead)
        asyncio.create_task(trainer_meta_watcher())
        # Auto-start scheduler if meta retrain or retrain features are enabled
        try:
            if _scheduler_mod is not None and (
                getattr(settings, 'META_RETRAIN_ENABLED', False)
                or getattr(settings, 'RETRAIN_ON_TRADE_CLOSE', False)
                or getattr(settings, 'PROB_DRIFT_ENABLED', False)
            ):
                tz = os.getenv('SCHEDULER_TZ')  # optional, e.g. 'UTC' or 'Asia/Seoul'
                _scheduler_mod.start_scheduler(timezone_name=tz)
                logging.info("Scheduler auto-started (tz=%s)", tz)
        except Exception as _se:
            logging.warning("Scheduler auto-start failed: %s", _se)
    except Exception:
        pass
    # Attempt to bind the websocket server; handle Windows "address already in use" (errno 10048) gracefully.
    # 환경변수 기반 ping 설정 (기본값 완화)
    ping_interval = int(os.getenv("WS_PING_INTERVAL", "30"))
    ping_timeout = int(os.getenv("WS_PING_TIMEOUT", "45"))
    try:
        server = await serve(
            ws_handler,
            WS_HOST,
            WS_PORT,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
        )
    except OSError as e:
        if getattr(e, 'errno', None) == 10048:
            logging.error(
                "WS port %d already in use. Another instance may be running. "
                "Kill the existing process or set WS_PORT to a free port.", WS_PORT
            )
            # Stop background loops we just started to avoid orphan tasks
            if not settings.DISABLE_BACKGROUND_LOOPS:
                try:
                    await state.collector.stop()
                except Exception:
                    pass
            return
        raise

    # Graceful shutdown signal handlers
    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, shutdown_event.set)
        logging.info("Registered SIGTERM/SIGINT handlers for graceful shutdown")
    except Exception:
        logging.warning("Signal handler registration failed; continuing without graceful shutdown handlers")

    async def shutdown_watcher():
        await shutdown_event.wait()
        logging.info("Shutdown signal received; broadcasting server_shutdown and closing clients")
        try:
            await broadcast({"type": "server_shutdown", "reason": "signal"})
        except Exception:
            pass
        # Close websocket clients politely
        for ws in list(state.clients):
            try:
                await ws.close(code=1001, reason="shutdown")
            except Exception:
                pass
        # Stop collector
        if not settings.DISABLE_BACKGROUND_LOOPS:
            try:
                await state.collector.stop()
            except Exception:
                pass
    asyncio.create_task(shutdown_watcher())

    async with server:
        logging.info(
            "WS server started at ws://%s:%d (ping_interval=%d ping_timeout=%d)",
            WS_HOST,
            WS_PORT,
            ping_interval,
            ping_timeout,
        )
        try:
            while not shutdown_event.is_set():
                await asyncio.sleep(3600)
        finally:
            # Persist feature calculator snapshots on shutdown
            try:
                with Session(engine) as session:
                    for sym, fc in state.collector.features_by_symbol.items():
                        save_feature_state(session, sym, settings.EXCHANGE_TYPE, settings.INTERVAL, fc.snapshot())
                    session.commit()
            except Exception:
                pass
            # Persist entry-meta thresholds on shutdown (best-effort)
            try:
                if getattr(registry, 'stacking', None) is not None:
                    g = getattr(registry.stacking, 'entry_threshold_dynamic', None)
                    m = getattr(registry.stacking, 'entry_threshold_dynamic_by_symbol', None)
                    if isinstance(g, (int, float)) and isinstance(m, dict):
                        pth = str(getattr(settings, 'ENTRY_META_STATE_PATH', ''))
                        if pth:
                            Path(pth).parent.mkdir(parents=True, exist_ok=True)
                            with open(pth, 'w', encoding='utf-8') as f:
                                json.dump({
                                    'dynamic_global': float(g),
                                    'dynamic_by_symbol': {str(k): float(v) for k, v in (m or {}).items()},
                                    'updated_at': datetime.utcnow().isoformat() + 'Z',
                                    'window': int(getattr(settings, 'ENTRY_WINRATE_WINDOW', 30)),
                                    'target': float(getattr(settings, 'ENTRY_WINRATE_TARGET', 0.75)),
                                }, f)
            except Exception:
                pass
            if not settings.DISABLE_BACKGROUND_LOOPS:
                try:
                    await state.collector.stop()
                except Exception:
                    pass


if __name__ == "__main__":
    asyncio.run(main())
