from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
from typing import Any, Dict

from .db import init_db, engine, get_session
from .collector import BinanceCollector
from .core.config import settings
from .gap_fill import fill_gaps, seed_feature_calculator, find_missing_ranges
from sqlmodel import Session, select
from .models import Candle
from .resampler import resample_from_1m, resample_incremental
from .snapshots import save_feature_state
from .logging_config import init_logging
from .snapshots import load_feature_state
from .predictor import RealTimePredictor
from .model_adapters import registry
from .core.config import settings as _settings
from .trade_manager import TradeManager
from .scheduler import start_scheduler, stop_scheduler, trigger_now_background, get_last_retrain_meta, trigger_prob_drift_retrain
from .drift import init_drift_monitor

init_logging()
app = FastAPI(title="Models API", version="0.1.0")
collector = BinanceCollector()
_resampler_task = None
_predict_task = None
_predictor: RealTimePredictor | None = None
_predictor_tasks_multi: dict[str, asyncio.Task] = {}
app.state.trade_manager = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "Hello from FastAPI backend"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/errors")
async def health_errors():
    """Return brief error/resilience related settings for observability."""
    return {
        "error_log_min_interval_seconds": getattr(_settings, 'ERROR_LOG_MIN_INTERVAL_SECONDS', 60),
        "feature_snapshot_every": getattr(_settings, 'FEATURE_SNAPSHOT_EVERY', 50),
        "resampler_interval_seconds": getattr(_settings, 'RESAMPLER_INTERVAL_SECONDS', 55),
        "alert_enabled": bool(getattr(_settings, 'ALERT_WEBHOOK_URL', None)),
        "alert_cooldown_seconds": getattr(_settings, 'ALERT_COOLDOWN_SECONDS', 300),
    }


@app.get("/health/models")
async def health_models():
    """Report model registry readiness for base and stacking."""
    from .model_adapters import registry
    try:
        # Ensure loaded at least once
        if not registry.adapters and not registry.stacking:
            registry.load_from_settings()
        status = registry.status()
        try:
            meta = get_last_retrain_meta()
            status.update({'retrain_meta': meta})
        except Exception:
            pass
        # Attach probability drift monitor state if present
        try:
            drift_state = getattr(app.state, 'prob_drift_state', None)
            if drift_state:
                status['prob_drift'] = drift_state
        except Exception:
            pass
        return status
    except Exception as e:
        logging.exception("health_models error: %s", e)
        return {"error": str(e)}


@app.get("/health/trading")
async def health_trading():
    """Expose trading-related settings and quick status.

    Returns leverage and cooldown from TradeManager, TP/SL/max_adds defaults from Trade model,
    and a compact summary of open trades including next add timing.
    """
    from datetime import datetime
    from .models import Trade, TradeFill
    tm: TradeManager | None = getattr(app.state, 'trade_manager', None)
    # Use an ephemeral Trade instance to read model defaults
    probe = Trade(symbol=settings.SYMBOL, exchange_type=settings.EXCHANGE_TYPE, interval=settings.INTERVAL, entry_price=0.0, avg_price=0.0)
    info: Dict[str, Any] = {
        "exchange_type": settings.EXCHANGE_TYPE,
        "interval": settings.INTERVAL,
        "symbols": settings.SYMBOLS,
        "leverage": getattr(tm, 'leverage', None),
        "add_cooldown_seconds": int(getattr(tm, 'add_cooldown_seconds', 0) or 0),
        "defaults": {
            "take_profit_pct": getattr(_settings, 'TAKE_PROFIT_PCT', probe.take_profit_pct),
            "stop_loss_pct": getattr(_settings, 'STOP_LOSS_PCT', probe.stop_loss_pct),
            "max_adds": getattr(_settings, 'MAX_ADDS', probe.max_adds),
            "tp_mode": getattr(_settings, 'TP_MODE', 'fixed'),
            "tp_trigger": getattr(_settings, 'TP_TRIGGER', None),
            "tp_step": getattr(_settings, 'TP_STEP', None),
            "tp_giveback": getattr(_settings, 'TP_GIVEBACK', None),
        },
    }
    try:
        with Session(engine) as session:
            # Restrict to configured symbols
            from sqlalchemy import desc
            # Use SQLAlchemy text label for ordering to satisfy type checker
            from sqlalchemy import text
            rows = session.exec(
                select(Trade)
                .where((Trade.status == 'open') & (Trade.symbol.in_(settings.SYMBOLS)))  # type: ignore[attr-defined]
                .order_by(text('created_at DESC'))
            )
            open_trades = []
            cooldown = int(getattr(tm, 'add_cooldown_seconds', 0) or 0)
            for t in rows:
                last_ts = session.exec(
                    select(TradeFill.timestamp).where(TradeFill.trade_id == t.id).order_by(TradeFill.timestamp.desc()).limit(1)  # type: ignore[attr-defined]
                ).first() or t.created_at
                next_add_in_seconds = None
                if cooldown > 0 and last_ts is not None:
                    elapsed = (datetime.utcnow() - last_ts).total_seconds()
                    next_add_in_seconds = int(max(0, cooldown - elapsed))
                open_trades.append({
                    "id": t.id,
                    "symbol": t.symbol,
                    "avg_price": t.avg_price,
                    "quantity": t.quantity,
                    "adds_done": t.adds_done,
                    "created_at": t.created_at.isoformat(),
                    "cooldown_seconds": cooldown,
                    "last_fill_at": last_ts.isoformat() if last_ts else None,
                    "next_add_in_seconds": next_add_in_seconds,
                })
            info["open_trades_count"] = len(open_trades)
            info["open_trades"] = open_trades
    except Exception as e:
        logging.exception("health_trading error: %s", e)
        info["error"] = str(e)
    return info


@app.get("/health/features")
async def health_features(symbol: str | None = None, include_psi: int = 1):
    """Return data freshness and feature completeness metrics for the primary stream.
    Lightweight to avoid heavy table scans.
    """
    from datetime import datetime, timezone, timedelta

    sym = (symbol or settings.SYMBOL).lower()
    ex = settings.EXCHANGE_TYPE
    itv = settings.INTERVAL
    now = datetime.now(tz=timezone.utc)
    # Allow heterogeneous value types; use Dict[str, Any]
    result: Dict[str, Any] = {
        "symbol": sym,
        "exchange_type": ex,
        "interval": itv,
        "server_time_utc": now.isoformat(),
    }

    base_fields = {
        "id","symbol","exchange_type","interval","open_time","close_time","open","high","low","close","volume","trades"
    }
    try:
        with Session(engine) as session:
            latest = session.exec(
                select(Candle).where((Candle.symbol==sym)&(Candle.exchange_type==ex)&(Candle.interval==itv)).order_by(Candle.open_time.desc()).limit(1)  # type: ignore[attr-defined]
            ).first()
            if latest:
                result["latest_open_time"] = latest.open_time.isoformat()
                result["latest_close_time"] = latest.close_time.isoformat() if getattr(latest, "close_time", None) else None
                result["data_fresh_seconds"] = max(0.0, (now - latest.open_time.replace(tzinfo=latest.open_time.tzinfo or timezone.utc)).total_seconds())
                feature_names = [f for f in Candle.__fields__.keys() if f not in base_fields]
                nulls = sum(1 for f in feature_names if getattr(latest, f) is None)
                result["latest_feature_null_ratio"] = float(nulls/len(feature_names)) if feature_names else 0.0
            else:
                result["latest_open_time"] = None
                result["latest_close_time"] = None
                result["data_fresh_seconds"] = None
                result["latest_feature_null_ratio"] = None

            # Missing minutes in last 24h (only for 1m)
            lookback_start = now - timedelta(hours=24)
            missing_ranges = find_missing_ranges(session, sym, ex, itv, lookback_start, now) if itv == "1m" else []
            def _range_minutes(a,b):
                return int((b - a).total_seconds()//60) + 1 if b>=a else 0
            missing_minutes_24h = sum(_range_minutes(a,b) for a,b in missing_ranges)
            result["missing_minutes_24h"] = int(missing_minutes_24h)
            result["missing_ranges_24h"] = [(a.isoformat(), b.isoformat()) for a,b in missing_ranges[:5]]  # cap preview

            # Resampled TF presence
            for tf in ("5m","15m"):
                r = session.exec(
                    select(Candle.open_time).where((Candle.symbol==sym)&(Candle.exchange_type==ex)&(Candle.interval==tf)).order_by(Candle.open_time.desc()).limit(1)  # type: ignore[attr-defined]
                ).first()
                result[f"{tf}_latest_open_time"] = r.isoformat() if r else None

            # Feature PSI (top drifts). Lightweight window: ref=last 7d excluding 24h, cur=last 24h
            if include_psi:
                try:
                    from datetime import timedelta as _td
                    ref_end = now - _td(hours=24)
                    ref_start = ref_end - _td(days=7)
                    cur_start = now - _td(hours=24)
                    cur_end = now
                    # Select core features present in Candle
                    candle_fields = set(Candle.__fields__.keys())
                    core = [
                        "rsi_14","bb_pct_b_20_2","macd_hist","vol_z_20","williams_r_14",
                        "drawdown_from_max_20","atr_14","cci_20","run_up","run_down","obv","mfi_14","cmf_20",
                        "body_pct_of_range","vwap_20_dev",
                    ]
                    features = [f for f in core if f in candle_fields]
                    if features:
                        from .psi import compute_feature_psi
                        psi_map = compute_feature_psi(session, sym, ex, itv, features, (ref_start, ref_end), (cur_start, cur_end), bins=10)
                        # Rank and summarize
                        items = [
                            {"feature": k, "psi": float(v)} for k, v in psi_map.items() if isinstance(v, (int, float))
                        ]
                        items.sort(key=lambda d: d["psi"], reverse=True)
                        threshold = 0.2
                        top = [x for x in items if x["psi"] >= threshold][:5]
                        result["feature_drift"] = {
                            "psi_threshold": threshold,
                            "ref_range": (ref_start.isoformat(), ref_end.isoformat()),
                            "cur_range": (cur_start.isoformat(), cur_end.isoformat()),
                            "top": top,
                            "max": items[0] if items else None,
                            "count_above_threshold": int(sum(1 for x in items if x["psi"] >= threshold)),
                        }
                except Exception:
                    pass
    except Exception as e:
        logging.exception("health_features error: %s", e)
        result["error"] = str(e)
    return result


@app.on_event("startup")
async def on_startup():
    init_db()
    # Init trade manager
    # Enforce cooldown between entry and each DCA add (configurable)
    # Use direct Session factory for internal loops (not the FastAPI dependency generator)
    app.state.trade_manager = TradeManager(lambda: Session(engine), leverage=10, add_cooldown_seconds=settings.ADD_COOLDOWN_SECONDS)
    # Load model registry early (non-fatal if missing)
    try:
        registry.load_from_settings()
        logging.info("Model registry startup status: %s", registry.status())
    except Exception as e:
        logging.exception("Model registry load failed: %s", e)
    # Optional gap fill before starting realtime collector
    if settings.GAP_FILL_ENABLED:
        # Run synchronously to ensure continuity for rolling indicators
        try:
            total_inserted = 0
            for sym in settings.SYMBOLS:
                inserted = fill_gaps(symbol=sym)
                total_inserted += inserted
                logging.info("Startup gap fill inserted %d candles for %s", inserted, sym)
            logging.info("Startup gap fill total inserted %d", total_inserted)
        except Exception as e:
            logging.exception("Gap fill failed: %s", e)
    # Seed realtime feature calculator from recent DB candles so indicators are stable from the first live bar
    # Multi-symbol seeding: try restoring snapshots first; fallback to DB seeding
    try:
        with Session(engine) as session:
            for sym in settings.SYMBOLS:
                # Access calculator created lazily by collector; ensure exists
                fc = collector.features_by_symbol.get(sym)
                if fc is None:
                    from .features import FeatureCalculator
                    fc = FeatureCalculator()
                    collector.features_by_symbol[sym] = fc
                snap = load_feature_state(session, sym, settings.EXCHANGE_TYPE, settings.INTERVAL)
                if snap:
                    from .features import FeatureCalculator
                    # Replace with restored instance
                    collector.features_by_symbol[sym] = FeatureCalculator.from_snapshot(snap)
                    logging.info("Restored feature state from snapshot for %s", sym)
                else:
                    seed_feature_calculator(
                        fc,
                        session,
                        sym,
                        settings.EXCHANGE_TYPE,
                        settings.INTERVAL,
                        lookback_minutes=400
                    )
            # Prefill sequence buffers from recent DB candles so sequence models can be ready immediately
            try:
                from .seq_buffer import get_buffer, extract_vector_from_candle
                from sqlmodel import select
                for sym in settings.SYMBOLS:
                    buf = get_buffer(sym)
                    rows = session.exec(
                        select(Candle).where((Candle.symbol==sym)&(Candle.exchange_type==settings.EXCHANGE_TYPE)&(Candle.interval==settings.INTERVAL))
                        .order_by(Candle.open_time.desc()).limit(settings.SEQ_LEN)  # type: ignore[attr-defined]
                    ).all()
                    for row in reversed(rows):
                        buf.append(extract_vector_from_candle(row))
                logging.info("Prefilled sequence buffers for %d symbols (up to %d length)", len(settings.SYMBOLS), settings.SEQ_LEN)
            except Exception as e:
                logging.exception("Prefill sequence buffers failed: %s", e)
    except Exception as e:
        logging.exception("Seeding feature calculators failed: %s", e)
    # Start collector background process (unless disabled by env)
    if not settings.DISABLE_BACKGROUND_LOOPS:
        await collector.start()
    # Start periodic resampler for 5m/15m
    async def _run_resampler():
        import asyncio
        while True:
            try:
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
    global _resampler_task
    if not settings.DISABLE_BACKGROUND_LOOPS:
        _resampler_task = asyncio.create_task(_run_resampler())

    # Start periodic predictor (every 30s) using latest closed candle
    # Initialize drift monitor (singleton) if enabled
    drift_monitor = init_drift_monitor()

    async def _run_predictor_for_symbol(sym: str):
        import asyncio
        from sqlmodel import select
        predictor = RealTimePredictor(symbol=sym, interval=settings.INTERVAL)
        interval = max(5, settings.PREDICT_INTERVAL_SECONDS)
        while True:
            try:
                with Session(engine) as session:
                    latest = session.exec(
                        select(Candle).where((Candle.symbol==sym)&(Candle.exchange_type==settings.EXCHANGE_TYPE)&(Candle.interval==settings.INTERVAL))
                        .order_by(Candle.open_time.desc()).limit(1)  # type: ignore[attr-defined]
                    ).first()
                    if latest:
                        # Override price with live (in-progress) price if available for more current prediction
                        live_price = collector.live_prices.get(sym)
                        price_source = 'closed'
                        live_price = collector.live_prices.get(sym)
                        if live_price is not None:
                            price_source = 'live'
                        nc = predictor.predict_from_row(latest, price_override=live_price if live_price is not None else None, price_source=price_source)
                        prev = getattr(app.state, 'latest_nowcast', {})
                        prev[sym] = nc.to_dict()
                        app.state.latest_nowcast = prev
                        # --- Probability drift monitor update (use final stacking prob if available) ---
                        try:
                            if drift_monitor and nc.stacking and nc.stacking.get('ready'):
                                p_use = nc.stacking.get('prob_final') or nc.stacking.get('prob') or nc.stacking.get('raw_prob')
                                if isinstance(p_use, (int, float)):
                                    state = drift_monitor.update(float(p_use))
                                    app.state.prob_drift_state = state
                                    # Trigger retrain if consecutive drift criterion met
                                    if state.get('enabled') and state.get('baseline_loaded') and state.get('consecutive_drift', 0) >= state.get('consecutive_required', 999):
                                        trigger_prob_drift_retrain()
                        except Exception:
                            pass
                        # --- Compute lightweight features health for trade gating ---
                        from datetime import datetime, timezone, timedelta
                        now = datetime.now(tz=timezone.utc)
                        from typing import Any, Dict
                        features_health: Dict[str, Any] = {
                            "data_fresh_seconds": max(0.0, (now - latest.open_time.replace(tzinfo=latest.open_time.tzinfo or timezone.utc)).total_seconds())
                        }
                        # Missing minutes in last 24h (only for 1m)
                        try:
                            lookback_start = now - timedelta(hours=24)
                            from .gap_fill import find_missing_ranges
                            missing_ranges = find_missing_ranges(session, sym, settings.EXCHANGE_TYPE, settings.INTERVAL, lookback_start, now) if settings.INTERVAL == "1m" else []
                            def _range_minutes(a,b):
                                from datetime import datetime as _dt
                                return int((b - a).total_seconds()//60) + 1 if b>=a else 0
                            features_health["missing_minutes_24h"] = int(sum(_range_minutes(a,b) for a,b in missing_ranges))
                            # Resampled TF presence
                            for tf in ("5m","15m"):
                                r = session.exec(
                                    select(Candle.open_time).where((Candle.symbol==sym)&(Candle.exchange_type==settings.EXCHANGE_TYPE)&(Candle.interval==tf)).order_by(Candle.open_time.desc()).limit(1)  # type: ignore[attr-defined]
                                ).first()
                                features_health[f"{tf}_latest_open_time"] = r.isoformat() if r else None
                        except Exception:
                            pass
                        # --- Trade manager process ---
                        try:
                            tm: TradeManager = app.state.trade_manager
                            if tm:
                                tm.process(symbol=sym, interval=settings.INTERVAL, exchange_type=settings.EXCHANGE_TYPE, price=nc.price, nowcast=nc.to_dict(), features_health=features_health)
                        except Exception as e:
                            logging.exception("TradeManager process error: %s", e)
                        logging.info("[nowcast] %s %s price=%.6f (%s) bottom_score=%.3f (rsi=%.1f bb%%b=%.2f dd20=%.3f volz=%.2f)",
                                     sym, settings.INTERVAL, nc.price, nc.price_source, nc.bottom_score,
                                     nc.components.get('rsi_14', 0.0), nc.components.get('bb_pct_b_20_2', 0.0),
                                     nc.components.get('drawdown_from_max_20', 0.0), nc.components.get('vol_z_20', 0.0))
            except Exception as e:
                logging.exception("Predictor loop error [%s]: %s", sym, e)
            await asyncio.sleep(interval)

    if not settings.DISABLE_BACKGROUND_LOOPS and settings.PREDICT_ENABLED:
        for sym in settings.SYMBOLS:
            _predictor_tasks_multi[sym] = asyncio.create_task(_run_predictor_for_symbol(sym))
    # Start monthly scheduler for training
    try:
        tz_name = getattr(settings, 'SCHEDULER_TZ', None)
        start_scheduler(timezone_name=tz_name)
    except Exception as e:
        logging.exception("Failed to start scheduler: %s", e)


@app.on_event("shutdown")
async def on_shutdown():
    # Stop scheduler first to avoid overlaps
    try:
        stop_scheduler()
    except Exception:
        pass
    if not settings.DISABLE_BACKGROUND_LOOPS:
        await collector.stop()
    # Cancel resampler task
    global _resampler_task
    if _resampler_task:
        _resampler_task.cancel()
    # Cancel predictor task
    global _predictor_tasks_multi
    for sym, task in list(_predictor_tasks_multi.items()):
        task.cancel()
    _predictor_tasks_multi.clear()
    # Persist latest feature calculator snapshots
    try:
        with Session(engine) as session:
            for sym, fc in collector.features_by_symbol.items():
                save_feature_state(session, sym, settings.EXCHANGE_TYPE, settings.INTERVAL, fc.snapshot())
            session.commit()
        logging.info("Saved feature state snapshots for %d symbols on shutdown", len(collector.features_by_symbol))
    except Exception as e:
        logging.exception("Failed to save snapshots on shutdown: %s", e)


@app.get("/nowcast")
async def nowcast(symbol: str | None = None):
    """Return latest periodic prediction(s).

    - If symbol provided, returns that symbol's nowcast when available.
    - Otherwise returns a mapping for all tracked symbols.
    """
    data = getattr(app.state, 'latest_nowcast', {})
    if symbol:
        return data.get(symbol.lower())
    # Attach summarised stacking snapshot for quick status
    try:
        from .model_adapters import registry as _reg
        status = _reg.status()
        stk = status.get('stacking') if isinstance(status, dict) else None
        if stk and stk.get('ready'):
            data['_stacking_meta'] = {
                'method': stk.get('method'),
                'method_meta': stk.get('method_meta'),
                'method_override': stk.get('method_override'),
                'threshold': stk.get('threshold'),
                'threshold_source': stk.get('threshold_source'),
                'used_models': stk.get('models'),
            }
    except Exception:
        pass
    return data

@app.get("/live_price")
async def live_price(symbol: str | None = None):
    """Return current in-progress candle live price(s) captured from websocket.

    If symbol provided returns single float (or null if not yet received).
    Otherwise returns mapping of symbol->price for all tracked symbols.
    """
    prices = getattr(collector, 'live_prices', {})
    if symbol:
        return prices.get(symbol.lower())
    return prices


@app.get("/trades")
async def list_trades(limit: int = 50):
    """Return recent trades with last status and recent fills."""
    from .models import Trade, TradeFill
    from datetime import datetime
    out = []
    with Session(engine) as session:
        from sqlalchemy import desc
        from sqlalchemy import text
        rows = session.exec(
            select(Trade)
            .where(Trade.symbol.in_(settings.SYMBOLS))  # type: ignore[attr-defined]
            .order_by(text('created_at DESC'))
            .limit(max(1, min(limit, 200)))
        ).all()
        for t in rows:
            fills = session.exec(
                select(TradeFill).where(TradeFill.trade_id == t.id).order_by(TradeFill.timestamp.asc())  # type: ignore[attr-defined]
            ).all()
            # Cooldown computation
            tm: TradeManager | None = getattr(app.state, 'trade_manager', None)
            cooldown = int(getattr(tm, 'add_cooldown_seconds', 0) or 0)
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
                "max_adds": t.max_adds,
                "take_profit_pct": t.take_profit_pct,
                "stop_loss_pct": t.stop_loss_pct,
                "last_price": t.last_price,
                "pnl_pct_snapshot": t.pnl_pct_snapshot,
                "cooldown_seconds": cooldown,
                "last_fill_at": last_fill_at.isoformat() if last_fill_at else None,
                "next_add_in_seconds": next_add_in_seconds,
                "fills": [
                    {"t": f.timestamp.isoformat(), "price": f.price, "qty": f.quantity} for f in fills
                ],
            })
    return out


@app.get("/trades/open")
async def list_open_trades():
    from .models import Trade, TradeFill
    from datetime import datetime
    with Session(engine) as session:
        from sqlalchemy import desc
        from sqlalchemy import text
        rows = session.exec(
            select(Trade)
            .where((Trade.status == 'open') & (Trade.symbol.in_(settings.SYMBOLS)))  # type: ignore[attr-defined]
            .order_by(text('created_at DESC'))
        )
        tm: TradeManager | None = getattr(app.state, 'trade_manager', None)
        cooldown = int(getattr(tm, 'add_cooldown_seconds', 0) or 0)
        out = []
        for t in rows:
            last_ts = session.exec(
                select(TradeFill.timestamp).where(TradeFill.trade_id == t.id).order_by(TradeFill.timestamp.desc()).limit(1)  # type: ignore[attr-defined]
            ).first() or t.created_at
            next_add_in_seconds = None
            if cooldown > 0 and last_ts is not None:
                elapsed = (datetime.utcnow() - last_ts).total_seconds()
                next_add_in_seconds = int(max(0, cooldown - elapsed))
            out.append({
                "id": t.id,
                "symbol": t.symbol,
                "avg_price": t.avg_price,
                "quantity": t.quantity,
                "adds_done": t.adds_done,
                "created_at": t.created_at.isoformat(),
                "cooldown_seconds": cooldown,
                "last_fill_at": last_ts.isoformat() if last_ts else None,
                "next_add_in_seconds": next_add_in_seconds,
            })
        return out


    @app.get("/admin/reload_models")
    async def admin_reload_models():
        """Reload model artifacts from settings paths (base + stacking)."""
        try:
            registry.load_from_settings()
            return {"status": "ok", "registry": registry.status()}
        except Exception as e:
            logging.exception("admin_reload_models error: %s", e)
            return {"error": str(e)}


    @app.post("/admin/trigger_monthly_training")
    async def admin_trigger_monthly_training():
        """Manually trigger the monthly 3-model training sequence now (async)."""
        try:
            asyncio.create_task(trigger_now_background())
            return {"status": "accepted"}
        except Exception as e:
            logging.exception("admin_trigger_monthly_training error: %s", e)
            return {"error": str(e)}

