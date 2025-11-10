from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging

from .db import init_db, engine
from .collector import BinanceCollector
from .core.config import settings
from .gap_fill import fill_gaps, seed_feature_calculator, find_missing_ranges
from sqlmodel import Session, select
from .models import Candle
from .resampler import resample_from_1m, resample_incremental
from .snapshots import save_feature_state
from .logging_config import init_logging
from .snapshots import load_feature_state

init_logging()
app = FastAPI(title="Models API", version="0.1.0")
collector = BinanceCollector()
_resampler_task = None

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


@app.get("/health/features")
async def health_features(symbol: str | None = None):
    """Return data freshness and feature completeness metrics for the primary stream.
    Lightweight to avoid heavy table scans.
    """
    from datetime import datetime, timezone, timedelta

    sym = (symbol or settings.SYMBOL).lower()
    ex = settings.EXCHANGE_TYPE
    itv = settings.INTERVAL
    now = datetime.now(tz=timezone.utc)
    result = {
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
                select(Candle).where((Candle.symbol==sym)&(Candle.exchange_type==ex)&(Candle.interval==itv)).order_by(Candle.open_time.desc()).limit(1)
            ).first()
            if latest:
                result["latest_open_time"] = latest.open_time.isoformat()
                result["latest_close_time"] = latest.close_time.isoformat() if latest.close_time else None
                result["data_fresh_seconds"] = max(0, (now - latest.open_time.replace(tzinfo=latest.open_time.tzinfo or timezone.utc)).total_seconds())
                feature_names = [f for f in Candle.__fields__.keys() if f not in base_fields]
                nulls = sum(1 for f in feature_names if getattr(latest, f) is None)
                result["latest_feature_null_ratio"] = nulls/len(feature_names) if feature_names else 0.0
            else:
                result.update({"latest_open_time": None, "latest_close_time": None, "data_fresh_seconds": None, "latest_feature_null_ratio": None})

            # Missing minutes in last 24h (only for 1m)
            lookback_start = now - timedelta(hours=24)
            missing_ranges = find_missing_ranges(session, sym, ex, itv, lookback_start, now) if itv == "1m" else []
            def _range_minutes(a,b):
                return int((b - a).total_seconds()//60) + 1 if b>=a else 0
            missing_minutes_24h = sum(_range_minutes(a,b) for a,b in missing_ranges)
            result["missing_minutes_24h"] = missing_minutes_24h
            result["missing_ranges_24h"] = [(a.isoformat(), b.isoformat()) for a,b in missing_ranges[:5]]  # cap preview

            # Resampled TF presence
            for tf in ("5m","15m"):
                r = session.exec(
                    select(Candle.open_time).where((Candle.symbol==sym)&(Candle.exchange_type==ex)&(Candle.interval==tf)).order_by(Candle.open_time.desc()).limit(1)
                ).first()
                result[f"{tf}_latest_open_time"] = r.isoformat() if r else None
    except Exception as e:
        logging.exception("health_features error: %s", e)
        result["error"] = str(e)
    return result


@app.on_event("startup")
async def on_startup():
    init_db()
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
    except Exception as e:
        logging.exception("Seeding feature calculators failed: %s", e)
    # Start collector background process
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
            await asyncio.sleep(55)
    global _resampler_task
    _resampler_task = asyncio.create_task(_run_resampler())


@app.on_event("shutdown")
async def on_shutdown():
    await collector.stop()
    # Cancel resampler task
    global _resampler_task
    if _resampler_task:
        _resampler_task.cancel()
    # Persist latest feature calculator snapshots
    try:
        with Session(engine) as session:
            for sym, fc in collector.features_by_symbol.items():
                save_feature_state(session, sym, settings.EXCHANGE_TYPE, settings.INTERVAL, fc.snapshot())
            session.commit()
        logging.info("Saved feature state snapshots for %d symbols on shutdown", len(collector.features_by_symbol))
    except Exception as e:
        logging.exception("Failed to save snapshots on shutdown: %s", e)

