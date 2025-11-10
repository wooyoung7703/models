import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from sqlmodel import Session, select

from .core.config import settings
from .db import engine, init_db
from .features import FeatureCalculator
from .models import Candle
from .storage import bulk_upsert_candles
from .backfill import fetch_klines  # reuse REST fetch

log = logging.getLogger(__name__)


def _utc(ms: int) -> datetime:
    # store naive UTC in DB to keep comparisons/unique constraints consistent
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).replace(tzinfo=None)


def _ms(dt: datetime) -> int:
    # Treat naive datetimes as UTC for epoch conversion
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ensure_utc(dt: datetime) -> datetime:
    """Return naive UTC datetime for uniformity (DB stores naive UTC)."""
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _upsert(session: Session, candle: Candle) -> None:
    stmt = select(Candle).where(
        (Candle.symbol == candle.symbol)
        & (Candle.exchange_type == candle.exchange_type)
        & (Candle.interval == candle.interval)
        & (Candle.open_time == candle.open_time)
    )
    existing = session.exec(stmt).first()
    if existing:
        for field, value in candle.model_dump().items():  # model_dump preferred over deprecated dict
            if field == "id":
                continue
            setattr(existing, field, value)
        session.add(existing)
    else:
        session.add(candle)


def seed_feature_calculator(fc: FeatureCalculator, session: Session, symbol: str, exchange_type: str, interval: str, lookback_minutes: int) -> None:
    """Seed rolling windows with the latest historical candles (up to maxlen).
    We fetch last N minutes (bounded by calculator capacity) and replay them through update().
    Outputs are discarded; state is built so new gap candles have consistent indicators.
    """
    max_needed = 400  # based on deque maxlen definitions in FeatureCalculator
    start_cutoff = datetime.utcnow() - timedelta(minutes=lookback_minutes)
    stmt = (
        select(Candle)
        .where(
            (Candle.symbol == symbol)
            & (Candle.exchange_type == exchange_type)
            & (Candle.interval == interval)
            & (Candle.open_time >= start_cutoff)
        )
        .order_by(Candle.open_time.asc())
    )
    candles: List[Candle] = session.exec(stmt).all()
    # If too many, keep only the tail that fits our max window size
    if len(candles) > max_needed:
        candles = candles[-max_needed:]
    for c in candles:
        fc.update(c.open, c.high, c.low, c.close, c.volume)  # discard features
    log.debug("Seeded feature calculator with %d historical candles", len(candles))


def fill_gaps(symbol: Optional[str] = None, interval: Optional[str] = None) -> int:
    """Fill forward gaps from last stored candle up to 'now'. Also handles empty DB by backfilling lookback period.
    Returns number of inserted/updated candles.
    """
    init_db()
    symbol = (symbol or settings.SYMBOL).lower()
    interval = interval or settings.INTERVAL

    now = datetime.utcnow()
    lookback_minutes = settings.GAP_FILL_LOOKBACK_MINUTES

    with Session(engine) as session:
        # Find last stored candle
        stmt_last = (
            select(Candle)
            .where(
                (Candle.symbol == symbol)
                & (Candle.exchange_type == settings.EXCHANGE_TYPE)
                & (Candle.interval == interval)
            )
            .order_by(Candle.open_time.desc())
            .limit(1)
        )
        last_candle = session.exec(stmt_last).first()

        if last_candle:
            start_dt = _ensure_utc(last_candle.open_time) + timedelta(minutes=1)
            if start_dt > now:
                log.info("No gap to fill (data already up to date).")
                return 0
            log.info("Gap fill starting from %s (last candle) to %s", start_dt.isoformat(), now.isoformat())
        else:
            # Empty DB: start lookback period
            start_dt = now - timedelta(minutes=lookback_minutes)
            log.info("DB empty - backfilling lookback window %s -> %s", start_dt.isoformat(), now.isoformat())

        # Seed feature calculator with prior data for indicator continuity
        fc = FeatureCalculator()
        seed_feature_calculator(fc, session, symbol, settings.EXCHANGE_TYPE, interval, lookback_minutes)

        start_ms = _ms(start_dt)
        end_ms = _ms(now)
        written = 0
        # Expected approximate count for progress (1m interval assumed)
        expected = max(1, int((end_ms - start_ms) / 1000 / 60))

        # 1) Scan and fill internal gaps within the lookback window
        scan_start = now - timedelta(minutes=lookback_minutes)
        missing = find_missing_ranges(session, symbol, settings.EXCHANGE_TYPE, interval, scan_start, now)
        if missing:
            log.info("Found %d internal missing ranges to backfill", len(missing))
        for rng_start, rng_end in missing:
            w = _fill_range_with_rest(session, fc, symbol, interval, rng_start, rng_end)
            written += w
            log.info("gap_fill_internal_range", extra={"range_start": rng_start.isoformat(), "range_end": rng_end.isoformat(), "inserted": w})

        buffer: List[Candle] = []
        while start_ms < end_ms:
            rows, last_close = fetch_klines(symbol, interval, start_ms)
            if not rows:
                break
            for r in rows:
                open_time_ms = int(r[0])
                close_time_ms = int(r[6])
                open_ = float(r[1])
                high = float(r[2])
                low = float(r[3])
                close = float(r[4])
                volume = float(r[5])
                trades = int(r[8]) if len(r) > 8 and r[8] is not None else 0

                feats = fc.update(open_, high, low, close, volume)

                candle = Candle(
                    symbol=symbol,
                    exchange_type=settings.EXCHANGE_TYPE,
                    interval=interval,
                    open_time=_utc(open_time_ms).replace(tzinfo=None),
                    close_time=_utc(close_time_ms).replace(tzinfo=None),
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    trades=trades,
                    **feats,
                )
                buffer.append(candle)
                if len(buffer) >= 1000:
                    bulk_upsert_candles(session, buffer)
                    written += len(buffer)
                    buffer.clear()
                    session.commit()
                    pct = written / expected
                    log.info("gap_fill_commit", extra={"written": written, "expected": expected, "pct": round(pct * 100, 2)})

            if buffer:
                bulk_upsert_candles(session, buffer)
                written += len(buffer)
                buffer.clear()
            session.commit()
            if written:
                pct = written / expected
                log.debug("gap_fill_page", extra={"written": written, "expected": expected, "pct": round(pct * 100, 2), "next_start": _utc(last_close).isoformat()})
            start_ms = last_close + 1

        log.info("gap_fill_completed", extra={"inserted": written})
        return written


def find_missing_ranges(session: Session, symbol: str, exchange_type: str, interval: str, start_dt: datetime, end_dt: datetime):
    """Return list of (start_dt, end_dt) ranges within [start_dt, end_dt] where 1m candles are missing.
    Works only for 1m interval; for other intervals, rely on resampler.
    """
    if interval != "1m":
        return []
    start_dt = _ensure_utc(start_dt)
    end_dt = _ensure_utc(end_dt)
    q = (
        select(Candle.open_time)
        .where((Candle.symbol == symbol) & (Candle.exchange_type == exchange_type) & (Candle.interval == interval) & (Candle.open_time >= start_dt) & (Candle.open_time <= end_dt))
        .order_by(Candle.open_time.asc())
    )
    existing = session.exec(q).all()
    existing_set = set(existing)
    missing_ranges = []
    cur = start_dt.replace(second=0, microsecond=0)
    end = end_dt.replace(second=0, microsecond=0)
    in_gap = False
    gap_start = None
    while cur <= end:
        if cur not in existing_set:
            if not in_gap:
                in_gap = True
                gap_start = cur
        else:
            if in_gap:
                # gap ends at previous minute
                missing_ranges.append((gap_start, cur - timedelta(minutes=1)))
                in_gap = False
                gap_start = None
        cur += timedelta(minutes=1)
    if in_gap and gap_start is not None:
        missing_ranges.append((gap_start, end))
    return missing_ranges


def _fill_range_with_rest(session: Session, fc: FeatureCalculator, symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> int:
    start_ms = _ms(_ensure_utc(start_dt))
    end_ms = _ms(_ensure_utc(end_dt))
    written = 0
    buffer: List[Candle] = []
    while start_ms <= end_ms:
        rows, last_close = fetch_klines(symbol, interval, start_ms)
        if not rows:
            break
        for r in rows:
            open_time_ms = int(r[0])
            close_time_ms = int(r[6])
            if open_time_ms < start_ms or open_time_ms > end_ms:
                continue
            open_ = float(r[1]); high = float(r[2]); low = float(r[3]); close = float(r[4]); volume = float(r[5])
            trades = int(r[8]) if len(r) > 8 and r[8] is not None else 0
            feats = fc.update(open_, high, low, close, volume)
            candle = Candle(
                symbol=symbol,
                exchange_type=settings.EXCHANGE_TYPE,
                interval=interval,
                open_time=_utc(open_time_ms).replace(tzinfo=None),
                close_time=_utc(close_time_ms).replace(tzinfo=None),
                open=open_, high=high, low=low, close=close, volume=volume, trades=trades,
                **feats,
            )
            buffer.append(candle)
        if buffer:
            bulk_upsert_candles(session, buffer)
            written += len(buffer)
            buffer.clear()
        session.commit()
        if last_close <= start_ms:
            break
        start_ms = last_close + 1
    return written


if __name__ == "__main__":
    # Allow manual invocation
    inserted = fill_gaps()
    print(f"Inserted/updated {inserted} candles via gap fill.")
