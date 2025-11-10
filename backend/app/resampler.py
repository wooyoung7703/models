from datetime import datetime, timezone, timedelta
from typing import Iterable, List

from sqlmodel import Session, select
from .storage import bulk_upsert_candles

from .models import Candle
from .features import FeatureCalculator


def _parse_minutes(interval: str) -> int:
    assert interval.endswith("m"), "Only minute-based intervals supported (e.g., '5m', '15m')"
    return int(interval[:-1])


def _bucket_start(ts: datetime, bucket_min: int) -> datetime:
    # Align to minute boundary and floor to bucket using naive UTC
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc)
    ts = ts.replace(second=0, microsecond=0)
    minute = (ts.minute // bucket_min) * bucket_min
    out = ts.replace(minute=minute)
    # Return naive UTC
    return out.replace(tzinfo=None)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc).replace(tzinfo=None)
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def resample_from_1m(session: Session, symbol: str, exchange_type: str, to_interval: str) -> int:
    """Aggregate 1m candles into a higher timeframe (e.g., 5m, 15m) and upsert into DB.

    Returns number of aggregated candles written.
    """
    target_min = _parse_minutes(to_interval)
    if target_min <= 1 or target_min % 1 != 0:
        raise ValueError("to_interval must be >1 minute and multiple of 1m")

    # Load all 1m candles for symbol/exchange ordered
    q = (
        select(Candle)
        .where(
            (Candle.symbol == symbol)
            & (Candle.exchange_type == exchange_type)
            & (Candle.interval == "1m")
        )
        .order_by(Candle.open_time)
    )
    rows: List[Candle] = list(session.exec(q).all())
    if not rows:
        return 0

    fc = FeatureCalculator()
    written = 0
    out_buffer: List[Candle] = []

    i = 0
    n = len(rows)
    while i < n:
        first = rows[i]
        bucket_start = _bucket_start(_ensure_utc(first.open_time), target_min)
        bucket_end = bucket_start + timedelta(minutes=target_min)

        # Accumulate candles within [bucket_start, bucket_end)
        bucket: List[Candle] = []
        while i < n and bucket_start <= _ensure_utc(rows[i].open_time) < bucket_end:
            bucket.append(rows[i])
            i += 1

        if not bucket:
            continue

        # Incomplete bucket -> skip
        if len(bucket) < target_min:
            continue

        open_time = _ensure_utc(bucket[0].open_time)
        close_time = _ensure_utc(bucket[-1].close_time)
        open_ = bucket[0].open
        close = bucket[-1].close
        high = max(c.high for c in bucket)
        low = min(c.low for c in bucket)
        volume = sum(c.volume for c in bucket)
        trades = sum((c.trades or 0) for c in bucket)

        feats = fc.update(open_, high, low, close, volume)

        out = Candle(
            symbol=symbol,
            exchange_type=exchange_type,
            interval=to_interval,
            open_time=open_time,
            close_time=close_time,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            trades=trades,
            **feats,
        )
        out_buffer.append(out)

    if out_buffer:
        bulk_upsert_candles(session, out_buffer)
        written += len(out_buffer)
        session.commit()
    return written


def resample_incremental(session: Session, symbol: str, exchange_type: str, to_interval: str) -> int:
    """Incrementally aggregate new 1m candles into higher timeframe.

    Logic:
    1. Determine last aggregated open_time for target interval.
    2. Load 1m candles with open_time > last_agg_open_time (or all if none).
    3. Build completed buckets only; skip partial trailing bucket.
    """
    target_min = _parse_minutes(to_interval)
    if target_min <= 1:
        return 0
    # Last aggregated candle for interval
    last_q = (
        select(Candle.open_time)
        .where((Candle.symbol == symbol) & (Candle.exchange_type == exchange_type) & (Candle.interval == to_interval))
        .order_by(Candle.open_time.desc())
        .limit(1)
    )
    last_open = session.exec(last_q).first()

    base_q = (
        select(Candle)
        .where(
            (Candle.symbol == symbol)
            & (Candle.exchange_type == exchange_type)
            & (Candle.interval == "1m")
            & (True if last_open is None else (Candle.open_time > last_open))
        )
        .order_by(Candle.open_time.asc())
    )
    rows: List[Candle] = list(session.exec(base_q).all())
    if not rows:
        return 0

    fc = FeatureCalculator()
    out_buffer: List[Candle] = []
    i = 0
    n = len(rows)
    while i < n:
        first = rows[i]
        bucket_start = _bucket_start(_ensure_utc(first.open_time), target_min)
        bucket_end = bucket_start + timedelta(minutes=target_min)
        bucket: List[Candle] = []
        while i < n and bucket_start <= _ensure_utc(rows[i].open_time) < bucket_end:
            bucket.append(rows[i]); i += 1
        if not bucket:
            continue
        if len(bucket) < target_min:
            # trailing partial bucket -> stop
            break
        open_time = _ensure_utc(bucket[0].open_time)
        close_time = _ensure_utc(bucket[-1].close_time)
        open_ = bucket[0].open
        close = bucket[-1].close
        high = max(c.high for c in bucket)
        low = min(c.low for c in bucket)
        volume = sum(c.volume for c in bucket)
        trades = sum((c.trades or 0) for c in bucket)
        feats = fc.update(open_, high, low, close, volume)
        out_buffer.append(Candle(
            symbol=symbol,
            exchange_type=exchange_type,
            interval=to_interval,
            open_time=open_time,
            close_time=close_time,
            open=open_, high=high, low=low, close=close, volume=volume, trades=trades,
            **feats,
        ))

    if out_buffer:
        bulk_upsert_candles(session, out_buffer)
        session.commit()
    return len(out_buffer)


def _upsert(session: Session, candle: Candle) -> None:
    # Deprecated single-row path kept for backward compatibility
    stmt = select(Candle).where(
        (Candle.symbol == candle.symbol)
        & (Candle.exchange_type == candle.exchange_type)
        & (Candle.interval == candle.interval)
        & (Candle.open_time == candle.open_time)
    )
    existing = session.exec(stmt).first()
    if existing:
        for field, value in candle.model_dump().items():
            if field == "id":
                continue
            setattr(existing, field, value)
        session.add(existing)
    else:
        session.add(candle)
