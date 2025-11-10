import argparse
import math
import time
from datetime import datetime, timedelta, timezone
from typing import List, Tuple
import urllib.parse
import urllib.request
import json
import logging

from sqlmodel import Session

from .core.config import settings
from .db import engine, init_db
from .features import FeatureCalculator
from .models import Candle
from .storage import bulk_upsert_candles
from .logging_config import init_logging


def _rest_base() -> str:
    if settings.EXCHANGE_TYPE == "spot":
        return "https://api.binance.com/api/v3/klines"
    else:
        return "https://fapi.binance.com/fapi/v1/klines"


def _ms(dt: datetime) -> int:
    # Treat naive datetimes as UTC for epoch conversion
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _utc(ms: int) -> datetime:
    # store naive UTC for consistency with DB
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).replace(tzinfo=None)
def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def fetch_klines(symbol: str, interval: str, start_time_ms: int, limit: int = 1000) -> Tuple[List[list], int]:
    """Fetch a page of klines. Returns (rows, last_close_time_ms).
    Uses public REST; handle networking errors with simple retry.
    """
    base = _rest_base()
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": str(limit),
        "startTime": str(start_time_ms),
    }
    url = base + "?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, dict) and data.get("code"):
                    raise RuntimeError(f"Binance error: {data}")
                if not data:
                    return [], start_time_ms
                last_close = int(data[-1][6])  # closeTime
                return data, last_close
        except Exception as e:
            sleep_s = 1.5 * (attempt + 1)
            time.sleep(sleep_s)
    return [], start_time_ms


def backfill(symbol: str, interval: str, since: datetime, until: datetime, commit_every: int = 1000, incremental: bool = False) -> int:
    init_db()
    fc = FeatureCalculator()
    written = 0
    start_ms = _ms(since)
    end_ms = _ms(until)
    # 예상 총 캔들 수 (분 단위 간격 가정) - 진행률 계산용
    expected_total = max(1, int((end_ms - start_ms) / 1000 / 60))
    last_log_time = time.time()

    with Session(engine) as session:
        if incremental:
            from sqlmodel import select
            last = session.exec(
                select(Candle.open_time).where(
                    (Candle.symbol == symbol.lower())
                    & (Candle.exchange_type == settings.EXCHANGE_TYPE)
                    & (Candle.interval == interval)
                ).order_by(Candle.open_time.desc()).limit(1)
            ).first()
            if last and _ensure_utc(last) > _ensure_utc(since):
                # resume from last + 1 minute
                since = _ensure_utc(last) + timedelta(minutes=1)
                start_ms = _ms(since)
        buffer: List[Candle] = []
        while start_ms < end_ms:
            rows, last_close = fetch_klines(symbol, interval, start_ms)
            if not rows:
                break
            for r in rows:
                # r fields: [openTime, open, high, low, close, volume, closeTime, ... , trades, ...]
                open_time = _utc(int(r[0]))
                close_time = _utc(int(r[6]))
                open_ = float(r[1])
                high = float(r[2])
                low = float(r[3])
                close = float(r[4])
                volume = float(r[5])
                trades = int(r[8]) if len(r) > 8 and r[8] is not None else 0

                feats = fc.update(open_, high, low, close, volume)

                candle = Candle(
                    symbol=symbol.lower(),
                    exchange_type=settings.EXCHANGE_TYPE,
                    interval=interval,
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
                buffer.append(candle)
                if len(buffer) >= commit_every:
                    bulk_upsert_candles(session, buffer)
                    session.commit()
                    written += len(buffer)
                    buffer.clear()
                    pct = written / expected_total
                    logging.info("[commit] %d/%d (%.2f%%) up to %s", written, expected_total, pct * 100, close_time.isoformat())

            if buffer:
                bulk_upsert_candles(session, buffer)
                written += len(buffer)
                buffer.clear()
            session.commit()
            page_pct = written / expected_total
            # 5초에 한 번 또는 페이지 끝에서 출력
            if time.time() - last_log_time > 5:
                logging.info("[page] rows=%d progress=%d/%d (%.2f%%) next_start=%s", len(rows), written, expected_total, page_pct * 100, _utc(last_close).isoformat())
                last_log_time = time.time()
            # Advance start to last_close + 1ms to avoid duplicates
            start_ms = last_close + 1
            # be friendly to API
            time.sleep(0.2)

    return written


def _upsert(session: Session, candle: Candle) -> None:
    # Kept for backward-compatibility; no longer used in main flow
    from sqlmodel import select
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


def main():
    parser = argparse.ArgumentParser(description="Backfill Binance klines into DB with features.")
    parser.add_argument("--symbol", default=settings.SYMBOL, help="Symbol, e.g., xrpusdt")
    parser.add_argument("--interval", default="1m", help="Interval, e.g., 1m")
    parser.add_argument("--years", type=float, default=2.0, help="How many years back to load (ignored if --since provided)")
    parser.add_argument("--since", help="Override start datetime (UTC, e.g., 2023-01-01T00:00:00)")
    parser.add_argument("--until", help="Override end datetime (UTC, default now)")
    parser.add_argument("--incremental", action="store_true", help="Resume from latest stored candle instead of full range")
    args = parser.parse_args()

    now = datetime.now(tz=timezone.utc)
    if args.until:
        until = datetime.fromisoformat(args.until).astimezone(timezone.utc)
    else:
        until = now

    if args.since:
        since = datetime.fromisoformat(args.since).astimezone(timezone.utc)
    else:
        since = until - timedelta(days=int(args.years * 365))

    total = backfill(args.symbol, args.interval, since, until, incremental=args.incremental)
    mode = "incremental" if args.incremental else "full"
    logging.info("Backfilled %d candles for %s %s [%s..%s] mode=%s exchange=%s", total, args.symbol, args.interval, since, until, mode, settings.EXCHANGE_TYPE)


if __name__ == "__main__":
    init_logging()
    main()
