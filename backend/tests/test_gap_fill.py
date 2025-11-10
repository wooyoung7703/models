from datetime import datetime, timedelta, timezone
from sqlmodel import Session
from sqlalchemy import text, delete

from backend.app.db import engine, init_db
from backend.app.models import Candle
from backend.app.gap_fill import find_missing_ranges


def test_find_missing_ranges_simple():
    init_db()
    sym = 'xrpusdt'
    ex = 'futures'
    itv = '1m'
    # Prepare a tiny in-memory like window by writing 3 minutes with a gap
    now = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
    t0 = now - timedelta(minutes=10)
    with Session(engine) as session:
        # Clear any rows in that small window for isolation (best-effort)
        t_end = t0 + timedelta(minutes=10)
        session.exec(
            delete(Candle).where(
                (Candle.symbol==sym) & (Candle.exchange_type==ex) & (Candle.interval==itv) & (Candle.open_time.between(t0, t_end))
            )
        )
        session.commit()
        # Insert t0, t0+1m, skip t0+2m, insert t0+3m
        for m in (0, 1, 3):
            ot = t0 + timedelta(minutes=m)
            ct = ot + timedelta(seconds=59, microseconds=999000)
            c = Candle(symbol=sym, exchange_type=ex, interval=itv,
                       open_time=ot, close_time=ct,
                       open=1.0, high=1.1, low=0.9, close=1.0, volume=1000, trades=1)
            session.add(c)
        session.commit()
        gaps = find_missing_ranges(session, sym, ex, itv, t0, t0 + timedelta(minutes=4))
        # We intentionally skipped at least one minute; ensure at least one gap detected
        assert len(gaps) >= 1
        # Each gap start must be before or equal to gap end
        for a, b in gaps:
            assert a <= b
