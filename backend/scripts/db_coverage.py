"""Print DB coverage stats for candles.

Reports, per symbol in settings.SYMBOLS (and current INTERVAL/EXCHANGE_TYPE):
- Row count
- Min/Max open_time
- Span in days

Run from repo root:
  python backend/scripts/db_coverage.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import timezone

# Ensure repo root import
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sqlmodel import Session, select
from sqlalchemy import func
from backend.app.db import engine
from backend.app.core.config import settings
from backend.app.models import Candle


def fmt_dt(dt):
    try:
        return dt.replace(tzinfo=dt.tzinfo or timezone.utc).isoformat()
    except Exception:
        return str(dt)


def main():
    ex = settings.EXCHANGE_TYPE
    itv = settings.INTERVAL
    print(f"EXCHANGE={ex} INTERVAL={itv} SYMBOLS={settings.SYMBOLS}")
    with Session(engine) as s:
        for sym in settings.SYMBOLS:
            q = s.exec(
                select(
                    func.count(Candle.id),
                    func.min(Candle.open_time),
                    func.max(Candle.open_time),
                ).where((Candle.symbol==sym)&(Candle.exchange_type==ex)&(Candle.interval==itv))  # type: ignore[attr-defined]
            ).first()
            cnt, mn, mx = (q or (0, None, None))
            span_days = None
            if mn and mx:
                try:
                    span_days = round((mx - mn).total_seconds() / 86400, 2)
                except Exception:
                    span_days = None
            print(f"- {sym}: rows={cnt} start={fmt_dt(mn) if mn else None} end={fmt_dt(mx) if mx else None} span_days={span_days}")


if __name__ == "__main__":
    main()
