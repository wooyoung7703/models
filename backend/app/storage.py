from typing import List, Dict, Tuple
from sqlmodel import Session, select
from sqlalchemy import and_, or_

def _normalize_dt(dt):
    """Ensure datetime is stored/compared as naive UTC (strip tzinfo)."""
    if dt is None:
        return dt
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt

from .models import Candle


def bulk_upsert_candles(session: Session, candles: List[Candle], chunk_size: int = 1000) -> int:
    """Bulk upsert Candle objects.
    Groups by (symbol, exchange_type, interval) to minimize queries.
    Returns number of candles processed.
    NOTE: Does not commit; caller should commit after invocation.
    """
    if not candles:
        return 0

    # Group candles by key
    groups: Dict[Tuple[str, str, str], List[Candle]] = {}
    for c in candles:
        key = (c.symbol, c.exchange_type, c.interval)
        groups.setdefault(key, []).append(c)

    for (symbol, exchange_type, interval), group in groups.items():
        # Process in chunks to avoid extremely large IN clauses
        idx = 0
        while idx < len(group):
            chunk = group[idx: idx + chunk_size]
            idx += chunk_size
            # Normalize datetimes to naive for consistent comparison (SQLite stores naive)
            for c0 in chunk:
                c0.open_time = _normalize_dt(c0.open_time)
                if c0.close_time is not None:
                    c0.close_time = _normalize_dt(c0.close_time)
            open_times = [c.open_time for c in chunk]
            stmt = (
                select(Candle)
                .where(
                    (Candle.symbol == symbol)
                    & (Candle.exchange_type == exchange_type)
                    & (Candle.interval == interval)
                    & (Candle.open_time.in_(open_times))
                )
            )
            existing_rows = session.exec(stmt).all()
            existing_map = { _normalize_dt(e.open_time): e for e in existing_rows }
            for c in chunk:
                existing = existing_map.get(_normalize_dt(c.open_time))
                if existing:
                    for field, value in c.model_dump().items():
                        if field == "id":
                            continue
                        setattr(existing, field, value)
                    session.add(existing)
                else:
                    session.add(c)
    return len(candles)
