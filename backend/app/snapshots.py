import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict
from sqlmodel import Session, select

from .models import FeatureState

log = logging.getLogger(__name__)


def save_feature_state(session: Session, symbol: str, exchange_type: str, interval: str, state: Dict) -> None:
    """Upsert feature state snapshot for (symbol, exchange_type, interval)."""
    payload = json.dumps(state)
    stmt = select(FeatureState).where(
        (FeatureState.symbol == symbol)
        & (FeatureState.exchange_type == exchange_type)
        & (FeatureState.interval == interval)
    )
    row = session.exec(stmt).first()
    if row:
        row.state_json = payload
        row.updated_at = datetime.now(tz=timezone.utc)
        row.version = 1
        session.add(row)
    else:
        row = FeatureState(
            symbol=symbol,
            exchange_type=exchange_type,
            interval=interval,
            state_json=payload,
            updated_at=datetime.now(tz=timezone.utc),
            version=1,
        )
        session.add(row)


def load_feature_state(session: Session, symbol: str, exchange_type: str, interval: str) -> Optional[Dict]:
    stmt = select(FeatureState).where(
        (FeatureState.symbol == symbol)
        & (FeatureState.exchange_type == exchange_type)
        & (FeatureState.interval == interval)
    )
    row = session.exec(stmt).first()
    if not row:
        return None
    try:
        return json.loads(row.state_json)
    except Exception as e:
        log.warning("Failed to parse feature state JSON for %s/%s/%s: %s", symbol, exchange_type, interval, e)
        return None
