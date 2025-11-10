import argparse
import json
import logging
import os
import pickle
from datetime import datetime, timezone
from typing import List

import numpy as np
from sqlmodel import Session, select

try:
    import xgboost as xgb
except Exception:
    raise SystemExit("xgboost is required for inference. Please install it.")

# Support running as a script from repo root (python backend/app/training/infer_bottom.py)
try:
    from ..core.config import settings
    from ..db import engine, init_db
    from ..models import Candle
except Exception:  # pragma: no cover - fallback path
    import sys
    sys.path.append('.')
    from backend.app.core.config import settings
    from backend.app.db import engine, init_db
    from backend.app.models import Candle


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
log = logging.getLogger("infer_bottom")


def _now_utc_naive() -> datetime:
    return datetime.now(tz=timezone.utc).replace(tzinfo=None, second=0, microsecond=0)


def _build_feature_row(row: Candle, feature_names: List[str], fill_value: float = 0.0) -> List[float]:
    feats: List[float] = []
    for f in feature_names:
        try:
            v = getattr(row, f)
            if v is None:
                v = fill_value
            feats.append(float(v))
        except Exception:
            feats.append(fill_value)
    return feats


def infer(args) -> None:
    init_db()

    model_path = args.model
    if not os.path.exists(model_path):
        raise SystemExit(f"Model file not found: {model_path}")
    with open(model_path, 'rb') as f:
        payload = pickle.load(f)
    booster = payload.get('booster')
    feature_names: List[str] = payload.get('features')
    mode = payload.get('mode', 'cls_bottom')
    if booster is None or feature_names is None:
        raise SystemExit("Invalid model payload: missing booster or features")
    if mode != 'cls_bottom':
        log.warning("Model mode is %s (expected 'cls_bottom'). Proceeding anyway.", mode)

    symbol = (args.symbol or settings.SYMBOL).lower()
    exchange_type = (args.exchange_type or settings.EXCHANGE_TYPE)
    interval = args.interval

    with Session(engine) as session:
        row: Candle | None = session.exec(
            select(Candle)
            .where(
                (Candle.symbol == symbol)
                & (Candle.exchange_type == exchange_type)
                & (Candle.interval == interval)
            )
            .order_by(Candle.open_time.desc())
            .limit(1)
        ).first()

    if row is None:
        raise SystemExit(f"No candle found for {symbol}/{exchange_type}/{interval}.")

    X = np.array([_build_feature_row(row, feature_names)], dtype=np.float32)
    dtest = xgb.DMatrix(X, feature_names=feature_names)
    prob = float(booster.predict(dtest)[0])  # logistic output

    result = {
        "symbol": symbol,
        "exchange_type": exchange_type,
        "interval": interval,
        "open_time": row.open_time.isoformat() if row.open_time else None,
        "close_time": row.close_time.isoformat() if row.close_time else None,
        "close": row.close,
        "prob_bottom": prob,
        "threshold": args.threshold,
        "is_bottom": prob >= args.threshold,
        "model": os.path.basename(model_path),
    }
    print(json.dumps(result, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description="Infer bottom probability for the latest candle using an XGBoost model")
    p.add_argument('--model', type=str, default='backend/app/training/models/xgb_bottom_futures_xrp_1m_14d.pkl')
    p.add_argument('--symbol', type=str, default=None, help='e.g., xrpusdt (defaults to settings.SYMBOL)')
    p.add_argument('--exchange-type', type=str, default=None, help="'spot' or 'futures' (defaults to settings.EXCHANGE_TYPE)")
    p.add_argument('--interval', type=str, default='1m')
    p.add_argument('--threshold', type=float, default=0.5)
    args = p.parse_args()
    infer(args)


if __name__ == '__main__':
    main()
