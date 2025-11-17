#!/usr/bin/env python
"""Inspect in-memory sequence buffers for all configured symbols.

Prints length and first few vectors. Run after ws_server startup to verify seeding
and ongoing candle ingestion for LSTM/Transformer readiness.
"""
import os
import sys
from sqlmodel import Session, select

# Ensure project root on path when running as standalone script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from backend.app.core.config import settings  # type: ignore
from backend.app.db import engine  # type: ignore
from backend.app.seq_buffer import get_buffer, extract_vector_from_candle  # type: ignore
from backend.app.models import Candle  # type: ignore

SHOW_ROWS = 3


def main() -> None:
    print(f"Symbols: {settings.SYMBOLS}, SEQ_LEN={settings.SEQ_LEN}, SEQ_MIN_READY={settings.SEQ_MIN_READY}")
    # Reconstruct expected sequence vectors directly from DB (stateless run)
    with Session(engine) as session:
        for sym in settings.SYMBOLS:
            rows = session.exec(
                select(Candle)
                .where((Candle.symbol == sym) & (Candle.exchange_type == settings.EXCHANGE_TYPE) & (Candle.interval == settings.INTERVAL))
                .order_by(Candle.open_time.desc())
                .limit(settings.SEQ_LEN)
            ).all()
            vecs = [extract_vector_from_candle(c) for c in reversed(rows)]
            seq_len = len(vecs)
            status = "ready" if seq_len >= settings.SEQ_MIN_READY else "warming"
            print(f"\nSymbol {sym}: reconstructed_len={seq_len} status={status}")
            if vecs:
                print(f"Oldest {min(SHOW_ROWS, len(vecs))} vectors (truncated):")
                for i, vec in enumerate(vecs[:SHOW_ROWS]):
                    print(f"  [{i}] size={len(vec)} sample={vec[:6]}")
                last = vecs[-1]
                print(f"Most recent size={len(last)} sample={last[:6]}")
            else:
                print("  (no candles found)")


if __name__ == "__main__":
    main()
