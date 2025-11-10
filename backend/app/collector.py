import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict

import websockets
from sqlmodel import Session

from .core.config import settings
from .db import engine
from .models import Candle
from .snapshots import save_feature_state
from .features import FeatureCalculator


log = logging.getLogger(__name__)


def _ws_url() -> str:
    # Support single or combined multi-symbol streams
    symbols = settings.SYMBOLS
    if len(symbols) <= 1:
        base = (
            settings.BINANCE_WS_BASE_SPOT
            if settings.EXCHANGE_TYPE == "spot"
            else settings.BINANCE_WS_BASE_FUTURES
        )
        stream = f"{symbols[0]}@kline_{settings.INTERVAL}"
        return f"{base}/{stream}"
    else:
        base = (
            settings.BINANCE_WS_COMBINED_SPOT
            if settings.EXCHANGE_TYPE == "spot"
            else settings.BINANCE_WS_COMBINED_FUTURES
        )
        streams = "/".join(f"{s}@kline_{settings.INTERVAL}" for s in symbols)
        return f"{base}?streams={streams}"


class BinanceCollector:
    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        # Maintain a FeatureCalculator per symbol for correct rolling state
        self.features_by_symbol: Dict[str, FeatureCalculator] = {}

    async def start(self):
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._stop.set()
        if self._task:
            await self._task

    async def _run(self):
        backoff = settings.RECONNECT_MIN_SEC
        while not self._stop.is_set():
            url = _ws_url()
            try:
                log.info("Connecting to %s", url)
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    backoff = settings.RECONNECT_MIN_SEC
                    async for message in ws:
                        if self._stop.is_set():
                            break
                        data = json.loads(message)
                        k = data.get("k") or data.get("data", {}).get("k")
                        if not k:
                            continue
                        is_closed = k.get("x", False)
                        if not is_closed:
                            # only store closed candles
                            continue
                        await self._handle_kline(k)
            except Exception as e:
                log.exception("Websocket error: %s", e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, settings.RECONNECT_MAX_SEC)

    async def _handle_kline(self, kline: dict):
        # Parse fields
        def _to_naive_utc(ms: int) -> datetime:
            return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).replace(tzinfo=None)
        open_time = _to_naive_utc(kline["t"])
        close_time = _to_naive_utc(kline["T"])
        symbol = (kline.get("s") or settings.SYMBOL).lower()
        open_ = float(kline["o"])  # strings in Binance payload
        high = float(kline["h"])
        low = float(kline["l"])
        close = float(kline["c"])
        volume = float(kline["v"])
        trades = int(kline.get("n", 0))

        # Get/Create calculator for this symbol
        fc = self.features_by_symbol.get(symbol)
        if fc is None:
            fc = FeatureCalculator()
            self.features_by_symbol[symbol] = fc
        feats = fc.update(open_, high, low, close, volume)

        candle = Candle(
            symbol=symbol,
            exchange_type=settings.EXCHANGE_TYPE,
            interval=settings.INTERVAL,
            open_time=open_time.replace(tzinfo=None),
            close_time=close_time.replace(tzinfo=None),
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            trades=trades,
            **feats,
        )

        # Upsert by unique index (symbol, exchange_type, interval, open_time)
        with Session(engine) as session:
            self._upsert_candle(session, candle)
            # Periodically persist feature state snapshot per symbol (every 50 candles)
            cnt = getattr(self, "_snap_counts", {})
            cur = cnt.get(symbol, 0) + 1
            cnt[symbol] = cur
            setattr(self, "_snap_counts", cnt)
            if cur % 50 == 0:
                try:
                    save_feature_state(session, symbol, settings.EXCHANGE_TYPE, settings.INTERVAL, self.features_by_symbol[symbol].snapshot())
                    session.commit()
                except Exception:
                    log.exception("Failed to save feature state snapshot for %s", symbol)

    def _upsert_candle(self, session: Session, candle: Candle) -> None:
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
        session.commit()
