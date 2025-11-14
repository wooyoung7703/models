import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict

from websockets.client import connect
from sqlmodel import Session

from .core.config import settings
from .db import engine
from .models import Candle
from .snapshots import save_feature_state
from .features import FeatureCalculator
from .seq_buffer import get_buffer, extract_vector_from_candle


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
        # Live prices per symbol from in-progress klines (updated on every WS tick)
        self.live_prices: Dict[str, float] = {}

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
        last_error_log: float = 0.0
        while not self._stop.is_set():
            url = _ws_url()
            try:
                log.info("Connecting to %s", url)
                async with connect(url, ping_interval=20, ping_timeout=20) as ws:
                    backoff = settings.RECONNECT_MIN_SEC
                    async for message in ws:
                        if self._stop.is_set():
                            break
                        data = json.loads(message)
                        k = data.get("k") or data.get("data", {}).get("k")
                        if not k:
                            continue
                        is_closed = k.get("x", False)
                        # Always capture live price from in-progress candles
                        try:
                            sym = (k.get("s") or settings.SYMBOL).lower()
                            live_c = float(k.get("c")) if k.get("c") is not None else None
                            if live_c is not None:
                                self.live_prices[sym] = live_c
                        except Exception:
                            pass
                        if not is_closed:
                            # Skip persisting until candle closes
                            continue
                        await self._handle_kline(k)
            except asyncio.CancelledError:
                # Graceful shutdown requested; exit loop without escalating
                log.info("Collector task cancelled; exiting websocket loop")
                break
            except Exception as e:
                # Throttle repetitive error logs (network flaps etc.)
                import time, traceback
                now_ts = time.time()
                if now_ts - last_error_log >= settings.ERROR_LOG_MIN_INTERVAL_SECONDS:
                    last_error_log = now_ts
                    log.exception("Websocket error (next log after %ds): %s", settings.ERROR_LOG_MIN_INTERVAL_SECONDS, e)
                    _try_alert(f"WS error: {e}\n{traceback.format_exc()[:500]}")
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

        # sanitize feature types for model schema
        # Ensure integer fields are cast; remove unknown keys
        if "run_up" in feats:
            try:
                feats["run_up"] = int(feats["run_up"])
            except Exception:
                feats["run_up"] = 0
        if "run_down" in feats:
            try:
                feats["run_down"] = int(feats["run_down"])
            except Exception:
                feats["run_down"] = 0
        for _k in ["id"]:
            feats.pop(_k, None)
        # Filter out any feature keys not defined on Candle to avoid type mismatches
        from .models import Candle as _CandleModel
        allowed_fields = set(_CandleModel.model_fields.keys())  # type: ignore[attr-defined]
        clean_kwargs = {k: v for k, v in feats.items() if k in allowed_fields and k not in {"run_up","run_down","id"}}
        candle = Candle(
            id=None,
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
            run_up=int(feats.get("run_up") or 0),
            run_down=int(feats.get("run_down") or 0),
            **clean_kwargs,
        )

        # Upsert by unique index (symbol, exchange_type, interval, open_time)
        with Session(engine) as session:
            try:
                self._upsert_candle(session, candle)
            except Exception as e:
                log.warning("DB upsert failure: %s", e)
                _try_alert(f"DB upsert failed: {e}")
                return
            # Periodically persist feature state snapshot per symbol (every 50 candles)
            cadence = settings.FEATURE_SNAPSHOT_EVERY if hasattr(settings, 'FEATURE_SNAPSHOT_EVERY') else 50
            cnt = getattr(self, "_snap_counts", {})
            cur = cnt.get(symbol, 0) + 1
            cnt[symbol] = cur
            setattr(self, "_snap_counts", cnt)
            if cadence > 0 and cur % cadence == 0:
                try:
                    save_feature_state(session, symbol, settings.EXCHANGE_TYPE, settings.INTERVAL, self.features_by_symbol[symbol].snapshot())
                    session.commit()
                except Exception:
                    log.exception("Failed to save feature state snapshot for %s", symbol)
        # Update sequence buffer (after commit) for sequence models
        try:
            buf = get_buffer(symbol)
            buf.append(extract_vector_from_candle(candle))
        except Exception:
            log.exception("Failed to append to sequence buffer for %s", symbol)

    def _upsert_candle(self, session: Session, candle: Candle) -> None:
        from sqlmodel import select
        from sqlalchemy.exc import IntegrityError

        stmt = select(Candle).where(
            (Candle.symbol == candle.symbol)
            & (Candle.exchange_type == candle.exchange_type)
            & (Candle.interval == candle.interval)
            & (Candle.open_time == candle.open_time)
        )
        existing = session.exec(stmt).first()
        if existing:
            try:
                for field, value in candle.model_dump().items():
                    if field == "id":
                        continue
                    setattr(existing, field, value)
                session.add(existing)
                session.commit()
            except IntegrityError:
                session.rollback()
                log.debug("[collector] duplicate during update (ignored) %s", candle.open_time)
            except Exception as e:
                session.rollback()
                log.warning("[collector] update failed: %s", e)
        else:
            try:
                session.add(candle)
                session.commit()
            except IntegrityError:
                session.rollback()
                log.debug("[collector] duplicate insert ignored %s", candle.open_time)
            except Exception as e:
                session.rollback()
                log.warning("[collector] insert failed: %s", e)
                _try_alert(f"Insert failed: {e}")


# --- Alert helper (non-blocking best-effort) ---
def _try_alert(message: str) -> None:
    url = getattr(settings, 'ALERT_WEBHOOK_URL', None)
    if not url:
        return
    import threading, requests
    def _send():
        try:
            requests.post(url, json={"text": message[:1000]})
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()
