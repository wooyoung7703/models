import asyncio
from datetime import datetime, timedelta

from backend.app.pipeline import PredictQueueManager, CandlePayload
from backend.app.models import Candle


def _make_candle(symbol: str, minutes: int) -> Candle:
    open_time = datetime.utcnow().replace(microsecond=0) + timedelta(minutes=minutes)
    return Candle(
        symbol=symbol,
        exchange_type="futures",
        interval="1m",
        open_time=open_time,
        close_time=open_time + timedelta(minutes=1),
        open=1.0 + minutes,
        high=1.1 + minutes,
        low=0.9 + minutes,
        close=1.05 + minutes,
        volume=100 + minutes,
        trades=10,
    )


def test_predict_queue_publish_and_get_preserves_order():
    async def _run():
        mgr = PredictQueueManager(maxsize=2)
        c1 = _make_candle("xrpusdt", 0)
        c2 = _make_candle("xrpusdt", 1)
        assert mgr.publish("xrpusdt", CandlePayload(candle=c1))
        assert mgr.publish("xrpusdt", CandlePayload(candle=c2))

        payload = await mgr.get("xrpusdt", timeout=0.1)
        assert payload is not None
        assert payload.candle.open_time == c1.open_time

    asyncio.run(_run())


def test_predict_queue_drop_oldest_on_overflow():
    async def _run():
        mgr = PredictQueueManager(maxsize=2)
        c1 = _make_candle("btcusdt", 0)
        c2 = _make_candle("btcusdt", 1)
        c3 = _make_candle("btcusdt", 2)

        mgr.publish("btcusdt", CandlePayload(candle=c1))
        mgr.publish("btcusdt", CandlePayload(candle=c2))
        mgr.publish("btcusdt", CandlePayload(candle=c3))

        first = await mgr.get("btcusdt", timeout=0.01)
        second = await mgr.get("btcusdt", timeout=0.01)

        assert first is not None and first.candle.open_time == c2.open_time
        assert second is not None and second.candle.open_time == c3.open_time

    asyncio.run(_run())