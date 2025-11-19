"""Async queues bridging collector and predictor to avoid hot DB polling."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import Candle

log = logging.getLogger(__name__)


@dataclass
class CandlePayload:
    """Payload transported from the collector to predictor loops."""

    candle: Candle
    feature_snapshot: Optional[Dict[str, Any]] = None
    seq_vector: Optional[List[float]] = None
    enqueued_at: float = field(default_factory=time.time)


class PredictQueueManager:
    """Per-symbol bounded queues with simple drop-oldest backpressure."""

    def __init__(self, maxsize: int = 4) -> None:
        self.maxsize = max(1, int(maxsize))
        self._queues: Dict[str, asyncio.Queue[CandlePayload]] = {}

    def _queue(self, symbol: str) -> asyncio.Queue[CandlePayload]:
        sym = symbol.lower()
        q = self._queues.get(sym)
        if q is None:
            # Queue creation must happen while an event loop is running.
            q = asyncio.Queue(maxsize=self.maxsize)
            self._queues[sym] = q
        return q

    def publish(self, symbol: str, payload: CandlePayload) -> bool:
        q = self._queue(symbol)
        if q.full():
            try:
                dropped = q.get_nowait()
                log.warning(
                    "predict_queue drop oldest symbol=%s open_time=%s",  # pragma: no cover - logging only
                    symbol,
                    getattr(getattr(dropped, "candle", None), "open_time", None),
                )
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(payload)
            return True
        except asyncio.QueueFull:
            log.error("predict_queue overflow persisted for symbol=%s", symbol)
            return False

    async def get(self, symbol: str, timeout: float) -> Optional[CandlePayload]:
        q = self._queue(symbol)
        if timeout is None or timeout <= 0:
            if q.empty():
                return None
            try:
                return q.get_nowait()
            except asyncio.QueueEmpty:
                return None
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def qsize(self, symbol: str) -> int:
        return self._queue(symbol).qsize()

    async def drain(self, timeout: float = 1.0) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        for q in self._queues.values():
            while not q.empty() and time.monotonic() < deadline:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover - loop guard
                    break


__all__ = ["CandlePayload", "PredictQueueManager"]
