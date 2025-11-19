"""
Lightweight per-symbol rolling sequence buffers for sequence models.

We capture a compact feature vector from each closed candle and keep the last N.
Adapters can read from these buffers to build inputs for LSTM/Transformer.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
import gzip
import json
import logging
import os
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from .core.config import settings


class SequenceBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._q: Deque[List[float]] = deque(maxlen=self.capacity)

    def append(self, vec: List[float]) -> None:
        self._q.append(vec)

    def to_list(self) -> List[List[float]]:
        return list(self._q)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._q)


_buffers: Dict[str, SequenceBuffer] = {}


def get_buffer(symbol: str) -> SequenceBuffer:
    sym = symbol.lower()
    buf = _buffers.get(sym)
    if buf is None:
        buf = SequenceBuffer(settings.SEQ_LEN)
        _buffers[sym] = buf
    return buf


def extract_vector_from_candle(candle: "Candle") -> List[float]:  # type: ignore[name-defined]
    """Return sequence feature vector.

    Expanded to 16 dimensions to better feed the Transformer checkpoint (feature_dim=16).
    LSTM adapter will still pad to larger feature_dim (e.g., 55) until retraining aligns.
    Order must remain stable; append new features only at the end.
    """
    def _f(name: str, default: float) -> float:
        try:
            v = getattr(candle, name)
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    vec = [
        _f("close", 0.0),            # 0 price
        _f("rsi_14", 50.0),          # 1 momentum
        _f("bb_pct_b_20_2", 0.5),    # 2 volatility position
        _f("macd_hist", 0.0),        # 3 momentum histogram
        _f("vol_z_20", 0.0),         # 4 volume anomaly
        _f("williams_r_14", -50.0),  # 5 oversold metric
        _f("drawdown_from_max_20", 0.0),  # 6 local drawdown
        _f("atr_14", 0.0),           # 7 volatility (range)
        _f("cci_20", 0.0),           # 8 typical price deviation
        _f("run_up", 0.0),           # 9 consecutive up closes
        _f("run_down", 0.0),         # 10 consecutive down closes
        _f("obv", 0.0),              # 11 on-balance volume
        _f("mfi_14", 50.0),          # 12 money flow index
        _f("cmf_20", 0.0),           # 13 chaikin money flow
        _f("body_pct_of_range", 0.0),# 14 candle body proportion
        _f("vwap_20_dev", 0.0),      # 15 deviation from short VWAP
    ]
    return vec


def snapshot_buffers() -> Dict[str, List[List[float]]]:
    return {
        sym: buf.to_list()
        for sym, buf in _buffers.items()
        if len(buf) > 0
    }


def save_buffers_to_path(path: str) -> int:
    if not path:
        return 0
    data = snapshot_buffers()
    payload = {
        "version": 1,
        "seq_len": int(settings.SEQ_LEN),
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "buffers": data,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    _archive_snapshot(payload)
    return len(data)


def load_buffers_from_path(path: str, *, seq_len: Optional[int] = None) -> int:
    if not path or not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    buffers = payload.get("buffers") or payload.get("symbols") or {}
    loaded = 0
    capacity = int(seq_len or settings.SEQ_LEN)
    for sym, vectors in buffers.items():
        if not isinstance(vectors, list):
            continue
        buf = SequenceBuffer(capacity)
        for vec in vectors[-capacity:]:
            if not isinstance(vec, list):
                continue
            cleaned: List[float] = []
            for val in vec:
                try:
                    cleaned.append(float(val))
                except Exception:
                    cleaned.append(0.0)
            buf.append(cleaned)
        if len(buf) == 0:
            continue
        _buffers[str(sym).lower()] = buf
        loaded += 1
    return loaded


def _archive_snapshot(payload: Dict[str, Any]) -> None:
    archive_dir = getattr(settings, "SEQ_BUFFER_SNAPSHOT_ARCHIVE_DIR", "") or ""
    if not archive_dir:
        return
    keep = max(0, int(getattr(settings, "SEQ_BUFFER_SNAPSHOT_ARCHIVE_KEEP", 0)))
    compress = bool(getattr(settings, "SEQ_BUFFER_SNAPSHOT_COMPRESS", False))
    directory = Path(archive_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - filesystem errors rare
        logging.warning("Sequence buffer archive dir create failed: %s", exc)
        return
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    suffix = ".json.gz" if compress else ".json"
    archive_path = directory / f"seq_buffer_{ts}{suffix}"
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    try:
        if compress:
            with gzip.open(archive_path, "wb") as fh:
                fh.write(data)
        else:
            archive_path.write_bytes(data)
    except Exception as exc:  # pragma: no cover - best-effort archival
        logging.warning("Sequence buffer archive write failed: %s", exc)
        return
    if keep > 0:
        _enforce_archive_retention(directory, keep)


def _enforce_archive_retention(directory: Path, keep: int) -> None:
    try:
        files = sorted(
            (p for p in directory.glob("seq_buffer_*.json*") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception as exc:  # pragma: no cover - filesystem errors rare
        logging.warning("Sequence buffer archive scan failed: %s", exc)
        return
    for old in files[keep:]:
        try:
            old.unlink()
        except Exception:
            logging.debug("Sequence buffer archive cleanup skipped for %s", old)


def clear_buffers() -> None:
    _buffers.clear()


__all__ = [
    "SequenceBuffer",
    "get_buffer",
    "extract_vector_from_candle",
    "snapshot_buffers",
    "save_buffers_to_path",
    "load_buffers_from_path",
    "clear_buffers",
]
