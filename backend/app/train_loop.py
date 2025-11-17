import asyncio
import logging
import os
from datetime import datetime, timezone

from .core.config import settings
from .scheduler import run_training_sequence

"""
Lightweight standalone training loop for dedicated trainer container.

Runs the full base model + stacking retrain sequence every interval (default 1h).
Environment variables:
  TRAIN_INTERVAL_SECONDS (default 3600)
  TRAIN_BASE_DAYS (fallback to settings.BOTTOM_TRAIN_DAYS)
  TRAIN_STACKING_DAYS (fallback to settings.STACKING_META_DAYS)
  TRAIN_LOG_LEVEL (default INFO)

Avoid overlapping runs: if a run exceeds the interval, the next cycle waits
for completion then immediately starts the next run.
"""

log_level = os.getenv("TRAIN_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("trainer")

TRAIN_INTERVAL_SECONDS = int(os.getenv("TRAIN_INTERVAL_SECONDS", "3600"))
# 학습 윈도우는 settings의 BOTTOM_TRAIN_DAYS / STACKING_META_DAYS 를 단일 소스로 사용하고,
# 필요할 경우 동일 이름의 env(BOTTOM_TRAIN_DAYS, STACKING_META_DAYS)로만 override 한다.
BASE_DAYS = int(os.getenv("BOTTOM_TRAIN_DAYS", str(getattr(settings, "BOTTOM_TRAIN_DAYS", 14))))
STACKING_DAYS = int(os.getenv("STACKING_META_DAYS", str(getattr(settings, "STACKING_META_DAYS", 45))))
LAST_RUN_PATH = os.getenv("TRAIN_LAST_RUN_PATH", "/app/backend/data/trainer_last_run.json")


async def run_once() -> None:
    start = datetime.now(tz=timezone.utc).isoformat()
    logger.info("[trainer] hourly start %s base_days=%d stacking_days=%d", start, BASE_DAYS, STACKING_DAYS)
    try:
        await run_training_sequence(BASE_DAYS, STACKING_DAYS, reason="hourly")
    except Exception as e:
        logger.exception("[trainer] training sequence failed: %s", e)
    end = datetime.now(tz=timezone.utc).isoformat()
    logger.info("[trainer] hourly end %s", end)
    # Persist last run meta for external health/monitoring (best-effort)
    try:
        import json
        payload = {
            "start": start,
            "end": end,
            "base_days": BASE_DAYS,
            "stacking_days": STACKING_DAYS,
        }
        with open(LAST_RUN_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass


HEARTBEAT_PATH = os.getenv("TRAIN_HEARTBEAT_PATH", "/app/backend/data/trainer_heartbeat.txt")


def _write_heartbeat() -> None:
    try:
        os.makedirs(os.path.dirname(HEARTBEAT_PATH), exist_ok=True)
        with open(HEARTBEAT_PATH, "w", encoding="utf-8") as f:
            f.write(datetime.now(tz=timezone.utc).isoformat())
    except Exception:
        # heartbeat is best-effort; don't fail the loop
        pass


async def loop() -> None:
    while True:
        _write_heartbeat()
        await run_once()
        _write_heartbeat()
        logger.info("[trainer] sleeping %ds", TRAIN_INTERVAL_SECONDS)
        # During sleep, update heartbeat periodically so healthcheck stays green
        slept = 0
        while slept < TRAIN_INTERVAL_SECONDS:
            await asyncio.sleep(min(30, TRAIN_INTERVAL_SECONDS - slept))
            slept += min(30, TRAIN_INTERVAL_SECONDS - (slept))
            _write_heartbeat()


def main() -> None:
    logger.info("[trainer] starting loop interval=%ds base_days=%d stacking_days=%d", TRAIN_INTERVAL_SECONDS, BASE_DAYS, STACKING_DAYS)
    asyncio.run(loop())


if __name__ == "__main__":
    main()
