import os
import sys
import time

HEARTBEAT_PATH = os.getenv("TRAIN_HEARTBEAT_PATH", "/app/backend/data/trainer_heartbeat.txt")
# Allow a wide window: healthy if heartbeat updated within last MAX_AGE seconds
# Default: 2 hours unless TRAIN_INTERVAL_SECONDS is provided
INTERVAL = int(os.getenv("TRAIN_INTERVAL_SECONDS", "3600"))
MAX_AGE = int(os.getenv("TRAIN_HEALTH_MAX_AGE", str(max(600, 2 * INTERVAL))))


def main() -> int:
    try:
        st = os.stat(HEARTBEAT_PATH)
    except FileNotFoundError:
        # If the file hasn't been written yet (first minute), tolerate
        # a short grace period by treating as unhealthy but with distinct code.
        print("NO_HEARTBEAT")
        return 1
    age = time.time() - st.st_mtime
    if age <= MAX_AGE:
        print("OK", int(age))
        return 0
    print("STALE", int(age))
    return 1


if __name__ == "__main__":
    sys.exit(main())
