# Backend (FastAPI)

Prerequisites: Python 3.10+

Install dependencies (recommended inside a venv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Environment variables (optional):

| Variable | Default | Description |
|----------|---------|-------------|
| DB_URL | sqlite:///backend/data/data.db | Database URL (SQLite or Postgres) |
| SYMBOL | xrpusdt | Binance symbol lowercase |
| INTERVAL | 1m | Kline interval |
| EXCHANGE_TYPE | spot | spot or futures |
| RECONNECT_MIN_SEC | 1 | Min reconnect backoff |
| RECONNECT_MAX_SEC | 30 | Max reconnect backoff |
| GAP_FILL_ENABLED | 1 | Startup gap fill enabled (1/0 or true/false) |
| GAP_FILL_LOOKBACK_MINUTES | 1440 | If DB empty, minutes to backfill at startup (default 1 day) |

Run the app (development):

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs for interactive API docs.

Collector: On startup a websocket connects to Binance and stores closed 1m candles with precomputed features in the `candles` table.

Gap fill: On startup (before starting the realtime collector), the app checks the latest stored candle and fills forward any missing 1m klines up to now via Binance REST, computing the same features and upserting by unique index. If the DB is empty, it backfills the last `GAP_FILL_LOOKBACK_MINUTES` minutes.

