# Copilot Instructions — Models (WS-first)

This repo runs a WebSocket-only backend for realtime updates and a Vue 3 frontend. FastAPI code remains for reference/admin but is not the default runtime path.

## Runtime Overview
- Backend: `backend/app/ws_server.py` (pure `websockets`; no HTTP).
  - Starts Binance collectors, gap fill, resampler, predictors, and a WS server on `ws://0.0.0.0:8022`.
  - Broadcasts messages:
    - `snapshot`: `{ nowcast, features, trades }` (initial payload)
    - `nowcast`: per-symbol updates
    - `trades`: compact trade list with cooldown meta
    - `features`: incremental feature-health deltas
- Frontend: `frontend` (Vite + Vue 3)
  - Connects by default to `ws://<host>:8022` with optional `VITE_WS_URL` override.
  - Falls back to HTTP polling against `apiBase` only if WS fails (HTTP endpoints are provided by `FastAPI` if you still run it).

## Start Locally (Windows)
Use bash (Git Bash) or PowerShell. WS backend is a plain Python process.

Bash:
```bash
cd /c/Users/wooyo/models
python -m venv .venv
source .venv/Scripts/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
# Minimal mode (skip heavy models):
LIGHT_MODE=1 EXCHANGE_TYPE=futures DISABLE_BACKGROUND_LOOPS=0 \
  python -m backend.app.ws_server
```

PowerShell:
```powershell
cd C:\Users\wooyo\models
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r backend\requirements.txt
$env:LIGHT_MODE='1'; $env:EXCHANGE_TYPE='futures'; $env:DISABLE_BACKGROUND_LOOPS='0'
python -m backend.app.ws_server
```

Frontend (separate terminal):
```bash
cd /c/Users/wooyo/models/frontend
npm ci || npm install
npm run dev
```
Open http://127.0.0.1:5173.

Tip: If backend runs on a different host/port, start the frontend with:
```bash
VITE_WS_URL=ws://127.0.0.1:8022 npm run dev
```

## Docker
```powershell
# From repo root
docker compose build --no-cache
docker compose up -d
# Frontend: http://localhost:8080
# WS backend: ws://localhost:8022
```
Note: Existing compose files still include a FastAPI service; preferred dev flow is WS server + vite dev.

## Tasks in VS Code
- `Run frontend (vite dev)`: starts the Vue dev server.
- For backend, prefer running `python -m backend.app.ws_server` in a terminal. Existing Uvicorn tasks target FastAPI and are optional.

## Configuration
Key env vars (see `backend/app/core/config.py`):
- `EXCHANGE_TYPE`: `spot|futures` (default `spot`)
- `SYMBOLS`: comma-separated list (defaults to `SYMBOL`)
- `INTERVAL`: bar interval (default `1m`)
- `WS_OPEN_TIMEOUT_SECONDS`: Binance WS open timeout (default `15`)
- `DISABLE_BACKGROUND_LOOPS`: `1` to disable collector/resampler/predictor (default `0`)
- `PREDICT_ENABLED`: enable periodic predictor broadcasts (default `1`)
- `RESAMPLER_INTERVAL_SECONDS`: build `5m/15m` aggregates (default `55`)
- `FEATURE_SNAPSHOT_EVERY`: periodic snapshot cadence (default `50`)
- `LIGHT_MODE`: `1` to skip model registry/predictor/trade manager for lightweight runs
- Stacking/trading: `STACKING_THRESHOLD`, `TP_MODE`, `TP_TRIGGER`, `TP_STEP`, `TP_GIVEBACK`, `ADD_COOLDOWN_SECONDS`

Frontend WS resolution (in `App.vue`):
- Tries `ws://<host>:8022` first.
- Accepts overrides `VITE_WS_URL` (env) or `window.VITE_WS_URL`.
- Falls back to HTTP polling if WS fails.

## Development Notes
- Pydantic v1/v2: use compatibility helpers that check `model_fields`/`__fields__`/`__annotations__` where needed.
- SQLModel/SQLAlchemy: keep queries typed; use `sqlalchemy.text` for explicit order-by labels.
- Long-running loops: respect `DISABLE_BACKGROUND_LOOPS` to keep tests fast.
- Error logs: collector uses throttled logging and explicit connect open timeout.
- Snapshots: on shutdown, feature calculator snapshots are persisted.

## Troubleshooting
- WS connect fails from frontend:
  - Ensure backend WS is running on `8022` and reachable.
  - Start frontend with `VITE_WS_URL=ws://127.0.0.1:8022 npm run dev`.
- Binance WS timeouts:
  - Adjust `WS_OPEN_TIMEOUT_SECONDS` and verify DNS/network.
- Terminal stuck PowerShell session:
  - Kill Uvicorn/FastAPI processes if any linger: use the provided `Kill Uvicorn 8000` task or `Get-CimInstance Win32_Process | where { $_.Name -match 'python' -and $_.CommandLine -match 'uvicorn backend.app.main:app' } | % { Stop-Process -Id $_.ProcessId -Force }`.
- Exit code 127 in bash when starting venv:
  - Ensure you typed `cd` (not `d`) before the path; re-run the full command block.

## When editing code (for agents)
- Prefer surgical changes; keep public APIs and file names stable unless required.
- Use `apply_patch` for edits; keep diffs narrow and avoid unrelated reformatting.
- Update only relevant docs; don’t add license headers.
- Keep heavy dependencies out of minimal/light paths; honor `LIGHT_MODE`.
- Verify with targeted runs or `pytest` for changed modules (see `backend/tests`).
