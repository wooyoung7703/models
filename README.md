# Models (monorepo)

This repository contains a minimal scaffold for a Python backend and a Vue 3 + TypeScript frontend.

Structure:

- backend/: FastAPI app
- frontend/: Vite + Vue 3 + TypeScript app

Quick start (backend):

```bash
# from repo root
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

Quick start (frontend):

```bash
cd frontend
npm install
npm run dev
```

Notes:
- These are minimal starter files to get you going. Restrict CORS and pin production dependency versions before deploying.

## Docker

Build and run both backend and frontend with Docker Compose:

```powershell
# From repo root
docker compose build
docker compose up -d

# Frontend: http://localhost:8080
# Backend (optional direct access): http://localhost:8000/health
```

Details
- `frontend`: Built with Node, served via Nginx on port 8080.
	- Nginx proxies `/api/*` → `backend:8000/*`.
	- A small `/config.js` sets `window.VITE_API_BASE='/api'` at runtime.
- `backend`: FastAPI on port 8000, started by `uvicorn`.
	- `./backend/data` is bind-mounted to persist SQLite and snapshots.
	- Toggle loops via `DISABLE_BACKGROUND_LOOPS=0|1` in `docker-compose.yml`.

Common commands
```powershell
# View logs
docker compose logs -f backend
docker compose logs -f frontend

# Restart services
docker compose restart backend; docker compose restart frontend

# Stop and remove
docker compose down
```

### Dev mode (hot backend reload, loops off)

```powershell
# Build once
docker compose build

# Start with dev override (backend: DEBUG, --reload, loops off, code mounted)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Logs
docker compose logs -f backend
```

Or use helper scripts (Windows PowerShell):

```powershell
# Prod-like
scripts\up.ps1

# Dev (reload + DEBUG)
scripts\up-dev.ps1

# Tear down
scripts\down.ps1
```

## ML training quickstart

Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Train (examples)

```bash
# LSTM (classification: bottom)
python -m backend.app.training.train_lstm --mode cls_bottom --epochs 2 --tb --log-dir runs

# Transformer (classification)
python -m backend.app.training.train_transformer --mode cls_bottom --epochs 2 --tb --log-dir runs

# XGBoost (classification)
python -m backend.app.training.train_xgboost --mode cls_bottom --n-estimators 120 --tb --log-dir runs

# Stacking (single-split, logistic meta)
python -m backend.app.training.stacking --models lstm tf xgb --ensemble logistic --tb

# OOF Stacking (time-series folds)
python -m backend.app.training.stacking_oof --models lstm tf xgb --folds 3 --ensemble logistic --tb

# Stacking inference
python -m backend.app.training.stacking_infer \
	--meta-config backend/app/training/models/stacking_meta.json \
	--base-probs lstm=/path/lstm.csv tf=/path/tf.csv xgb=/path/xgb.csv \
	--output /tmp/pred.csv

# HPO (Optuna)
python -m backend.app.training.hpo --model lstm --mode cls_bottom --trials 10 --epochs 2
```

Key flags (classification)

- `--calibration {none,platt,isotonic}`: 확률 보정
- `--min-coverage`: 정밀도 최적화 임계치 탐색 시 최소 커버리지 제약
- `--regime-filter low_vol` + `--regime-percentile`: 검증 메트릭에 저변동성 구간 적용
- `--t-low`, `--t-high`: 더블 임계치 지표 기록
- `--tb --log-dir`: TensorBoard 로깅 활성화

Artifacts

- 모델 체크포인트: `backend/app/training/models/*.pt|*.pkl`
- 메트릭 사이드카: 동일 경로의 `*.metrics.json`
- 스태킹 메타: `stacking_meta.json` (+ `.joblib`가 있을 수 있음)

### 실데이터 기반 스태킹 설정 (Option C)

`.env` (백엔드) 주요 항목:

```env
MODEL_TYPE=stacking
ENABLE_BASE_MODELS=1
ENABLE_STACKING=1
MODEL_XGB_PATH=backend/app/training/models/xgb_bottom_real_1m.pkl
MODEL_LSTM_PATH=backend/app/training/models/lstm_bottom_real_1m.pt
MODEL_TRANSFORMER_PATH=backend/app/training/models/transformer_bottom_real_1m.pt
STACKING_META_PATH=backend/app/training/models/stacking_meta.json
STACKING_THRESHOLD=0.95   # 커버리지 확대 위해 sidecar (≈0.978)보다 낮춤
SEQ_LEN=30
```

Threshold 정책:

1. 환경변수(`STACKING_THRESHOLD`)가 0 이상이면 최우선 사용
2. 없으면 sidecar(`*.metrics.json`)의 `best_threshold_precision` 활용
3. 없으면 디폴트(결정 없음 -> 프론트엔드에서 `decision` False)

### 진입 조건 (TradeManager)

`backend/app/trade_manager.py` 내 `EntryConfig` (완화된 값):

- `min_margin=0.01` (스태킹 확률 - 임계값)
- `min_conf=0.08` (실질적으로 margin 절대값)
- `min_bottom=0.55` (휴리스틱 바텀 점수)
- `min_z=1.2` (스태킹 raw logit 강도)

향후 튜닝:

- 거래 없음 → threshold 추가 하향 (예: 0.93)
- 거래 과다 / 승률 급락 → threshold 재상향 또는 margin 복구
- 동적 가중치 실험: `--ensemble dynamic` 으로 OOF 재생성

### OOF 스태킹 (실데이터)

예시 실행 (60일, dynamic 가중치):

```bash
python -m backend.app.training.stacking_oof \
	--days 60 --seq-len 30 --interval 1m \
	--models lstm tf xgb --folds 3 --ensemble dynamic \
	--data-source real --min-coverage 0.01 \
	--meta-out backend/app/training/models/stacking_meta.json
```

Sidecar 메트릭에서 활용 가능한 항목: `precision_at_best_t`, `coverage_at_best_t`, `double_t_low/high`.

### 백테스트

30일: `python backend/scripts/backtest_last_month.py`

90일 변형 필요 시 스크립트를 복제하여 `--days 90` 인자 추가 구현.

### 튜닝 체크리스트

1. threshold 조정 (정밀도 vs 커버리지)
2. EntryConfig 조정 (margin / bottom / z)
3. 데이터 기간 확장 (21d → 60d → 90d) 재학습
4. base 모델 재학습 시 class imbalance 보정 (pos_weight, focal loss 등)
5. precision-recall 곡선 기반 효용 최적화 (향후)


Troubleshooting

- CUDA가 없거나 설치되지 않은 경우 `--device cpu` 사용
- `xgboost`/`sklearn` 관련 경고는 테스트/로컬 실행에 영향 없는 경우가 많습니다
- 테스트는 `pytest`로 실행합니다: `python -m pytest -q`
