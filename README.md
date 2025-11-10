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

Troubleshooting

- CUDA가 없거나 설치되지 않은 경우 `--device cpu` 사용
- `xgboost`/`sklearn` 관련 경고는 테스트/로컬 실행에 영향 없는 경우가 많습니다
- 테스트는 `pytest`로 실행합니다: `python -m pytest -q`
