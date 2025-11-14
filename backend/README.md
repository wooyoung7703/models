# Backend (FastAPI)

Prerequisites: Python 3.10+

Install dependencies (recommended inside a venv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

## Environment variables (optional)

Core

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

Realtime prediction

| Variable | Default | Description |
|----------|---------|-------------|
| DISABLE_BACKGROUND_LOOPS | 0 | Disable long-running loops (collector/resampler/predictor) |
| PREDICT_ENABLED | 1 | Enable periodic predictor loop |
| PREDICT_INTERVAL_SECONDS | 30 | Prediction tick seconds (e.g., 10) |

Models & stacking

| Variable | Default | Description |
|----------|---------|-------------|
| ENABLE_BASE_MODELS | 0 | Enable base models (xgb/lstm/tf) if artifacts exist |
| ENABLE_STACKING | 0 | Enable stacking combiner |
| MODEL_XGB_PATH | backend/app/training/models/xgb_bottom_synth_1m_14d_cal.pkl | XGBoost artifact path |
| MODEL_LSTM_PATH | backend/app/training/models/lstm_model.pt | LSTM checkpoint path |
| MODEL_TRANSFORMER_PATH | backend/app/training/models/transformer_bottom_synth_1m_14d.pt | Transformer checkpoint path |
| STACKING_META_PATH | backend/app/training/models/stacking_meta.json | Stacking meta config |
| STACKING_THRESHOLD | -1 | Override decision threshold (>=0). If <0, use sidecar best_threshold |
| BOTTOM_TRAIN_DAYS | 14 | Shared training horizon (days of history) for bottom-detection models |
| SEQ_LEN | 30 | Sequence length for LSTM/TF |
| SEQ_MIN_READY | 10 | Minimum ready length before seq inference |
| LABEL_SCHEMA_VERSION | bottom_v1 | Version tag for bottom-label definition stored in training sidecars and stacking meta |
| ENABLE_VOL_LABELING | 0 | Enable ATR%-scaled tolerance for bottom labeling (1/0) |
| VOL_LABEL_ATR_FEATURE | atr_14 | ATR feature column name used for scaling |
| VOL_LABEL_BASE_ATR_PCT | 0.01 | Baseline ATR% (e.g., 0.01 => 1%) to normalize regimes |
| VOL_LABEL_MIN_SCALE | 0.5 | Min scale clamp for effective tolerance multiplier |
| VOL_LABEL_MAX_SCALE | 2.0 | Max scale clamp for effective tolerance multiplier |

## Run the app (development)

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs for interactive API docs.

## What happens on startup

- Collector: WebSocket connects to Binance and stores closed 1m candles with precomputed features in the `candles` table.
- Gap fill: Checks the latest stored candle and fills forward any missing 1m klines via Binance REST (upsert). If the DB is empty, it backfills the last `GAP_FILL_LOOKBACK_MINUTES` minutes.
- Models: If enabled by env and artifacts exist, base models (xgb/lstm/tf) and stacking combiner are loaded. Sequence buffers are prefilled from recent DB candles.

## Endpoints

- `GET /health` — service liveness
- `GET /health/models` — model registry readiness and details
- `GET /health/features` — recent data freshness and gaps (lightweight)
- `GET /nowcast` — latest periodic predictions per symbol (see schema below)
	- Includes a top-level `_stacking_meta` snapshot with method, threshold, and used models for convenience in UIs.
- `GET /live_price` — current in-progress candle prices captured from websocket

Admin

- `GET /admin/reload_models` — force reload of model artifacts (base + stacking) from paths in settings.
- `POST /admin/trigger_monthly_training` — immediately trigger the monthly 3-model training sequence (runs async in background).

### /nowcast schema (subset)

```jsonc
{
	"xrpusdt": {
		"symbol": "xrpusdt",
		"interval": "1m",
		"timestamp": "2025-11-11T01:40:49Z",
		"price": 2.5348,
		"price_source": "live", // or "closed" if last closed candle
		"bottom_score": 0.185,
		"components": { "rsi_14": 52.9, "bb_pct_b_20_2": 0.47, "drawdown_from_max_20": -0.009 },
		"base_probs": { "xgb": 0.17, "lstm": 0.21, "tf": 0.14 },
		"base_info": {
			"lstm": { "seq_len": 30, "feature_dim": 16 },
			"tf": { "seq_len": 30, "feature_dim": 16 }
		},
		"stacking": {
			"ready": true,
			"method": "logistic",
			"used_models": ["lstm", "tf", "xgb"],
			"prob": 0.19,
			"z": -1.02, // raw logit if available
			"threshold": 0.10,
			"threshold_source": "sidecar", // explicit | env | sidecar
			"decision": false,
			"above_threshold": false,
			"confidence": 0.09,
			"margin": 0.09 - 0.10 // prob - threshold
		}
	},
	"_stacking_meta": {
		"method": "logistic",
		"threshold": 0.10,
		"used_models": ["lstm", "tf", "xgb"],
		"threshold_source": "sidecar"
	}
}
```

Notes

- Threshold precedence (우선순위):
	1. Explicit request query param (e.g. `/nowcast?threshold=0.25`) if provided and >=0.
	2. Env override `STACKING_THRESHOLD` if >=0.
	3. Sidecar metrics `stacking_meta.metrics.json` field `best_threshold_precision` (or similar) when present.
	4. Fallback default (internally may be 0.5) if none of the above.
	The chosen source is exposed as `threshold_source` in both `stacking` block and `_stacking_meta`.
- `margin = prob - threshold` (양수일 때 의사결정 영역 상단 여유).
- `confidence`는 `(prob - threshold)`의 스케일된 형태(작게 정규화)로, 0~약 0.3 범위. 색상 레전드(frontend)는 <10% / 10~19% / ≥20%.
- `base_probs`는 모델 로드 실패 등 상황에서 null일 수 있습니다.
- `components`는 핵심 지표 및 변환값(예: s_rsi) 요약입니다.
- `_stacking_meta`는 UI 편의 제공(모델/메서드/임계치 스냅샷)으로 각 심볼 블록과 중복 필드 포함.

### Multi-symbol mode

여러 심볼을 동시에 수집/예측하려면 `.env` 또는 환경 변수에 `SYMBOLS`를 콤마로 구분해 설정합니다.

```bash
SYMBOLS=xrpusdt,btcusdt,ethusdt
EXCHANGE_TYPE=futures
INTERVAL=1m
```

Collector는 Binance combined stream을 사용하며 `/nowcast` 응답은 아래처럼 여러 키를 포함합니다:

```jsonc
{
	"xrpusdt": { /* ... */ },
	"btcusdt": { /* ... */ },
	"ethusdt": { /* ... */ },
	"_stacking_meta": { /* ... common snapshot ... */ }
}
```

Frontend는 심볼 필터/정렬 기능을 통해 다수 심볼을 표시합니다.

## Training quick links

Real-data XGBoost (bottom classification, 16 features):

```bash
python -m backend.app.training.train_xgboost \
	--mode cls_bottom --days 14 --interval 1m \
	--data-source real --use-feature-set 16 \
	--test-ratio 0.15 \
	--n-estimators 120 --max-depth 5 --learning-rate 0.08 \
	--model-out backend/app/training/models/xgb_bottom_synth_1m_14d_cal.pkl
```

Stacking (logistic, real-data, 16 features):

```bash
python -m backend.app.training.stacking \
	--data-source real --use-feature-set 16 --seq-len 30 --days 14 --interval 1m \
	--models lstm tf xgb --ensemble logistic \
	--meta-out backend/app/training/models/stacking_meta.json

Fine-tune (incremental) on recent window:

```bash
# LSTM: load last checkpoint and train 2 epochs on recent 14 days
python -m backend.app.training.train_lstm \
	--mode cls_bottom --data-source real --use-feature-set 16 --seq-len 30 --interval 1m \
	--fine-tune --ft-days 14 --ft-epochs 2 --ft-lr 5e-4 \
	--model-out backend/app/training/models/lstm_model.pt

# Transformer: similar fine-tune path
python -m backend.app.training.train_transformer \
	--mode cls_bottom --data-source real --use-feature-set 16 --seq-len 30 --interval 1m \
	--fine-tune --ft-days 14 --ft-epochs 2 --ft-lr 3e-4 \
	--model-out backend/app/training/models/transformer_model.pt
```
```

Tip: set `BOTTOM_TRAIN_DAYS` in your `.env` to change the shared window once and keep all training/stacking CLIs aligned without updating every command.

Notes on splits

- All trainers keep time order. Default is 2-way split with `--val-ratio` (e.g., 0.2 → 80/20).  
- You can enable a 3-way split by adding `--test-ratio` (e.g., 0.15). The most recent segment becomes the test holdout.  
- Test metrics are written into the `*.metrics.json` sidecars (appended after training).

Artifacts

- 모델 체크포인트: `backend/app/training/models/*.pt|*.pkl`
- 메트릭 사이드카: 동일 경로의 `*.metrics.json`
- 스태킹 메타: `stacking_meta.json` (+ `.joblib`가 있을 수 있음)

## Label Schema Versioning

- The bottom-label definition can evolve (e.g., `past_window`, `future_window`, `min_gap`, `tolerance_pct`). To keep artifacts auditable and compatible, we stamp a version into all training sidecars and stacking meta.
- Configure via env `LABEL_SCHEMA_VERSION` (default `bottom_v1`).
- Sidecars include a `label_schema` block with:
	- `version`: current label schema version
	- labeling params used for that run (if applicable)
- Stacking meta includes `label_schema_version`.
This allows detecting mismatches between live labeling and trained artifacts.

## Adaptive Volatility Labeling

- When `ENABLE_VOL_LABELING=1`, the bottom label tolerance `tolerance_pct` is scaled per-sample by the recent ATR%: `effective_tol = tolerance_pct * clamp(ATR% / VOL_LABEL_BASE_ATR_PCT, VOL_LABEL_MIN_SCALE, VOL_LABEL_MAX_SCALE)`.
- Real datasets use the configured ATR column (default `atr_14`). Synthetic datasets simulate regimes and adjust class balance accordingly.
- This helps maintain comparable event density across calm and volatile regimes.

Env summary

| Variable | Default | Description |
|----------|---------|-------------|
| ENABLE_VOL_LABELING | 0 | Enable ATR%-scaled tolerance labeling |
| VOL_LABEL_ATR_FEATURE | atr_14 | ATR feature name in DB |
| VOL_LABEL_BASE_ATR_PCT | 0.01 | Baseline ATR percent |
| VOL_LABEL_MIN_SCALE | 0.5 | Min multiplier |
| VOL_LABEL_MAX_SCALE | 2.0 | Max multiplier |

## AutoEncoder Augmentation (optional)

- When `AE_AUGMENT=1`, sequence datasets are augmented with a latent vector from a lightweight timestep AutoEncoder trained on the same tabular features. This can enrich inputs for LSTM/Transformer.
- Augmentation activates only if a saved AE model exists at `AE_MODEL_PATH` and its `input_dim` matches the current feature set size.

Env summary

| Variable | Default | Description |
|----------|---------|-------------|
| AE_AUGMENT | 0 | Enable AE latent augmentation for sequence datasets |
| AE_MODEL_PATH | backend/app/training/models/ae_timestep.pt | Path to saved AE model |

Train AE on recent real features (16-feature set example):

```bash
python -m backend.app.training.ae_train \
	--data-source real --feature-set 16 --days 30 --interval 1m \
	--latent-dim 8 --epochs 5 --batch-size 512 --lr 1e-3 \
	--out backend/app/training/models/ae_timestep.pt
```

Notes

- After training, set `AE_AUGMENT=1` and keep `AE_MODEL_PATH` pointing to the saved file. The sequence dataset builders will automatically append the latent vector to features during LSTM/Transformer training and inference.
- If the AE model is missing or the feature dimensions do not match, augmentation is skipped gracefully.

## Stacking Meta: OOF, Non-linear, Bayes

- Meta options (CLI `--ensemble` or env `STACKING_META_ENSEMBLE`): `logistic` (default), `dynamic`, `mlp`, `lgbm`, `bayes`.
- Time-ordered OOF folds: enable with `--oof-folds K` (or `STACKING_META_OOF_FOLDS`); produces unbiased meta training via expanding-window folds and evaluates on the final holdout.
- Logistic requires scikit-learn (already in requirements). `mlp` uses scikit-learn MLP. `lgbm` requires LightGBM.
- `bayes`: Bayesian-smoothed weighted logit average of base models using validation performance (AUPRC; fallback precision@top1%). Control smoothing via `--bayes-alpha` or env `STACKING_BAYES_ALPHA` (default 1.0).
- Bagging for base models: set `--bagging-n N` (or env `STACKING_BAGGING_N`) to average predictions over N seeded runs per base model.
 - Runtime override/rollback: set `STACKING_OVERRIDE_METHOD` to force a method (`logistic`, `dynamic`, `bayes`, or `mean`) regardless of what’s stored in the meta file.
## Risk & Rollback

- Runtime toggles:
	- `STACKING_OVERRIDE_METHOD`: force ensemble (`logistic|dynamic|bayes|mean`).
	- `STACKING_THRESHOLD` (>=0): override threshold (precedence: explicit > env > sidecar).
	- `ENABLE_CALIBRATION`, `ENABLE_PROB_SMOOTHING`, `ENABLE_ADAPTIVE_THRESHOLD`: disable to revert to raw probability + static threshold.
- Training toggles:
	- `--ensemble`, `--bagging-n`, `--bayes-alpha`, `--oof-folds`, `--use-ordinal`, `--use-feature-set`.
	- `--quick-refit`: recalibrate recent window only.
- Versioning in `stacking_meta.json`:
	- Captures `oof_folds`, `bagging_n`, `bayes_alpha`, `use_ordinal`, `interval`, `seq_len`, `val_ratio`, `data_source`, and feature set info to aid rollback and audit.


Optional install for LightGBM:

```bash
pip install lightgbm
```

Example (5-fold OOF, MLP meta):

```bash
STACKING_META_ENSEMBLE=mlp STACKING_META_OOF_FOLDS=5 \
python -m backend.app.training.stacking \
	--data-source real --use-feature-set 16 --seq-len 30 --days 14 --interval 1m \
	--models lstm tf xgb --val-ratio 0.2 --meta-out backend/app/training/models/stacking_meta.json
```

Example (Bayes ensemble + bagging over base models):

```bash
STACKING_META_ENSEMBLE=bayes STACKING_BAYES_ALPHA=1.0 STACKING_BAGGING_N=3 \
python -m backend.app.training.stacking \
	--data-source real --use-feature-set 16 --seq-len 30 --days 14 --interval 1m \
	--models lstm tf xgb --val-ratio 0.2 \
	--bagging-n 3 --bayes-alpha 1.0 \
	--meta-out backend/app/training/models/stacking_meta.json
```

## Threshold EV Optimizer

- Optimize the decision threshold by maximizing realized expected value (PnL) over recent trades.
- Provide data via CSV/JSON (`prob,pnl`) or read from the DB (closed trades with `strategy_json.prob` and `pnl_pct_snapshot`).

Usage (CSV):

```bash
python -m backend.app.training.threshold_ev \
	--source csv --input path/to/recent_trades.csv \
	--t-low 0.50 --t-high 0.995 --steps 60 \
	--min-coverage 0.005 --min-trades 5 \
	--write-to-meta backend/app/training/models/stacking_meta.json
```

Usage (DB, if `pnl_pct_snapshot` and entry prob are stored):

```bash
python -m backend.app.training.threshold_ev \
	--source db --days 30 --symbol xrpusdt --exchange-type futures \
	--write-to-meta backend/app/training/models/stacking_meta.json
```

Notes

- The CLI appends an `ev_opt` block in `stacking_meta.metrics.json` with `best_threshold_ev` and summary stats. At runtime you may choose this value (precedence remains configurable).
- If DB records lack usable `(prob,pnl)`, prefer the CSV/JSON path by exporting your trade log.

## Monthly Training (자동 학습)

- The backend runs a scheduler that, every month on the 1st at 00:00, trains three base models sequentially (XGBoost, LSTM, Transformer), then trains the stacking meta-ensemble. All runs use real-data features. Artifacts are written to the paths in settings:
	- Base: `MODEL_XGB_PATH`, `MODEL_LSTM_PATH`, `MODEL_TRANSFORMER_PATH`
	- Stacking meta: `STACKING_META_PATH`
- After training completes, the in-memory model registry is reloaded so new artifacts are used immediately (base + stacking).

### 거래 종료 기반 즉시 재학습 (Event-driven Retrain)

- 설정 `RETRAIN_ON_TRADE_CLOSE=1` 이면 어떤 거래가 **익절(TP)** 또는 **손절(SL)** 로 `status=closed` 되는 순간 쿨다운을 만족할 경우 즉시 동일한 시퀀스( XGBoost → LSTM → Transformer → Stacking )를 다시 실행합니다.
- 최소 간격은 `RETRAIN_MIN_INTERVAL_HOURS` (기본 12시간) 으로 제어하며, 이전 재학습 완료 시각 기준입니다.
- 중복 실행 방지: 실행 중(active)일 때는 추가 트리거 무시됩니다.
- 트리거 로직은 `trade_manager.py` 에서 close 이벤트 후 `scheduler.trigger_trade_close_retrain()` 호출로 구현되어 있습니다.

환경변수 요약 (추가)

| Variable | Default | Description |
|----------|---------|-------------|
| RETRAIN_ON_TRADE_CLOSE | 0 | 거래 종료(TP/SL) 시 자동 재학습 시도 (1/0) |
| RETRAIN_MIN_INTERVAL_HOURS | 12 | 마지막 재학습 이후 최소 대기 시간(시간 단위) |
| RETRAIN_DAYS | BOTTOM_TRAIN_DAYS/2 (>=14) | 이벤트(거래 종료) 재학습용 베이스 모델 학습 기간 |
| STACKING_META_DAYS | max(45, BOTTOM_TRAIN_DAYS) | 월간/수동 재학습 시 스태킹 메타 학습 기간 |
| RETRAIN_STACKING_DAYS | (optional) | 이벤트 재학습 시 스태킹 메타 기간 (미설정 시 STACKING_META_DAYS) |
### 운영 매트릭스 (권장 윈도우 전략)

| 상황 | Base(XGB/LSTM/TF) days | Stacking days | 목적 |
|------|------------------------|---------------|------|
| 초기(Cold Start) | 30~45 | 45 | 빠른 가동/안정화 |
| 월간 재학습 | 60~90 | 60~90 또는 45 | 희귀 이벤트 축적 + 안정성 |
| 이벤트(거래 종료) | RETRAIN_DAYS (기본 절반) | RETRAIN_STACKING_DAYS (기본 STACKING_META_DAYS) | 빠른 적응/드리프트 대응 |
| 급격한 시장 변화 | 30 + 최근 14일 가중 | 30~45 | 민감도 재조정 |
| 스태킹 재캘리브레이션만 | (변경 없음) | 30 | 임계치/보정 최적화 |

설정 요약:
- 월간: `BOTTOM_TRAIN_DAYS`, `STACKING_META_DAYS`
- 이벤트: `RETRAIN_DAYS`, `RETRAIN_STACKING_DAYS` (없으면 `STACKING_META_DAYS`)
- 공통: `RETRAIN_MIN_INTERVAL_HOURS` 로 과도한 반복 제한


수동 트리거와의 관계
- `/admin/trigger_monthly_training` 은 언제든 즉시 실행 요청(수동 / 테스트 용).
- 이벤트 트리거는 쿨다운 로직을 통과해야 실행됩니다.


Controls

- Optional timezone: set `SCHEDULER_TZ` (e.g., `Asia/Seoul` or `UTC`). If not set, APScheduler's default timezone is used.
- Manual trigger for testing:

```bash
curl -X POST http://localhost:8000/admin/trigger_monthly_training
```

- Manual reload of models (if you replace artifacts out-of-band):

```bash
curl http://localhost:8000/admin/reload_models
```

Implementation notes

- The scheduler invokes the following module CLIs internally:
	- `python -m backend.app.training.train_xgboost --mode cls_bottom --data-source real --use-feature-set 16 --model-out $MODEL_XGB_PATH`
	- `python -m backend.app.training.train_lstm --mode cls_bottom --data-source real --use-feature-set 16 --seq-len $SEQ_LEN --model-out $MODEL_LSTM_PATH`
	- `python -m backend.app.training.train_transformer --mode cls_bottom --data-source real --use-feature-set 16 --seq-len $SEQ_LEN --model-out $MODEL_TRANSFORMER_PATH`
	- `python -m backend.app.training.stacking --data-source real --use-feature-set 16 --seq-len $SEQ_LEN --days $BOTTOM_TRAIN_DAYS --interval $INTERVAL --models lstm tf xgb --ensemble logistic --meta-out $STACKING_META_PATH`
	- Shared days window: `BOTTOM_TRAIN_DAYS` (env) and interval `INTERVAL`.
	- Split is time-ordered with `--val-ratio 0.2`; test holdout is disabled by default for the automated run.


