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
# Frontend: http://localhost:8080
# Backend (optional direct access): http://localhost:8000/health
# Restart services
docker compose restart backend; docker compose restart frontend
docker compose down
### Dev mode (hot backend reload, loops off)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
docker compose logs -f backend
Or use helper scripts (Windows PowerShell):
## Docker (WebSocket-first runtime)

The current preferred runtime is a standalone WebSocket backend (no FastAPI HTTP layer) plus an Nginx-served frontend.

Ports:
- Backend WS: `8022` (mapped to host `8022`)
- Frontend (Nginx static SPA): `8080`

### Build & Run (production style)

```bash
# From repo root
docker compose build --no-cache
docker compose up -d

# Check
docker compose ps
docker compose logs -f backend
docker compose logs -f frontend
```

Open: `http://localhost:8080` (frontend will attempt `ws://<host>:8022` by default).

Check container health (backend):
```bash
docker inspect --format='{{json .State.Health}}' models-backend
```
Status should show `"Status":"healthy"` once WS port is accepting connections.

### Stopping & Cleanup
```bash
Environment

```

### Runtime configuration (key env vars)
Set in `docker-compose.yml` under the `backend` service (or override with `docker compose run -e VAR=...`):
- `EXCHANGE_TYPE=spot|futures`
- `LIGHT_MODE=1` (skip heavy models)
- `DISABLE_BACKGROUND_LOOPS=1` (disable collector/resampler/predictor)
- `STACKING_THRESHOLD` (force threshold override)
- `LOG_LEVEL=INFO|DEBUG`
- `WS_PORT` (default 8022; change mapping accordingly)

To change exposed port:
```yaml
services:
	backend:
		ports:
			- "8123:8123"    # host:container
		environment:
			- WS_PORT=8123
```
Update frontend WS URL at runtime by injecting a `config.js` (already created by Docker build) or overriding with an env when running dev locally:
```bash
VITE_WS_URL=ws://127.0.0.1:8123 npm run dev
```

### Logs & Restart
```bash
```bash
python -m venv .venv
source .venv/bin/activate
```

### Dev iteration (optional)
For rapid local development you can still use the legacy FastAPI + auto-reload path:
```bash
pip install -r backend/requirements.txt
```
However, the default production `backend/Dockerfile` runs `python -m backend.app.ws_server` (pure WebSocket). Prefer that unless you need HTTP endpoints.

If you need live code edits for the WS server, mount the source and run directly on the host (simpler than container reload loops):
```bash
LIGHT_MODE=1 python -m backend.app.ws_server
```

### Port already in use (Windows errno 10048)
If you see: `OSError: [Errno 10048] ... address already in use`
```bash
netstat -ano | grep :8022
taskkill //PID <PID> //F
```
Then restart: `docker compose up -d backend`.

### Common commands
```bash
# Rebuild only frontend
docker compose build frontend
# Rebuild only backend
docker compose build backend
# Tail both
docker compose logs -f --tail=100 backend frontend
```

### Frontend runtime API / WS config
`frontend/nginx.conf` serves the SPA only (no API proxy). The client is WS-first. For production behind reverse proxies, ensure port 8022 is reachable, or set `window.VITE_WS_URL` in `/usr/share/nginx/html/config.js` inside the container.

Example override after container start:
```bash
```
```

### Data persistence
SQLite / snapshots stored in volume `models-backend-data` mapped to `/app/backend/data`.
Backup:
```bash

```

### Legacy FastAPI endpoints
Still available if you run the dev compose (port 8000) but not required for WebSocket operation. The frontend falls back to HTTP polling only if WS fails during dev mode.

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

### 실거래 타협안 구성 (Compromise Live Mode)

스태킹 + Entry Meta 게이트 활성 / 고정 TP(+0.8% 순익) / 무손절 정책을 기본 실행값으로 적용한 **타협안** 런타임.

핵심 변경점:
- `TAKE_PROFIT_PCT=0.008` (순익 +0.8% 고정 목표)
- `TP_MODE=fixed` (트레일링 대신 고정)
- `TP_DECISION_ON_NET=1` (수수료 반영 후 순익 기준으로 TP 판정)
- `DISABLE_STOP_LOSS=1` (손절 비활성 → DCA 관리 중요)
- `ENABLE_ENTRY_META=1`, `ENTRY_META_GATE_ENABLED=1` (정밀도 게이트 항상 사용)

PowerShell 실행 (WS 서버):
```powershell
./start_backend_compromise.ps1
```

Git Bash / Linux 등 셸 직접 실행:
```bash
EXCHANGE_TYPE=futures \
ENABLE_ENTRY_META=1 ENTRY_META_GATE_ENABLED=1 \
DISABLE_STOP_LOSS=1 TAKE_PROFIT_PCT=0.008 TP_MODE=fixed TP_DECISION_ON_NET=1 \
python -m backend.app.ws_server
```

선택적 리스크 제어(꼬리 DCA 완화):
```bash
ADD_COOLDOWN_SECONDS=900 \
EXCHANGE_TYPE=futures ENABLE_ENTRY_META=1 ENTRY_META_GATE_ENABLED=1 \
DISABLE_STOP_LOSS=1 TAKE_PROFIT_PCT=0.008 TP_MODE=fixed TP_DECISION_ON_NET=1 \
python -m backend.app.ws_server
```

운용 모니터링 포인트:
운용 모니터링 포인트:
- DCA 추가 빈도 과도 시: `ADD_COOLDOWN_SECONDS` 상향으로 속도 제어
- 승률 100% 구간 지속 시: TP 목표 추가 상향(예: 0.009) 테스트 가능
- DCA 꼬리 리스크 높을 경우 쿨다운↑ 또는 Entry 게이트 강화 (threshold 상향)

롤백 방법(기본 트레일링 모드로 복귀):
```bash
EXCHANGE_TYPE=futures ENABLE_ENTRY_META=1 ENTRY_META_GATE_ENABLED=1 \
DISABLE_STOP_LOSS=0 TAKE_PROFIT_PCT=0.005 TP_MODE=trailing TP_DECISION_ON_NET=0 \
python -m backend.app.ws_server
```

> 참고: 코드 기본값(`core/config.py`)은 이미 타협안으로 업데이트되어 환경변수 미설정 시에도 해당 정책이 적용됩니다.

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

## 자동 메타 재학습 (Bottom vs Forecast 확률 보정)

실시간 스태킹 확률(`bottom_prob`)을 미래 예측(fwd_pred_*) 기반 특징으로 2차 로지스틱으로 재보정하는 **Bottom vs Forecast Meta** 레이어가 주기적으로 자동 재학습됩니다.

### 동작 개요
1. 평가 CSV(`BOTTOM_VS_FORECAST_EVAL_CSV_PATH`)에서 `bottom_prob`, `bottom_label`, `close`, `fwd_pred_1..H`를 읽습니다.
2. 특징 생성: `room_to_forecast_min`, `rel_to_forecast_mean`, `forecast_expected_return`.
3. 순수 NumPy 로지스틱(gradient descent)으로 계수 재추정.
4. 기존 메타 JSON(`BOTTOM_VS_FORECAST_META_PATH`)과 Brier score 비교 → 상대 개선율(rel_improve)이 `META_RETRAIN_MIN_REL_BRIER_IMPROVE` 이상일 때만 overwrite.
5. 성공 시 레지스트리 강제 reload → 새 계수 즉시 반영.

### 스케줄 방식 (환경 변수)
- `META_RETRAIN_ENABLED=1` 활성화.
- 주기 선택:
	- `META_RETRAIN_EVERY_MINUTES>0` → Interval Job (분 단위 반복)
	- 그렇지 않고 `META_RETRAIN_DAILY_AT=HH:MM` 설정 시 → Daily Job
	- 둘 다 미지정 시 자동 잡 없음 (수동 실행만 가능)

### 쿨다운 & 개선 조건
- `META_RETRAIN_MIN_INTERVAL_MINUTES`: 마지막 성공/실패 시점부터 최소 대기 분.
- `META_RETRAIN_MIN_REL_BRIER_IMPROVE`: 기존 대비 Brier 상대 개선율 최소치 (예: 0.01 = 1% 개선 필요). 0 이면 항상 overwrite.

### 경로 관련
- `BOTTOM_VS_FORECAST_EVAL_CSV_PATH`: 재학습 입력 평가 CSV.
- `BOTTOM_VS_FORECAST_META_PATH`: 계수 JSON (in-place overwrite).

### 로그
스케줄러(job) 출력 예:
```
[meta-retrain] start csv=... meta=... min_rel=0.0100
[meta] wrote new meta -> data/bottom_vs_forecast_meta.json
[meta-retrain] exit code=0 overwritten=True
```
또는 개선 부족 시:
```
[meta] NOT overwritten (relative improvement insufficient)
[meta-retrain] exit code=0 overwritten=False
```

### 수동 실행
```bash
python -m scripts.retrain_bottom_vs_forecast_meta \
	--csv data/bottom_eval_sample.csv \
	--current-meta data/bottom_vs_forecast_meta.json \
	--out-meta data/bottom_vs_forecast_meta.json \
	--min-rel-brier-improve 0.01
```

### 런타임 반영 확인
메타 업데이트 후 실시간 확률 구조에서 `prob_final_base` 대비 조정 확률(`prob_final`) 및 델타 안정화 지표(Δ mean_50/200)가 정상적으로 갱신되는지 모니터링하십시오.

### 운영 팁
- Interval이 너무 짧으면(예: 5분) 개선율이 소폭 → 빈번한 overwrite로 변동성 증가 가능. 초기엔 60~180분 권장.
- Daily 모드 사용 시 장기 drift 또는 분포 shift 감지와 병행(확률 분포 감시)하면 안전.
- 개선율이 장기간 0% 근처라면 평가 CSV 품질(레이블, forward 예측 신뢰도) 재점검.

### 메타 재학습 모니터링 확장 지표
웹소켓 `training_status` 메시지에 다음 필드들이 포함되어 대시보드에서 시각화됩니다:

| 필드 | 의미 | 해석 가이드 |
|------|------|-------------|
| `meta_history[]` | 최근 재학습 레코드 (최대 40개 방송, 200개 롤링 저장) | 각 항목: `ts`, `brier_old`, `brier_new`, `rel_improve`, `overwritten`, `samples` |
| `rel_improve` | (Brier_old - Brier_new) / Brier_old | 양수면 개선, 음수면 악화 |
| `samples` | 재학습에 사용된 샘플 수 | 너무 작을 경우(예: < 500) 통계적 변동성 ↑ |
| `meta_corr_rel_improve_samples` | Pearson 상관 (ΔBrier 개선율 vs 샘플 수) | >0: 더 많은 샘플일수록 개선율 증가 경향; <0: 샘플 많아도 개선 어려움 |
| `meta_neg_streak` | 연속 음수/미반영(개선 실패) 횟수 | 임계 초과 시 분포 변화/데이터 품질 점검 필요 |
| `meta_neg_streak_warn` | 경고 플래그 (`meta_neg_streak` ≥ 임계) | UI에서 강조/아이콘 표시 |
| `meta_reg_slope` | 단순 선형회귀 slope (rel_improve ~ samples) | 양수: 샘플 증가시 개선율 완만 상승; 절대값 매우 작으면 영향 미미 |
| `meta_reg_pvalue` | slope 유의성 2-측 p-value (정규 근사) | ≤0.05: 추세 유의 가능성 높음; >0.15: 불확실 |
| `meta_reg_window` | 회귀 계산에 사용한 최근 레코드 개수 | 기본 30; 데이터 적으면 slope/p-value 제공 안함 |
| 동적 쿨다운 | 음수 연속 스택 × `META_BACKOFF_FACTOR` | 쿨다운 배수 적용 후 재시도 간격 증가 |
| 롤백 여부 | streak 경고 + 롤백 활성 | 마지막 양의 개선 스냅샷 파일로 복구 |
| `meta_coef_cosine` | 현재 vs 마지막 양의 개선 메타 계수 코사인 유사도 | 1에 가까울수록 안정; 낮으면 계수 방향 급변 가능성 |
| `meta_coef_drift_warn` | 계수 유사도 경고 플래그 | 유사도 < `META_COEF_DRIFT_WARN_COS` 시 True |

#### 음수 스택(neg streak) 정의
다음 두 조건 중 하나라도 만족하면 "음수/실패"로 간주하여 streak 계산에 포함:
1. `rel_improve <= 0` (개선율 비양수)
2. `overwritten == False` (개선 충분치 않아 메타 파일 덮어쓰기 실패)

#### 추가 환경 변수
| Env | 기본값 | 설명 |
|-----|--------|------|
| `META_NEG_STREAK_WARN` | 3 | 연속 음수/실패 재학습 경고 임계치 |
| `META_REG_WINDOW` | 30 | 회귀/상관/추세 계산에 사용할 최근 레코드 최대 개수 |
| `META_BACKOFF_FACTOR` | 0.5 | 음수 streak 당 추가 배수 (min_interval * (1 + streak*factor)) |
| `META_NEG_STREAK_ROLLBACK` | 0 | 1이면 경고 임계 초과 시 마지막 양의 개선 메타로 롤백 |
| `META_LAST_GOOD_META_PATH` | data/bottom_vs_forecast_meta_last_good.json | 양의 개선 시 저장될 스냅샷 경로 |
| `META_REG_USE_NORMAL_APPROX` | 1 | 0이면 df<=40 시 mpmath 이용 t-분포 p-value 계산 시도 |
| `META_COEF_DRIFT_WARN_COS` | 0.92 | 코사인 유사도 임계치 (이하일 때 드리프트 경고) |

#### 활용 시나리오
1. `meta_neg_streak_warn` 활성 + `meta_corr_rel_improve_samples < 0`: 데이터 분포가 바뀌었을 가능성 → 평가 CSV 라벨/forward 예측 재검증.
2. `meta_reg_slope > 0` & `meta_reg_pvalue <= 0.05`: 더 많은 샘플 축적이 개선율을 지속 향상 → Interval 단축 가능성 검토.
3. `meta_reg_slope ~ 0` & `rel_improve` 장기적으로 0% 근처: 메타 특징(예: forecast expected return) 정보력 약화 → 새로운 forward feature 후보 실험.
4. `meta_corr_rel_improve_samples` 높음(>0.6) + 작은 샘플 러닝: 더 긴 CSV 유지 기간(`META_EVAL_KEEP_DAYS`) 늘려 안정성 확보.
5. 음수 streak 증가로 동적 쿨다운이 크게 확대된 경우: Interval 자체를 낮추는 대신 데이터 품질 개선 또는 메타 특징 강화(새 forward feature) 우선.
6. `meta_coef_drift_warn` 활성 + 개선율 저하: 로지스틱 결정 경계 급변 → 원본 특징 분포 혹은 forward 예측 스케일 변화 여부 확인.

#### 권장 후속 액션 (경고 발생 시)
- 최근 N개(예: 10개) 레코드만 별도 CSV 필터하여 개선율 추세 수동 검증.
- forward 예측 모델 자체 재학습/검증 (예측 분포 변화 여부 확인).
- 평가 CSV 샘플 추출 → 극단값/레이블 누락 비율 점검.
- 롤백 발생 시: 롤백된 메타(JSON)과 최신 실패 메타의 계수/특징 중요도 차이 비교 후 원인 파악.
- 계수 드리프트 경고 시: 현재/스냅샷 계수 벡터 코사인, L2 norm, 개별 특징 계수 비율(bottom_prob 대비 room/rel/exp_ret) 비교로 특정 특징 과잉 적응 여부 진단.

#### CSV 내보내기 (프론트)
대시보드 "CSV 내보내기" 버튼은 현재 `meta_history` 전체를 `meta_retrain_history.csv`로 다운로드합니다. 추가 오프라인 분석(예: rolling z-score, quantile band)을 위해 바로 활용하세요.

#### 향후 확장 아이디어
- t-분포 기반 정확한 p-value (소표본)
- 음수 streak 자동 백오프 (Interval 동적 증가)
- 이전 "최적" 메타 버전 자동 롤백 (개선율 급락 시)
- 다변량 회귀 (개선율 ~ samples + 최근 변동성 지표 등)
 - 메타 계수 드리프트 감시 (벡터 코사인 유사도 기반 경고)

