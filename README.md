# models

암호화폐 가격 예측을 위한 머신러닝 모델 저장소

## 개요

이 프로젝트는 XGBoost, LSTM, Transformer 등의 머신러닝 모델을 사용하여 암호화폐 가격(특히 low 가격)을 예측합니다.

## 기능

- ✅ **XGBoost 모델 훈련**: 암호화폐 1분봉 데이터를 사용한 모델 훈련
- ✅ **예측 기능**: 훈련된 모델을 사용한 가격 예측
- ✅ **데이터 로깅**: 훈련에 사용된 데이터 양 로깅
- 🚧 LSTM 모델 (개발 예정)
- 🚧 Transformer 모델 (개발 예정)

## 설치

```bash
# 의존성 설치
pip install -r requirements.txt
```

## 사용법

### 1. XGBoost 모델 훈련

```bash
cd backend/app
python training/train_xgboost.py
```

이 명령은:
- 샘플 암호화폐 데이터 생성
- XGBoost 모델 훈련
- 모델을 `models/xgboost/` 디렉토리에 저장
- 훈련 데이터 크기와 피처 정보 로깅

### 2. XGBoost 모델 예측

```bash
cd backend/app
python models/predict_xgboost.py
```

이 명령은:
- 저장된 모델 로드
- 최신 데이터로 예측 수행
- 단일 예측 및 배치 예측 테스트
- 신뢰 구간과 함께 예측 결과 출력

### 3. Python 코드에서 사용

```python
from backend.app.training.train_xgboost import XGBoostTrainer, create_sample_data
from backend.app.models.predict_xgboost import XGBoostPredictor

# 데이터 준비
data = create_sample_data(n_samples=1000)
feature_cols = [col for col in data.columns if col not in ['timestamp', 'low']]

# 모델 훈련
trainer = XGBoostTrainer()
X_train, X_test, y_train, y_test = trainer.prepare_data(
    data,
    target_column='low',
    feature_columns=feature_cols
)
trainer.train(X_train, y_train, X_test, y_test)
trainer.save_model()

# 예측
predictor = XGBoostPredictor()
predictor.load_model()
predictions = predictor.predict_batch(X_test)
```

## 프로젝트 구조

```
models/
├── backend/
│   └── app/
│       ├── training/           # 모델 훈련 모듈
│       │   ├── __init__.py
│       │   └── train_xgboost.py
│       ├── models/             # 모델 예측 모듈
│       │   ├── __init__.py
│       │   └── predict_xgboost.py
│       └── __init__.py
├── models/                     # 저장된 모델 파일
│   └── xgboost/
│       ├── xgboost_model.json
│       └── feature_names.pkl
├── requirements.txt
└── README.md
```

## 주요 기능

### XGBoost 훈련 모듈

- **데이터 로깅**: 훈련에 사용된 데이터 크기, 피처 정보 자동 로깅
- **자동 전처리**: 훈련/테스트 데이터 분리 및 피처 선택
- **모델 저장/로드**: JSON 형식으로 모델 저장 및 재사용

### XGBoost 예측 모듈

- **단일 예측**: 개별 데이터 포인트 예측
- **배치 예측**: 여러 데이터 포인트 동시 예측
- **신뢰 구간**: 예측값과 함께 신뢰 구간 제공

## 로그 예시

```
2025-11-10 08:00:00 - __main__ - INFO - 데이터 전처리 시작 - 전체 데이터 크기: 5000 행
2025-11-10 08:00:00 - __main__ - INFO - 사용할 피처 개수: 9
2025-11-10 08:00:00 - __main__ - INFO - 피처 목록: ['open', 'high', 'close', 'volume', ...]
2025-11-10 08:00:00 - __main__ - INFO - 훈련 데이터: 4000 행
2025-11-10 08:00:00 - __main__ - INFO - 테스트 데이터: 1000 행
2025-11-10 08:00:01 - __main__ - INFO - XGBoost 모델 훈련 시작
2025-11-10 08:00:01 - __main__ - INFO - 훈련 데이터 크기: (4000, 9)
2025-11-10 08:00:05 - __main__ - INFO - 모델 훈련 완료
2025-11-10 08:00:05 - __main__ - INFO - 모델 로드 성공 - 예측 준비 완료
2025-11-10 08:00:05 - __main__ - INFO - 예측 완료 - 데이터 크기: (10, 9), 예측 결과 크기: (10,)
```

## 상태

### ✅ 완료된 기능
- XGBoost 모델 훈련 및 저장
- XGBoost 모델 로드 및 예측
- 데이터 전처리 및 로깅
- 샘플 데이터 생성

### 🚧 개발 예정
- 실제 데이터베이스 연동
- LSTM 모델 구현
- Transformer 모델 구현
- API 서버 구현
- 성능 모니터링 대시보드

## 라이선스

MIT