# 백엔드 예측/저점 판별 개선 계획

## 1. 목표
- 실시간 예측 정확도 안정화
- 멀티 심볼 확장 대비 성능 최적화
- 설정/아티팩트 일관성 확보로 운영 리스크 최소화
- 변경마다 재현 가능한 테스트 플로우 확립

## 2. 개선 항목별 구현 및 테스트 계획

| 우선순위 | 개선 항목 | 구현 계획 | 테스트 계획 |
| --- | --- | --- | --- |
| 1 | 피처-모델 정합성 검증 강화 | 1) `backend/app/model_adapters.py` 로딩 시 요구 피처 목록(`feature_names.json`)을 아티팩트와 함께 배포<br>2) 로딩 단계에서 DB 컬럼/피처 세트와 비교해 누락 시 예외 발생<br>3) `backend/tests/test_features.py`에 피처 스냅샷과 비교하는 단위 테스트 추가 | - `pytest backend/tests/test_features.py -k feature_alignment`<br>- dev 환경에서 모델 로드 실패 시 docker 로그로 검증 |
| 1 | 휴리스틱 스코어 분리 및 학습형 대체 | 1) `backend/app/predictor.py`에서 휴리스틱 계산을 별도 모듈(`heuristic_scoring.py`)로 분리<br>2) 동일 피처 셋으로 경량 XGBoost 학습 파이프라인 추가(`scripts/train_heuristic_xgb.py`)<br>3) ENV로 휴리스틱 모드/학습 모델 토글 | - 신규 모듈 단위 테스트 (`pytest backend/tests/test_predictor.py -k heuristic`)<br>- 실서버에서는 `nowcast` 로그 diff 확인 |
| 2 | 시퀀스 버퍼 지속성 확보 | 1) `backend/app/seq_buffer.py`에 스냅샷 저장/로드 API 추가 (JSON 혹은 Parquet)<br>2) `ws_server` 시작 시 스냅샷 로드, 종료 시 저장<br>3) LIGHT_MODE=1에서도 최소 버퍼 유지 | - `pytest backend/tests/test_stacking.py` 실행으로 버퍼 준비 여부 확인<br>- docker 재시작 후 `nowcast` 지연 시간 측정 |
| 2 | Collector→Predictor 메모리 큐 전환 | 1) `collector.py`에서 새 캔들을 `asyncio.Queue`에 push<br>2) `predictor.py`는 DB 조회 대신 큐에서 소비, 필요 시 fallback<br>3) 멀티 심볼 대응 위해 심볼별 큐 매핑 유지 | - `pytest backend/tests/test_nowcast_api.py`로 API 레이턴시 비교<br>- 부하 테스트: `scripts/backfill.py`로 데이터 밀어넣으며 CPU/DB 사용률 관측 |
| 3 | Feature 계산 가속 | 1) 핵심 지표를 NumPy/Numba로 재작성(`features_fast.py`)<br>2) 기존 계산기와 결과 diff 검사 후 스위칭<br>3) 대규모 심볼 시 asyncio gather로 병렬화 | - `pytest backend/tests/test_features.py -k parity`로 기존과 값 비교<br>- 프로파일링: `python -m backend.app.features --benchmark` 추가 |
| 3 | Config 일원화 | 1) `backend/app/core/config.py`를 Pydantic `BaseSettings` 기반으로 리팩터<br>2) 다른 모듈들은 Config 객체 DI<br>3) `.env.example` 업데이트 | - `pytest backend/tests/test_gap_fill.py` 등 Config 의존 테스트 재실행<br>- docker compose dev/prod 동시에 기동 확인 |
| 3 | 아티팩트 스키마 문서화 및 검증 | 1) `docs/model_artifacts.schema.json` 작성<br>2) `model_adapters`에서 `jsonschema`로 로드 시 검증<br>3) 훈련 스크립트가 스키마에 맞춰 파일 생성하도록 수정 | - `pytest backend/tests/test_hpo.py` (메타 의존) 실행<br>- 잘못된 JSON 주입 시 예외 로그 확인 |
| 4 | 트레이드 전략 플러그인화 | 1) `trade_manager.py`에서 전략 인터페이스(`BaseStrategy`) 정의<br>2) 기존 로직을 `StackingStrategy`로 이동<br>3) 새 전략 추가 시 config로 선택 가능 | - `pytest backend/tests/test_trade_manager.py` 전체 실행<br>- 실거래 모드에서 전략 스위칭 A/B 테스트 |

## 3. 단계별 실행 로드맵
1. **단계 A (주요 정확도 이슈)**: 피처-모델 정합성 + 휴리스틱 분리 (테스트: feature alignment, predictor unit tests)
2. **단계 B (신뢰성/재시동)**: 시퀀스 버퍼 지속성 + 큐 기반 파이프라인 (테스트: nowcast API, stacking readiness)
3. **단계 C (성능/운영)**: Feature 가속 + Config 리팩터 + 아티팩트 스키마 (테스트: 전체 pytest, docker smoke)
4. **단계 D (전략 확장)**: 트레이드 전략 플러그인화 및 신규 전략 실험

## 4. 테스트 운영 원칙
- 변경마다 `pytest backend/tests -k <관련 키워드>` 최소 실행
- docker-compose.dev 환경에서 backend/trainer/frontend 재기동 확인
- 주요 시나리오는 GitHub Actions (또는 VS Code Test Task)로 자동화 예정
- 실거래 배포 전에는 24시간 이상 paper trading 모드로 검증
