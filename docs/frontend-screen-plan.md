# Frontend Rebuild Plan (Backend-Driven)

## 1. 목적 및 범위
- **목표**: 백엔드 WS 서버(`backend/app/ws_server.py`)가 제공하는 실시간 기능(나우캐스트, 트레이딩, 스케줄링, 트레이너 메타 등)을 정면에서 드러내는 경량 프론트엔드를 재설계한다.
- **범위**: 내비게이션, 화면(뷰), 컴포넌트, 데이터 바인딩, 오류 처리, 점진적 향상(WS ↔ HTTP 폴백)까지 포함한 설계 문서. 구현은 추후 단계별 진행.

## 2. 백엔드 데이터 및 기능 정리
| 채널 | 페이로드 | 활용 예정 기능 |
| --- | --- | --- |
| **WebSocket snapshot** | `{ type: 'snapshot', nowcast, trades, features, training_status, trainer, entry_metrics }` | 초기 로딩, 모든 상태 초기화 |
| **WebSocket nowcast** | `{ type: 'nowcast', symbol, data }` | 심볼 카드, 차트, 알림 업데이트 |
| **WebSocket trades** | `{ type: 'trades', data }` | 실시간 트레이드 패널, 알림 |
| **WebSocket features** | `{ type: 'features', data }` | 데이터 품질 카드, 헬스 배지 |
| **WebSocket training_status / trainer / entry_metrics** | 다양한 메타 필드 | 트레이너 타임라인, 스케줄러, Entry 메타 카드 |
| **WebSocket admin_ack** | `{ type: 'admin_ack', action, ok, error }` | 스케줄러/관리 명령 피드백 |
| **HTTP Fallback** | `/nowcast`, `/trades`, `/health/features`, `/training/status` | WS 실패 시 폴백 fetch 주기 조절 |
| **Admin WS 명령** | `{ type: 'admin', action, token }` | 스케줄러 제어(예: `scheduler:start:<job>`), 트레이너 트리거 |

추가적으로 collector → predictor 파이프라인, seq 버퍼 스냅샷 등의 상태도 노출 예정(메타 카드/다이얼로그로 확장 여지 확보).

## 3. 사용자 역할 & 핵심 워크플로우
1. **옵스 관제자**: 실시간 Bottom Detector 상태 확인, 스택 신호 감시, 이상 징후 알림 수신.
2. **트레이더**: 실거래 내역 모니터링, 거래 시그널 응답, 쿨다운/추가매수 조건 체크.
3. **ML 엔지니어**: 트레이너 상태, 메타 히스토리, 피처 헬스 확인 및 스케줄 제어.

이 세 역할을 한 화면 내에서 탭/패널 분리로 서포트한다.

## 4. 내비게이션 설계
| 1차 메뉴 | 설명 | 연관 데이터/기능 | 2차 구획 |
| --- | --- | --- | --- |
| **Monitoring** (기본) | 실시간 대시보드 | snapshot, nowcast, features, entry_metrics | `System Overview`, `Symbol Board`, `Detail Panel`, `Meta & Training`, `Trades`, `Notifications` |
| **Charting** | 특정 심볼에 집중한 시각화 | nowcast.signals, trades | `Signal Timeline`, `Price/Feature overlays`, `Manual controls` |
| **Scheduler & Admin** | 배치/트레이너 제어 | training_status.next_runs, admin WS | `Scheduler Grid`, `Command Center`, `Job Logs` |
| **Diagnostics** (신규) | 버퍼/피처 헬스 심층 뷰 | features, seq_buffer snapshot, collector health | `Feature Health Matrix`, `Sequence Buffer Inspector`, `WS Metrics` |

탑 레벨 탭은 `App.vue`에서 상태(`activeView`)로 관리한다. 추후 다국어, 사용자 권한을 위해 라우터 도입 가능하나 초기에는 단일 페이지 탭 구조 유지.

## 5. 화면별 레이아웃 & 컴포넌트

### 5.1 Monitoring View
1. **Masthead**
   - API 입력, WS 상태 배지, 마지막 업데이트 상대 시간, 수동 동기화 버튼.
   - 백엔드: `wsConnected`, `apiBase`, `lastUpdated`.

2. **System Strip (Metric Cards)**
   - `SystemHealthCard`: healthDot, error 메시지.
   - `SignalDistributionCard`: active/high confidence 비율.
   - `StackingMetaCard`: `_stacking_meta` 정보(메서드, threshold, 모델 목록).
   - `EntryMetaCard`: `entry_metrics` 집계 + 스파크라인.
   - `TrainerStatusCard`: heartbeat, run info.
   - `TrainingStatusCard`: retrain 상태, ETA, sparkline/스크래터.
   - `PollingIntervalControl`: range slider→`pollInterval`.

3. **Notification Center**
   - WebSocket 상태, 트레이드 이벤트, 스케줄러 명령 결과 연동.

4. **Layout Presets**
   - 필터/정렬 상태를 `localStorage` 기반으로 저장/적용/삭제.

5. **Trade Signal & History Panels**
   - `TradeSignalPanel`: latest trade, open trade count, WS 상태.
   - `TradeHistoryList`: 최근 fills.

6. **Low Price Indicator**
   - 평균 bottom score + 활성/고신뢰 비율로 지표.

7. **Symbol Board (Grid)**
   - Filter input, Sort controls(`sortKey`, `sortDir`).
   - `SymbolCard` (새 컴포넌트): 
     - header: symbol, interval, timestamp.
     - metrics: price, bottom score, stacking prob, adjusted prob, entry prob, margin.
     - sparkline: `components` 데이터 활용.
     - badges: freshness, gap, TF, SEQ, strength, WR.
   - Card selection→Detail panel 연동.

8. **Detail Panel**
   - `SymbolDetailHeader`: pill, confidence, margin, 모델 태그.
   - `ModelStatusList`: base model readiness.
   - `ProbStructure`: stacking 구조 시각화.
   - `MetaProbMetrics`: 임계치, 교정 정보.
   - `BaseProbList`: 각 모델 확률 progress bar.
   - `FeatureContributionList`: 주요 컴포넌트 값.
   - `FeatureHealthCard`: freshness/gap/소스.

9. **JobLogTable**
   - WS/스케줄러/트레이드 조치 로그.

10. **Trades Pane**
    - table + expandable fills.

11. **SchedulerPanel (mini)**
    - 주요 배치 제어 quick access.

12. **Empty State**
    - 데이터 미도착 시 안내.

### 5.2 Charting View
구성 요소
- `ChartLayout`(좌) : 심볼 선택, range controls, overlay toggles.
- `NowcastChart`: price + bottom score + stacking prob.
- `SignalMarkers`: trades/signals.
- `FeatureDeltaList`: 선택한 시점의 피처 변화.
- `PlaybackControls`: 과거 구간 재생(WS 데이터 캐싱 필요 → 초기에는 snapshot buffer 기반).
- 백엔드 연계: `chartSignals`, `chartNowcast`, `trades`.

### 5.3 Scheduler & Admin View
섹션
1. **Filters & Bulk Actions**: type/state 필터, 전체 stop/start 버튼.
2. **Scheduler Status Grid**: 카드형으로 ETA, interval, 설명 표시.
3. **SchedulerList**: 행 + command 버튼(`start/stop`), 진행중인 명령 상태(`schedulerCmdLoading`).
4. **SchedulerPanel (full)**: form-based admin action들(메타 훈련 트리거 등) + WebSocket 전송.
5. **SchedulerLog**: 최근 로그 목록.
6. **AdminAckBanner**: 최신 ack 텍스트.

### 5.4 Diagnostics View (신규)
제안 컴포넌트
- `FeatureHealthMatrix`: 심볼 vs 지표(누락 분, freshness, TF coverage).
- `SequenceBufferInspector`: seq 버퍼 길이, 마지막 갱신, snapshot restore 여부.`settings.SEQ_BUFFER_STATE_PATH`와 연동.
- `WsMetricsPanel`: WS Uptime, reconnect attempts, queue size, polling fallback 상태.
- `BackendConfigSummary`: 주요 env (LIGHT_MODE, EXCHANGE_TYPE 등) 읽기.

## 6. 컴포넌트 상세 사양

### 6.1 Generic 상태 관리
- `useWsConnection()` (composition 함수 후보): 연결 상태, 재시도, 폴백 제어 로직 캡슐화 → `App.vue` 슬림화.
- `useNotifications()`: queue/ toast 관리.

### 6.2 주요 컴포넌트 속성 & 이벤트
| 컴포넌트 | Props | Emits / Slots | 데이터 출처 |
| --- | --- | --- | --- |
| `SystemHealthCard` | `healthDot`, `healthText`, `error`, `lastUpdated` | - | `nowcasts`, `error`, `lastUpdated` |
| `SignalDistributionCard` | `activeCount`, `highCount`, `total` | - | `displayedSymbols`, `activeSignals`, `highConfidenceSignals` |
| `StackingMetaCard` | `meta` | - | `_stacking_meta` |
| `EntryMetaCard` | `entryMeta`, `sparkPoints` | - | `entryMetrics` |
| `TrainerStatusCard` | `trainerMeta`, `trainingStatus` | - | WS snapshot/updates |
| `TrainingStatusCard` | `status`, `historyDisplayCount` | `update:historyDisplayCount`, `export` | `trainingStatus` |
| `LayoutPresets` | `presets`, `currentFilter`, `sortKey`, `sortDir` | `save`, `apply`, `delete` | localStorage |
| `SymbolCard` | `symbol`, `nowcast`, `features`, `selected` | `select(symbol)` | nowcasts map |
| `SymbolDetailPanel` | `symbol`, `stacking`, `featureSnapshot`, `baseInfo`, `components` | - | selected symbol |
| `TradeSignalPanel` | `latest`, `openTrades`, `activeSignalCount`, `wsConnected` | - | trades, computed |
| `TradeHistoryList` | `trades` | `toggle(tradeId)` | trades |
| `SchedulerPanel` | `apiBase`, `trainingStatus`, `wsConnected`, `useWs`, `sendWs`, `adminAck` | `trigger(action)` | admin commands |
| `SchedulerList` | `items`, `loadingKey` | `command({ key, action })` | `trainingStatus.next_runs` |
| `FeatureHealthMatrix` (신규) | `features`, `symbols` | `select(symbol)` | `features` map |
| `SequenceBufferInspector` | `buffersMeta` | - | 추후 HTTP endpoint 필요 (우선 placeholder) |

## 7. 상태/데이터 흐름 다이어그램 (요약)
1. **WS 연결** → `snapshot` 수신 시 각 store ref 세팅 → HTTP 폴백 중지.
2. `nowcast`/`features`/`trades`/`training_status` 개별 메시지 → 해당 ref 머지.
3. HTTP 폴백 타이머는 WS 끊길 때만 가동.
4. Admin 명령 → WS `sendAdminWs` → `admin_ack` 메시지 → Notification + Banner.

## 8. 구현 단계 제안
1. **상태 훅 분리**: WS/HTTP/polling 로직을 `composables`로 이동 → `App.vue` 단순화.
2. **Monitoring 뼈대 복원**: 기존 레이아웃을 컴포넌트 단위로 분해하고 Storybook 없이 단독 컴포넌트 작동 확인.
3. **Scheduler/Admin 정리**: 새 Diagnostics 탭 추가 전까지 기존 Scheduler 뷰 리팩터.
4. **Diagnostics 탭 MVP**: feature health matrix부터 구현.
5. **차트 뷰 개선**: lightweight-charts 활용, 심볼별 데이터 decimation.

## 9. 향후 고려사항
- 라우터 도입 시 탭 ↔ 라우트 매핑.
- i18n: 모든 라벨을 메시지 테이블로 정리.
- 접근성: 키보드 내비게이션, 라이브 영역(`aria-live`)으로 WS 알림 표기.
- 테스트: vitest + component 테스트, WS mock을 위한 util 추가.

---
이 문서는 프론트엔드 작업 진행 순서를 안내하는 기준으로 사용하며, 각 단계 구현 시 체크박스/업데이트 섹션을 추가해 추적할 수 있다.