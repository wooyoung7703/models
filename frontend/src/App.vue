<template>
  <div class="app-shell">
    <header class="app-header">
      <div class="brand">
        <strong>Realtime Monitor</strong>
        <span class="subline">{{ trainerHeartbeatLabel }}</span>
      </div>

      <div class="connection">
        <span class="status" :class="wsConnected ? 'ok' : 'warn'">
          {{ wsConnected ? 'ws live' : 'polling' }}
        </span>
        <span v-if="lastUpdated" class="muted">업데이트 {{ timeAgo }} 전</span>
        <button class="ghost" @click="manualRefresh">신선도 확인</button>
      </div>
    </header>

    <nav class="app-tabs">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        type="button"
        :class="['tab', { active: tab.id === activeTab }]"
        @click="activeTab = tab.id"
      >
        {{ tab.label }}
      </button>
    </nav>

    <main class="app-main">
      <section v-if="activeTab === 'monitoring'" class="panel-stack">
        <div class="panel primary">
          <header>
            <h2>시세 감시</h2>
            <span class="tag">{{ totalSymbols }} symbols</span>
          </header>
          <div class="metrics">
            <div>
              <p class="metric">{{ readySymbols }}</p>
              <p class="muted">준비된 스택킹</p>
            </div>
            <div>
              <p class="metric">{{ activeTrades.length }}</p>
              <p class="muted">최근 체결</p>
            </div>
            <div>
              <p class="metric">{{ healthySymbols }}</p>
              <p class="muted">특징 정상</p>
            </div>
            <div>
              <p class="metric">{{ trainingProgress }}</p>
              <p class="muted">훈련 스테이터스</p>
            </div>
          </div>
        </div>

        <div class="panel grid-two">
          <TradeSignalPanel
            :latest="latestTrade"
            :open-trades="openTrades"
            :active-signals="topNowcasts.length"
            :ws-connected="wsConnected"
          />
          <NotificationCenter :items="recentNotifications" @dismiss="dismissNotification" />
        </div>

        <div class="panel">
          <header>
            <h3>상위 시그널 미리보기</h3>
            <small class="muted">추후 전용 컴포넌트로 분리 예정</small>
          </header>
          <ul class="preview-list">
            <li v-for="entry in topNowcasts" :key="entry.symbol">
              <span class="symbol">{{ entry.symbol }}</span>
              <span class="value">{{ entry.score.toFixed(3) }}</span>
              <span class="muted">{{ entry.posture }}</span>
            </li>
            <li v-if="!topNowcasts.length" class="muted">데이터 수신 대기중…</li>
          </ul>
        </div>
      </section>

      <ChartView
        v-else-if="activeTab === 'chart'"
        :symbols="visibleSymbols"
        :chart-symbol="chartSymbol"
        :api-base="apiBase"
        :chart-signals="chartSignals"
        :chart-nowcast="chartNowcast"
        :ws-connected="wsConnected"
        @update:chart-symbol="(value) => (chartSymbol = value)"
      />

      <TradesView v-else-if="activeTab === 'trades'" :trades="trades" />

      <TrainingView
        v-else-if="activeTab === 'training'"
        :api-base="apiBase"
        :training-status="trainingStatus"
        :trainer-meta="trainerMeta"
        :entry-metrics="entryMetrics"
        :ws-connected="wsConnected"
        :admin-ack="adminAck"
        :manual-refresh="manualRefresh"
        :send-admin-command="sendAdminCommand"
      />

      <FeaturesView
        v-else-if="activeTab === 'features'"
        :features="features"
        :nowcasts="nowcasts"
        :manual-refresh="manualRefresh"
      />

      <DiagnosticsView
        v-else-if="activeTab === 'diagnostics'"
        :api-base="apiBase"
        :ws-connected="wsConnected"
        :ws-attempts="wsAttempts"
        :ws-connected-since="wsConnectedSince"
        :last-updated="lastUpdated"
        :error="error"
        :features="features"
        :nowcasts="nowcasts"
        :manual-refresh="manualRefresh"
        :connect-ws="connectWs"
        :disconnect-ws="disconnectWs"
      />

      <section v-else class="panel placeholder">
        <p>{{ activeLabel }} 화면 준비중입니다.</p>
        <p class="muted">관련 서브뷰와 차트는 문서화된 계획을 따라 이어서 구현할 예정입니다.</p>
      </section>
    </main>

    <aside class="notification-tray">
      <header>
        <h4>알림</h4>
        <button class="ghost" @click="sendAdminCommand('ping')">Ping</button>
      </header>
      <ul>
        <li v-for="note in trayNotifications" :key="note.id">
          <span class="badge" :class="note.level">{{ note.level }}</span>
          <span>{{ note.message }}</span>
        </li>
        <li v-if="!trayNotifications.length" class="muted">알림 없음</li>
      </ul>
      <p v-if="adminAck" class="muted ack">{{ adminAck }}</p>
    </aside>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
// @ts-ignore script-setup default export shim
import TradeSignalPanel from './components/TradeSignalPanel.vue'
// @ts-ignore script-setup default export shim
import NotificationCenter from './components/NotificationCenter.vue'
// @ts-ignore script-setup default export shim
import ChartView from './components/views/ChartView.vue'
// @ts-ignore script-setup default export shim
import TradesView from './components/views/TradesView.vue'
// @ts-ignore script-setup default export shim
import FeaturesView from './components/views/FeaturesView.vue'
// @ts-ignore script-setup default export shim
import TrainingView from './components/views/TrainingView.vue'
// @ts-ignore script-setup default export shim
import DiagnosticsView from './components/views/DiagnosticsView.vue'
import { useRealtimeData } from './composables/useRealtimeData'
import type { ChartSignal, NowcastEntry, TradeRow } from './types/realtime'

type DerivedNowcast = {
  symbol: string
  score: number
  posture: string
}

const tabs = [
  { id: 'monitoring', label: 'Monitoring' },
  { id: 'chart', label: 'Chart' },
  { id: 'trades', label: 'Trades' },
  { id: 'training', label: 'Training' },
  { id: 'features', label: 'Features' },
  { id: 'diagnostics', label: 'Diagnostics' },
]

const activeTab = ref('monitoring')

const {
  apiBase,
  wsConnected,
  wsAttempts,
  wsConnectedSince,
  lastUpdated,
  error,
  nowcasts,
  trades,
  features,
  trainingStatus,
  trainerMeta,
  entryMetrics,
  notifications,
  adminAck,
  manualRefresh,
  sendAdminCommand,
  connectWs,
  disconnectWs,
} = useRealtimeData({ autoStart: true })

const activeLabel = computed(() => tabs.find((tab) => tab.id === activeTab.value)?.label || '')

const visibleSymbols = computed(() =>
  Object.keys(nowcasts.value).filter((symbol) => !symbol.startsWith('_'))
)
const totalSymbols = computed(() => visibleSymbols.value.length)
const readySymbols = computed(() =>
  Object.values(nowcasts.value).filter((entry) => entry?.stacking?.ready).length
)
const featuresFreshLimitSeconds = 5 * 60
const healthySymbols = computed(() =>
  Object.values(features.value).filter((snapshot) => {
    if (!snapshot || typeof snapshot.data_fresh_seconds !== 'number') return false
    return snapshot.data_fresh_seconds < featuresFreshLimitSeconds
  }).length
)
const topNowcasts = computed<DerivedNowcast[]>(() =>
  Object.entries(nowcasts.value)
    .filter(([symbol]) => !symbol.startsWith('_'))
    .map(([symbol, payload]) => deriveNowcast(symbol, payload))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)
)
const activeTrades = computed<TradeRow[]>(() => trades.value.slice(0, 5))
const latestTrade = computed(() => trades.value[0] || null)
const openTrades = computed(() => trades.value.filter((row) => row.status === 'open'))
const chartSymbol = ref('')
const chartSignalLimit = 40
const chartNowcast = computed(() => (chartSymbol.value ? nowcasts.value[chartSymbol.value] || null : null))
const chartSignals = computed<ChartSignal[]>(() => {
  const symbol = chartSymbol.value
  if (!symbol) return []
  const payload = nowcasts.value[symbol]
  const derived: ChartSignal[] = []
  const rawSignals = (payload?.chart_signals || payload?.signals || []) as Array<ChartSignal | Record<string, any>>
  rawSignals.forEach((entry, idx) => {
    const normalized = normalizeChartSignal(entry, `${symbol}-sig-${idx}`)
    if (normalized) derived.push(normalized)
  })
  trades.value
    .filter((trade) => trade.symbol === symbol)
    .forEach((trade) => {
      const marker = tradeToChartSignal(trade)
      if (marker) derived.push(marker)
    })
  return derived.slice(-chartSignalLimit)
})
const recentNotifications = computed(() => notifications.value.slice(0, 4))
const trayNotifications = computed(() => notifications.value.slice(0, 6))
const trainingProgress = computed(() => {
  if (!trainingStatus.value) return 'idle'
  if (trainingStatus.value.pending_meta_retrain) return 'meta pending'
  if (trainingStatus.value.meta_neg_streak_warn) return 'needs review'
  const schedules = trainingStatus.value.next_runs
  const nextKey = schedules ? Object.keys(schedules)[0] : null
  if (nextKey) {
    const target = schedules?.[nextKey]
    if (target?.state === 'running') return `${nextKey} running`
    if (typeof target?.eta_seconds === 'number') {
      const minutes = Math.max(1, Math.round(target.eta_seconds / 60))
      return `${nextKey} η ${minutes}m`
    }
  }
  return 'idle'
})
const timeAgo = computed(() => {
  if (!lastUpdated.value) return '미수신'
  const delta = Date.now() - lastUpdated.value.getTime()
  if (delta < 30_000) return '방금'
  if (delta < 120_000) return `${Math.round(delta / 1000)}초`
  const minutes = Math.floor(delta / 60000)
  return `${minutes}분`
})
const trainerHeartbeatLabel = computed(() => {
  if (!trainerMeta.value) return 'Trainer heartbeat 미수신'
  const healthy = trainerMeta.value.heartbeat_healthy
  const age = trainerMeta.value.heartbeat_age_seconds
  const ageLabel = typeof age === 'number' ? `${Math.round(age / 60)}m ago` : 'n/a'
  return healthy ? `Trainer heartbeat · ${ageLabel}` : `Trainer stale · ${ageLabel}`
})

watch(
  visibleSymbols,
  (symbols) => {
    if (!symbols.length) {
      chartSymbol.value = ''
      return
    }
    if (!chartSymbol.value || !symbols.includes(chartSymbol.value)) {
      chartSymbol.value = symbols[0]
    }
  },
  { immediate: true }
)

function deriveNowcast(symbol: string, payload: NowcastEntry | undefined): DerivedNowcast {
  if (!payload) return { symbol, score: 0, posture: 'pending' }
  const rawScore =
    payload.stacking?.prob_final ??
    payload.stacking?.prob ??
    payload.bottom_score ??
    0
  const score = typeof rawScore === 'number' ? rawScore : 0
  const posture = payload.stacking?.decision === true ? 'long' : payload.stacking?.decision === false ? 'flat' : 'pending'
  return { symbol, score, posture }
}

function dismissNotification(id: number) {
  notifications.value = notifications.value.filter((note) => note.id !== id)
}

function normalizeChartSignal(
  entry: ChartSignal | Record<string, any>,
  fallbackId: string
): ChartSignal | null {
  if (!entry) return null
  if ('ts' in entry && entry.ts && typeof entry.ts === 'string' && entry.side) {
    return {
      id: entry.id ?? fallbackId,
      ts: entry.ts,
      side: entry.side === 'sell' ? 'sell' : 'buy',
      price: typeof entry.price === 'number' ? entry.price : undefined,
      label: entry.label,
    }
  }
  const ts = (entry as any).ts || (entry as any).timestamp || (entry as any).created_at
  if (!ts) return null
  const rawSide = (entry as any).side || (entry as any).direction || (entry as any).action
  const side = rawSide === 'sell' || rawSide === 'short' ? 'sell' : 'buy'
  const priceCandidate = (entry as any).price ?? (entry as any).level ?? (entry as any).value
  const label = (entry as any).label || (entry as any).reason || (entry as any).type
  return {
    id: (entry as any).id ?? fallbackId,
    ts: typeof ts === 'number' ? new Date(ts).toISOString() : ts,
    side,
    price: typeof priceCandidate === 'number' ? priceCandidate : undefined,
    label,
  }
}

function tradeToChartSignal(trade: TradeRow): ChartSignal | null {
  if (!trade?.created_at) return null
  const rawDirection = (trade as any).side || (trade as any).direction
  const side: ChartSignal['side'] = rawDirection === 'sell' || rawDirection === 'short' ? 'sell' : 'buy'
  return {
    id: trade.id ?? `${trade.symbol}-${trade.created_at}`,
    ts: trade.created_at,
    side,
    price: trade.entry_price,
    label: trade.status === 'closed' ? 'exit' : 'entry',
  }
}
</script>

<style scoped>
.app-shell {
  display: grid;
  grid-template-columns: 1fr 320px;
  grid-template-rows: auto auto 1fr;
  min-height: 100vh;
  background: #0f1115;
  color: #f4f6fb;
  gap: 16px;
  padding: 24px;
}

.app-header {
  grid-column: 1 / 3;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.brand {
  display: flex;
  flex-direction: column;
}

.subline {
  font-size: 0.85rem;
  color: #9ca3af;
}

.connection {
  display: flex;
  align-items: center;
  gap: 12px;
}

.status {
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 0.8rem;
  text-transform: uppercase;
}

.status.ok {
  background: #14532d;
  color: #86efac;
}

.status.warn {
  background: #4c1d1d;
  color: #fca5a5;
}

.muted {
  color: #9ca3af;
  font-size: 0.85rem;
}

.app-tabs {
  grid-column: 1 / 2;
  display: flex;
  gap: 8px;
}

.tab {
  flex: 1;
  padding: 8px 12px;
  border-radius: 8px;
  border: 1px solid #1f2933;
  background: transparent;
  color: inherit;
  text-align: center;
  cursor: pointer;
}

.tab.active {
  background: #1f2933;
}

.app-main {
  background: #111827;
  border-radius: 12px;
  padding: 16px;
}

.panel-stack {
  display: grid;
  gap: 16px;
}

.panel {
  background: #0b0f19;
  border: 1px solid #1f2933;
  border-radius: 12px;
  padding: 16px;
}

.panel.primary {
  border-color: #2563eb;
}

.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
  margin-top: 16px;
}

.metric {
  font-size: 1.6rem;
  font-weight: 600;
}

.preview-list {
  list-style: none;
  padding: 0;
  margin: 12px 0 0;
}

.preview-list li {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px solid #1f2933;
}

.preview-list li:last-child {
  border-bottom: none;
}

.symbol {
  font-weight: 600;
}

.value {
  font-variant-numeric: tabular-nums;
}

.notification-tray {
  grid-column: 2 / 3;
  grid-row: 2 / 4;
  background: #0b0f19;
  border: 1px solid #1f2933;
  border-radius: 12px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.notification-tray ul {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
  flex: 1;
  overflow-y: auto;
}

.notification-tray li {
  display: flex;
  gap: 8px;
  align-items: center;
}

.badge {
  padding: 2px 6px;
  border-radius: 6px;
  font-size: 0.75rem;
  text-transform: uppercase;
}

.badge.info {
  background: #1d4ed8;
}

.badge.warn {
  background: #92400e;
}

.badge.error {
  background: #991b1b;
}

.ghost {
  background: transparent;
  border: 1px solid #1f2933;
  color: inherit;
  padding: 6px 10px;
  border-radius: 6px;
  cursor: pointer;
}

.ack {
  font-size: 0.8rem;
}

@media (max-width: 960px) {
  .app-shell {
    grid-template-columns: 1fr;
    grid-template-rows: auto auto auto 1fr auto;
  }

  .app-header,
  .app-tabs,
  .app-main {
    grid-column: 1 / 2;
  }

  .notification-tray {
    grid-column: 1 / 2;
    grid-row: auto;
  }
}
</style>
