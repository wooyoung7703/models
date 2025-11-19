<template>
  <section class="diagnostics-view">
    <header class="head">
      <div>
        <h2>Diagnostics</h2>
        <p class="muted">WS 연결, HTTP 폴백, 피처 수집 상태를 점검합니다.</p>
      </div>
      <div class="head-tools">
        <button class="ghost" :disabled="refreshing" @click="handleRefresh">
          {{ refreshing ? '갱신중…' : '스냅샷 갱신' }}
        </button>
        <button class="ghost" :disabled="reconnecting" @click="handleReconnect">
          {{ reconnecting ? '재연결중…' : 'WS 재연결' }}
        </button>
        <button class="ghost warn" :disabled="disconnecting" @click="handleDisconnect">
          {{ disconnecting ? '해제중…' : 'WS 강제 해제' }}
        </button>
      </div>
    </header>

    <div class="summary-grid">
      <article class="summary-card">
        <p class="label">WS Status</p>
        <p class="value" :class="wsConnected ? 'ok' : 'warn'">{{ wsStatusLabel }}</p>
        <p class="hint">Uptime {{ wsUptimeLabel }}</p>
      </article>
      <article class="summary-card">
        <p class="label">HTTP Base</p>
        <p class="value">{{ apiBase }}</p>
        <p class="hint">마지막 fetch {{ snapshotAgo }}</p>
      </article>
      <article class="summary-card">
        <p class="label">Feature Coverage</p>
        <p class="value">{{ healthySymbols }} / {{ totalSymbols }}</p>
        <p class="hint">Stale {{ staleSymbols }} · flagged {{ flaggedSymbols }}</p>
      </article>
      <article class="summary-card">
        <p class="label">Nowcast map</p>
        <p class="value">{{ nowcastSymbols }}</p>
        <p class="hint">숨김 심볼 제외</p>
      </article>
    </div>

    <div class="grid">
      <article class="table-card">
        <header>
          <h3>Feature Freshness Inspector</h3>
          <span class="muted">상위 {{ freshnessRows.length }}건</span>
        </header>
        <table>
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Fresh (s)</th>
              <th>Missing 24h (m)</th>
              <th>Status</th>
              <th>Latest 15m</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in freshnessRows" :key="row.symbol">
              <td>
                <strong>{{ row.symbol }}</strong>
                <p class="muted small">{{ row.interval }}</p>
              </td>
              <td :class="freshClass(row.freshSeconds)">{{ row.freshLabel }}</td>
              <td>{{ row.missingLabel }}</td>
              <td><span class="status" :class="row.statusClass">{{ row.statusLabel }}</span></td>
              <td>{{ row.latest15m || '—' }}</td>
            </tr>
            <tr v-if="!freshnessRows.length">
              <td colspan="5" class="muted empty">피처 정보 없음</td>
            </tr>
          </tbody>
        </table>
      </article>

      <article class="table-card">
        <header>
          <h3>Connection Timeline</h3>
          <span class="muted">재시도 {{ wsAttempts }}</span>
        </header>
        <ul class="timeline">
          <li>
            <strong>마지막 업데이트</strong>
            <span>{{ snapshotAgo }}</span>
          </li>
          <li>
            <strong>WS 연결 시각</strong>
            <span>{{ wsConnectedSinceLabel }}</span>
          </li>
          <li>
            <strong>Fallback 상태</strong>
            <span>{{ fallbackLabel }}</span>
          </li>
          <li>
            <strong>오류 메시지</strong>
            <span>{{ error || '없음' }}</span>
          </li>
        </ul>
        <div class="placeholder">
          <p class="muted">Sequence Buffer / collector metrics endpoint 준비중</p>
        </div>
      </article>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import type { FeatureSnapshot, NowcastMap } from '../../types/realtime'

const props = defineProps<{
  apiBase: string
  wsConnected: boolean
  wsAttempts: number
  wsConnectedSince: number | null
  lastUpdated: Date | null
  error: string
  features: Record<string, FeatureSnapshot>
  nowcasts: NowcastMap
  manualRefresh?: () => Promise<void>
  connectWs?: () => Promise<void>
  disconnectWs?: () => Promise<void>
}>()

const refreshing = ref(false)
const reconnecting = ref(false)
const disconnecting = ref(false)

async function handleRefresh() {
  if (!props.manualRefresh) return
  refreshing.value = true
  try {
    await props.manualRefresh()
  } finally {
    refreshing.value = false
  }
}

async function handleReconnect() {
  if (!props.connectWs) return
  reconnecting.value = true
  try {
    await props.connectWs()
  } finally {
    reconnecting.value = false
  }
}

async function handleDisconnect() {
  if (!props.disconnectWs) return
  disconnecting.value = true
  try {
    await props.disconnectWs()
  } finally {
    disconnecting.value = false
  }
}

const wsStatusLabel = computed(() => (props.wsConnected ? 'connected' : 'fallback'))
const wsUptimeLabel = computed(() => {
  if (!props.wsConnectedSince) return 'n/a'
  const minutes = Math.max(0, Date.now() - props.wsConnectedSince) / 60000
  return minutes >= 60 ? `${(minutes / 60).toFixed(1)}h` : `${Math.round(minutes)}m`
})
const wsConnectedSinceLabel = computed(() => {
  if (!props.wsConnectedSince) return '정보 없음'
  return new Date(props.wsConnectedSince).toLocaleTimeString()
})
const snapshotAgo = computed(() => {
  if (!props.lastUpdated) return '미수신'
  const delta = Date.now() - props.lastUpdated.getTime()
  if (delta < 60_000) return '방금'
  if (delta < 3_600_000) return `${Math.round(delta / 60000)}분 전`
  return `${(delta / 3_600_000).toFixed(1)}시간 전`
})
const fallbackLabel = computed(() => (props.wsConnected ? 'WS 우선' : 'HTTP 폴백 활성'))

const filteredSymbols = computed(() =>
  Object.keys(props.features || {}).filter((symbol) => !symbol.startsWith('_'))
)
const totalSymbols = computed(() => filteredSymbols.value.length)
const healthySymbols = computed(() =>
  filteredSymbols.value.filter((symbol) => {
    const fresh = props.features[symbol]?.data_fresh_seconds
    return typeof fresh === 'number' && fresh < 300
  }).length
)
const staleSymbols = computed(() =>
  filteredSymbols.value.filter((symbol) => {
    const fresh = props.features[symbol]?.data_fresh_seconds
    return typeof fresh === 'number' && fresh >= 600
  }).length
)
const flaggedSymbols = computed(() =>
  filteredSymbols.value.filter((symbol) => {
    const snap = props.features[symbol]
    return Boolean(snap?.status && snap.status !== 'ok')
  }).length
)
const nowcastSymbols = computed(() =>
  Object.keys(props.nowcasts || {}).filter((symbol) => !symbol.startsWith('_')).length
)

const freshnessRows = computed(() =>
  filteredSymbols.value
    .map((symbol) => buildFreshnessRow(symbol, props.features[symbol]))
    .sort((a, b) => (b.flagScore || 0) - (a.flagScore || 0))
    .slice(0, 10)
)

function buildFreshnessRow(symbol: string, snapshot: FeatureSnapshot | undefined) {
  const freshSeconds = typeof snapshot?.data_fresh_seconds === 'number' ? snapshot.data_fresh_seconds : null
  const missingMinutes = typeof snapshot?.missing_minutes_24h === 'number' ? snapshot.missing_minutes_24h : null
  const status = String(snapshot?.status || (freshSeconds != null && freshSeconds < 900 ? 'ok' : 'warn'))
  const flagScore = (freshSeconds || 0) + (missingMinutes || 0)
  return {
    symbol,
    interval: snapshot?.interval || snapshot?.bucket || '-',
    freshSeconds,
    missingMinutes,
    latest15m: snapshot?.['15m_latest_open_time'] || snapshot?.['5m_latest_open_time'] || snapshot?.latest_open_time,
    statusLabel: status.toUpperCase(),
    statusClass: status === 'ok' ? 'ok' : status === 'error' ? 'bad' : 'warn',
    freshLabel: formatNumber(freshSeconds, 's'),
    missingLabel: formatNumber(missingMinutes, 'm'),
    flagScore,
  }
}

function freshClass(value: number | null) {
  if (value == null) return ''
  if (value > 900) return 'bad'
  if (value > 300) return 'warn'
  return 'ok'
}

function formatNumber(value: number | null, unit: 's' | 'm') {
  if (value == null || Number.isNaN(value)) return '—'
  const base = Math.round(value)
  return `${base}${unit}`
}
</script>

<style scoped>
.diagnostics-view {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.head {
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 1rem;
}

.head-tools {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
}

.summary-card {
  border: 1px solid rgba(96, 165, 250, 0.35);
  border-radius: 18px;
  padding: 0.9rem 1rem;
  background: rgba(8, 14, 27, 0.85);
}

.summary-card .label {
  text-transform: uppercase;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  color: rgba(203, 213, 225, 0.85);
}

.summary-card .value {
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0.12rem 0;
}

.summary-card .value.ok { color: #4ade80; }
.summary-card .value.warn { color: #facc15; }

.summary-card .hint {
  font-size: 0.8rem;
  color: rgba(148, 163, 184, 0.85);
}

.grid {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 1.2rem;
}

.table-card {
  border: 1px solid rgba(79, 98, 143, 0.35);
  border-radius: 20px;
  padding: 1.1rem 1.2rem;
  background: rgba(6, 10, 20, 0.92);
}

.table-card table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}

th,
td {
  padding: 0.55rem 0.45rem;
  text-align: left;
  border-bottom: 1px solid rgba(148, 163, 184, 0.2);
}

.status {
  padding: 0.18rem 0.55rem;
  border-radius: 999px;
  font-size: 0.72rem;
  border: 1px solid transparent;
  text-transform: uppercase;
}
.status.ok { border-color: rgba(74,222,128,0.5); color: #4ade80; }
.status.warn { border-color: rgba(250,204,21,0.45); color: #facc15; }
.status.bad { border-color: rgba(248,113,113,0.45); color: #f87171; }

.timeline {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.6rem;
}

.timeline li {
  display: flex;
  justify-content: space-between;
}

.placeholder {
  margin-top: 1rem;
  padding: 0.9rem;
  border: 1px dashed rgba(148, 163, 184, 0.4);
  border-radius: 14px;
  text-align: center;
}

.ghost.warn {
  border-color: rgba(248, 113, 113, 0.6);
  color: #f87171;
}

.small { font-size: 0.75rem; }
.empty { text-align: center; padding: 1.2rem 0; }

@media (max-width: 960px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
</style>
