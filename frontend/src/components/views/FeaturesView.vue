<template>
  <section class="features-view">
    <header class="head">
      <div>
        <h2>Feature Health</h2>
        <p class="muted">수집·가공 파이프라인의 최신 상태와 결측을 추적합니다.</p>
      </div>
      <div class="head-tools">
        <span class="pill" :class="staleRatio > 0.25 ? 'warn' : 'ok'">
          {{ healthyCount }} / {{ totalSymbols }} healthy
        </span>
        <button class="ghost" :disabled="refreshing" @click="handleRefresh">
          {{ refreshing ? '갱신중…' : '데이터 새로고침' }}
        </button>
      </div>
    </header>

    <div class="summary-grid">
      <article class="summary-card">
        <p class="label">평균 freshness</p>
        <p class="value">{{ avgFresh }}</p>
        <p class="hint">data_fresh_seconds 기준</p>
      </article>
      <article class="summary-card">
        <p class="label">Stale symbols</p>
        <p class="value" :class="{ warn: staleCount > 0 }">{{ staleCount }}</p>
        <p class="hint">{{ staleHint }}</p>
      </article>
      <article class="summary-card">
        <p class="label">24h 누락</p>
        <p class="value">{{ missing24hAvg }}</p>
        <p class="hint">평균 missing_minutes_24h</p>
      </article>
      <article class="summary-card">
        <p class="label">감시 지표</p>
        <p class="value">{{ flagCount }}</p>
        <p class="hint">상태/경고 필드 감지</p>
      </article>
    </div>

    <div class="matrix-layout">
      <article class="matrix-card">
        <header>
          <h3>심볼별 상태</h3>
          <span class="muted">최근 업데이트 순</span>
        </header>
        <table>
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Fresh (s)</th>
              <th>Missing 24h (m)</th>
              <th>Status</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="row in featureRows"
              :key="row.symbol"
              :class="{ selected: row.symbol === selectedSymbol }"
              @click="selectSymbol(row.symbol)"
            >
              <td>
                <strong>{{ row.symbol }}</strong>
                <p class="muted small">{{ row.intervalLabel }}</p>
              </td>
              <td :class="freshClass(row.freshSeconds)">{{ formatNumber(row.freshSeconds) }}</td>
              <td>{{ formatNumber(row.missingMinutes) }}</td>
              <td>
                <span class="status" :class="row.statusClass">{{ row.statusLabel }}</span>
              </td>
              <td class="muted small">{{ row.note }}</td>
            </tr>
            <tr v-if="!featureRows.length">
              <td colspan="5" class="muted empty">수집된 Feature snapshot이 없습니다.</td>
            </tr>
          </tbody>
        </table>
      </article>

      <article class="detail-card" v-if="selectedRow">
        <header>
          <div>
            <h3>{{ selectedRow.symbol }} 상세</h3>
            <p class="muted">최근 업데이트 {{ selectedRow.freshLabel }}</p>
          </div>
          <span class="status" :class="selectedRow.statusClass">{{ selectedRow.statusLabel }}</span>
        </header>

        <section class="detail-grid">
          <div v-for="entry in detailEntries" :key="entry.key">
            <p class="key">{{ entry.key }}</p>
            <p class="value">{{ entry.value }}</p>
          </div>
        </section>

        <section class="nowcast" v-if="selectedRow.nowcast">
          <header>
            <h4>Nowcast Snapshot</h4>
            <p class="muted">Stacking prob 및 시세 메타</p>
          </header>
          <ul>
            <li>
              <span>Stacking prob</span>
              <strong>{{ formatProb(selectedRow.nowcast?.stacking?.prob_final) }}</strong>
            </li>
            <li>
              <span>Bottom score</span>
              <strong>{{ formatProb(selectedRow.nowcast?.bottom_score) }}</strong>
            </li>
            <li>
              <span>Price</span>
              <strong>{{ formatPrice(selectedRow.nowcast?.price) }}</strong>
            </li>
          </ul>
        </section>
      </article>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { FeatureSnapshot, NowcastMap } from '../../types/realtime'

const props = defineProps<{
  features: Record<string, FeatureSnapshot>
  nowcasts: NowcastMap
  manualRefresh?: () => Promise<void>
}>()

const refreshing = ref(false)
const selectedSymbol = ref('')

async function handleRefresh() {
  if (!props.manualRefresh) return
  refreshing.value = true
  try {
    await props.manualRefresh()
  } finally {
    refreshing.value = false
  }
}

const featureRows = computed(() =>
  Object.entries(props.features || {})
    .filter(([symbol]) => !symbol.startsWith('_'))
    .map(([symbol, snapshot]) => buildRow(symbol, snapshot))
    .sort((a, b) => (a.freshSeconds ?? Infinity) - (b.freshSeconds ?? Infinity))
)

watch(featureRows, (rows) => {
  if (!rows.length) {
    selectedSymbol.value = ''
  } else if (!rows.find((row) => row.symbol === selectedSymbol.value)) {
    selectedSymbol.value = rows[0].symbol
  }
}, { immediate: true })

const selectedRow = computed(() => featureRows.value.find((row) => row.symbol === selectedSymbol.value))

const totalSymbols = computed(() => featureRows.value.length)
const healthyCount = computed(() => featureRows.value.filter((row) => row.statusClass === 'ok').length)
const staleCount = computed(() => featureRows.value.filter((row) => row.statusClass === 'warn').length)
const flagCount = computed(() => featureRows.value.filter((row) => row.flagged).length)
const staleRatio = computed(() => (totalSymbols.value ? staleCount.value / totalSymbols.value : 0))
const avgFresh = computed(() => {
  const values = featureRows.value.map((row) => row.freshSeconds).filter(isFiniteNumber)
  if (!values.length) return 'n/a'
  const avg = values.reduce((sum, value) => sum + (value ?? 0), 0) / values.length
  return avg >= 3600 ? `${(avg / 3600).toFixed(1)}h` : `${Math.round(avg / 60)}m`
})
const missing24hAvg = computed(() => {
  const values = featureRows.value.map((row) => row.missingMinutes).filter(isFiniteNumber)
  if (!values.length) return 'n/a'
  const avg = values.reduce((sum, value) => sum + (value ?? 0), 0) / values.length
  return `${Math.round(avg)}m`
})
const staleHint = computed(() => (staleCount.value ? '재시작 혹은 백필 필요' : '모두 정상 새로고침 중'))

const detailEntries = computed(() => {
  const row = selectedRow.value
  if (!row) return []
  return row.detailEntries
})

function selectSymbol(symbol: string) {
  selectedSymbol.value = symbol
}

function buildRow(symbol: string, snapshot: FeatureSnapshot | undefined) {
  const freshSeconds = typeof snapshot?.data_fresh_seconds === 'number' ? snapshot.data_fresh_seconds : null
  const missingMinutes = typeof snapshot?.missing_minutes_24h === 'number' ? snapshot.missing_minutes_24h : null
  const status = (snapshot?.status as string) || deriveStatus(freshSeconds, missingMinutes)
  const note = String(snapshot?.status_reason || snapshot?.note || snapshot?.source || '')
  const intervalLabel = snapshot?.interval || snapshot?.bucket || '-'
  const flagged = Boolean(snapshot?.gap_detected || snapshot?.missing || (snapshot?.status && snapshot.status !== 'ok'))
  const detailEntries = extractDetailEntries(snapshot)
  const nowcast = props.nowcasts?.[symbol]
  return {
    symbol,
    snapshot,
    freshSeconds,
    missingMinutes,
    statusLabel: status.toUpperCase(),
    statusClass: status === 'ok' ? 'ok' : status === 'error' ? 'bad' : 'warn',
    note,
    intervalLabel,
    flagged,
    detailEntries,
    nowcast,
    freshLabel: freshSeconds == null ? 'unknown' : `${Math.round(freshSeconds / 60)}m ago`,
  }
}

function deriveStatus(freshSeconds: number | null, missingMinutes: number | null) {
  if (freshSeconds == null) return 'unknown'
  if (freshSeconds > 15 * 60 || (missingMinutes ?? 0) > 60) return 'warn'
  return 'ok'
}

function extractDetailEntries(snapshot?: FeatureSnapshot) {
  if (!snapshot) return []
  const preferredKeys = [
    'data_fresh_seconds',
    'missing_minutes_24h',
    'latest_open_time',
    '15m_latest_open_time',
    '5m_latest_open_time',
    'source',
    'status',
    'status_reason',
    'gap_detected',
    'missing',
  ]
  const dynamicKeys = Object.keys(snapshot)
    .filter((key) => !preferredKeys.includes(key) && typeof snapshot[key as keyof FeatureSnapshot] !== 'object')
  const keys = [...preferredKeys, ...dynamicKeys].slice(0, 14)
  return keys.map((key) => ({ key, value: formatDetailValue(snapshot[key as keyof FeatureSnapshot]) }))
}

function formatDetailValue(value: unknown) {
  if (value == null) return '—'
  if (typeof value === 'number') {
    if (Math.abs(value) > 9999) return value.toExponential(2)
    return Number.isInteger(value) ? value.toString() : value.toFixed(3)
  }
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  return String(value)
}

function freshClass(value: number | null) {
  if (value == null) return ''
  if (value > 900) return 'bad'
  if (value > 300) return 'warn'
  return 'ok'
}

function formatNumber(value: number | null) {
  if (value == null || Number.isNaN(value)) return '—'
  return value >= 100 ? Math.round(value).toString() : value.toFixed(1)
}

function formatProb(value?: number | null) {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'n/a'
  return value.toFixed(3)
}

function formatPrice(value?: number | null) {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'n/a'
  if (value >= 1000) return value.toFixed(0)
  if (value >= 10) return value.toFixed(2)
  return value.toFixed(4)
}

function isFiniteNumber(value: number | null | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}
</script>

<style scoped>
.features-view {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.head {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}

.head-tools {
  display: flex;
  gap: 0.75rem;
  align-items: center;
}

.pill {
  padding: 0.3rem 0.8rem;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.4);
}

.pill.ok {
  border-color: rgba(74, 222, 128, 0.4);
  color: #4ade80;
}

.pill.warn {
  border-color: rgba(248, 113, 113, 0.5);
  color: #f87171;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
}

.summary-card {
  border: 1px solid rgba(59, 130, 246, 0.3);
  border-radius: 18px;
  padding: 0.9rem 1rem;
  background: rgba(8, 14, 27, 0.85);
}

.summary-card .label {
  text-transform: uppercase;
  font-size: 0.74rem;
  letter-spacing: 0.08em;
  color: rgba(203, 213, 225, 0.85);
}

.summary-card .value {
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0.1rem 0;
}

.summary-card .value.warn {
  color: #facc15;
}

.summary-card .hint {
  font-size: 0.8rem;
  color: rgba(148, 163, 184, 0.85);
}

.matrix-layout {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 1.4rem;
}

.matrix-card,
.detail-card {
  border: 1px solid rgba(99, 121, 173, 0.35);
  border-radius: 20px;
  padding: 1.1rem 1.2rem;
  background: rgba(7, 11, 23, 0.92);
}

.matrix-card table {
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

tbody tr.selected {
  background: rgba(37, 99, 235, 0.1);
}

tbody tr:hover {
  background: rgba(59, 130, 246, 0.08);
}

.status {
  padding: 0.18rem 0.6rem;
  border-radius: 999px;
  font-size: 0.72rem;
  border: 1px solid transparent;
  text-transform: uppercase;
}

.status.ok {
  border-color: rgba(74, 222, 128, 0.5);
  color: #4ade80;
}

.status.warn {
  border-color: rgba(250, 204, 21, 0.45);
  color: #facc15;
}

.status.bad {
  border-color: rgba(248, 113, 113, 0.45);
  color: #f87171;
}

.detail-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.75rem;
  margin-top: 1rem;
}

.detail-grid .key {
  font-size: 0.75rem;
  opacity: 0.7;
  margin-bottom: 0.15rem;
}

.detail-grid .value {
  font-variant-numeric: tabular-nums;
}

.nowcast {
  margin-top: 1.2rem;
  border-top: 1px solid rgba(148, 163, 184, 0.25);
  padding-top: 1rem;
}

.nowcast ul {
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  gap: 0.4rem;
}

.nowcast li {
  display: flex;
  justify-content: space-between;
}

.small {
  font-size: 0.75rem;
}

.empty {
  text-align: center;
  padding: 1.4rem 0;
}

.bad {
  color: #f87171;
}

.warn {
  color: #fbbf24;
}

.ok {
  color: #4ade80;
}

@media (max-width: 1024px) {
  .matrix-layout {
    grid-template-columns: 1fr;
  }
}
</style>
