<template>
  <section class="trades-view">
    <header class="head">
      <div>
        <h2>실시간 포지션 현황</h2>
        <p class="muted">WS 스트림 + HTTP 폴백 데이터 기반</p>
      </div>
      <div class="meta">
        <span v-if="latestTradeTime" class="muted">최근 갱신 · {{ latestTradeTime }}</span>
        <span class="pill" :class="{ ok: openCount > 0 }">{{ openCount }} open</span>
        <span class="pill" :class="{ warn: pendingCount > 0 }">{{ pendingCount }} pending</span>
      </div>
    </header>

    <div class="metric-grid">
      <article class="metric-card">
        <p class="label">평균 PnL 스냅샷</p>
        <p class="value" :class="pnlClass">{{ avgPnl }}</p>
        <p class="hint">{{ profitableShare }} 포지션이 이익 구간</p>
      </article>
      <article class="metric-card">
        <p class="label">평균 보유 시간</p>
        <p class="value">{{ avgHold }}</p>
        <p class="hint">활성 포지션 기준</p>
      </article>
      <article class="metric-card">
        <p class="label">24h 종료</p>
        <p class="value">{{ closed24h }}</p>
        <p class="hint">마지막 24시간 내 청산</p>
      </article>
      <article class="metric-card">
        <p class="label">쿨다운 잔여</p>
        <p class="value">{{ maxCooldown }}</p>
        <p class="hint">다음 증액까지 예상</p>
      </article>
    </div>

    <div class="content-grid">
      <article class="table-card">
        <header>
          <h3>포지션 테이블</h3>
          <span class="muted">상위 {{ tableRows.length }}건</span>
        </header>
        <table>
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Status</th>
              <th>Qty · Lev</th>
              <th>Entry → Avg</th>
              <th>TP / SL</th>
              <th>PnL</th>
              <th>Cooldown</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in tableRows" :key="row.id">
              <td>
                <strong>{{ row.symbol }}</strong>
                <p class="muted small">{{ durationLabel(row) }}</p>
              </td>
              <td>
                <span class="status" :class="row.status">{{ row.status }}</span>
              </td>
              <td>
                <p>{{ formatQty(row.quantity) }}</p>
                <p class="muted small">{{ formatLeverage(row.leverage) }}</p>
              </td>
              <td>
                <p>{{ formatPrice(row.entry_price) }}</p>
                <p class="muted small">→ {{ formatPrice(row.avg_price) }}</p>
              </td>
              <td>
                <p>{{ formatSignedPct(row.take_profit_pct) }}</p>
                <p class="muted small">/ {{ formatSignedPct(row.stop_loss_pct) }}</p>
              </td>
              <td>
                <p :class="{ pos: (row.pnl_pct_snapshot ?? 0) >= 0, neg: (row.pnl_pct_snapshot ?? 0) < 0 }">
                  {{ formatSignedPct(row.pnl_pct_snapshot) }}
                </p>
                <p class="muted small">adds {{ row.adds_done }}</p>
              </td>
              <td>
                <p>{{ formatCooldown(row.cooldown_seconds, row.next_add_in_seconds) }}</p>
                <p class="muted small">{{ formatNextAdd(row.next_add_in_seconds) }}</p>
              </td>
            </tr>
            <tr v-if="!tableRows.length">
              <td colspan="7" class="empty muted">표시할 포지션이 없습니다.</td>
            </tr>
          </tbody>
        </table>
      </article>

      <article>
        <TradeHistoryList :trades="recentTrades" />
      </article>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
// @ts-ignore script-setup default export shim
import TradeHistoryList from '../TradeHistoryList.vue'
import { formatSignedPct } from '../../utils'
import type { TradeRow } from '../../types/realtime'

const props = defineProps<{ trades: TradeRow[] }>()

const sortedTrades = computed(() => [...props.trades].sort(sortByCreated))
const tableRows = computed(() => sortedTrades.value.slice(0, 12))
const recentTrades = computed(() => sortedTrades.value.slice(0, 8))

const openCount = computed(() => sortedTrades.value.filter((row) => row.status === 'open').length)
const pendingCount = computed(() => sortedTrades.value.filter((row) => row.status === 'pending').length)
const closed24h = computed(() =>
  sortedTrades.value.filter((row) => row.status === 'closed' && withinHours(row.closed_at, 24)).length
)
const avgPnlValue = computed(() => {
  const set = sortedTrades.value.map((row) => row.pnl_pct_snapshot).filter(isFiniteNumber)
  if (!set.length) return 0
  return set.reduce((sum, value) => sum + (value || 0), 0) / set.length
})
const avgPnl = computed(() => formatSignedPct(avgPnlValue.value))
const pnlClass = computed(() => (avgPnlValue.value >= 0 ? 'pos' : 'neg'))
const profitableShare = computed(() => {
  const total = sortedTrades.value.length || 1
  const winners = sortedTrades.value.filter((row) => (row.pnl_pct_snapshot || 0) > 0).length
  return `${Math.round((winners / total) * 100)}%`
})
const avgHold = computed(() => {
  const active = sortedTrades.value.filter((row) => row.status !== 'closed')
  if (!active.length) return 'n/a'
  const mins =
    active
      .map((row) => holdMinutes(row))
      .filter((m) => m > 0)
      .reduce((sum, value, _, arr) => sum + value / arr.length, 0) || 0
  return mins >= 60 ? `${(mins / 60).toFixed(1)}h` : `${Math.round(mins)}m`
})
const maxCooldown = computed(() => {
  const cd = sortedTrades.value
    .map((row) => row.cooldown_seconds ?? row.next_add_in_seconds ?? 0)
    .filter((value) => typeof value === 'number' && value > 0)
  if (!cd.length) return '—'
  const max = Math.max(...cd)
  return secondsToLabel(max)
})
const latestTradeTime = computed(() => {
  const row = sortedTrades.value[0]
  if (!row?.created_at) return ''
  return new Date(row.created_at).toLocaleTimeString()
})

function durationLabel(row: TradeRow) {
  const minutes = holdMinutes(row)
  if (!minutes || Number.isNaN(minutes)) return '—'
  return minutes >= 60 ? `${(minutes / 60).toFixed(1)}h` : `${Math.round(minutes)}m`
}

function holdMinutes(row: TradeRow) {
  const start = Date.parse(row.created_at || row.closed_at || '')
  if (Number.isNaN(start)) return 0
  const end = row.closed_at ? Date.parse(row.closed_at) : Date.now()
  if (Number.isNaN(end)) return 0
  return (end - start) / 60000
}

function withinHours(dateValue: string | null | undefined, hours: number) {
  if (!dateValue) return false
  const target = Date.parse(dateValue)
  if (Number.isNaN(target)) return false
  return Date.now() - target <= hours * 3600 * 1000
}

function sortByCreated(a: TradeRow, b: TradeRow) {
  const av = Date.parse(a.created_at || a.closed_at || '')
  const bv = Date.parse(b.created_at || b.closed_at || '')
  if (Number.isNaN(av) && Number.isNaN(bv)) return 0
  if (Number.isNaN(av)) return 1
  if (Number.isNaN(bv)) return -1
  return bv - av
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function formatQty(qty?: number) {
  if (typeof qty !== 'number') return '—'
  return qty >= 100 ? qty.toFixed(0) : qty >= 1 ? qty.toFixed(2) : qty.toFixed(4)
}

function formatLeverage(lev?: number) {
  if (!lev) return '—'
  return `${lev.toFixed(1)}x`
}

function formatPrice(value?: number) {
  if (typeof value !== 'number') return '—'
  return value >= 1000 ? value.toFixed(1) : value >= 10 ? value.toFixed(2) : value.toFixed(4)
}

function secondsToLabel(value: number) {
  if (value >= 3600) return `${(value / 3600).toFixed(1)}h`
  if (value >= 60) return `${Math.round(value / 60)}m`
  return `${value}s`
}

function formatCooldown(cd?: number | null, nextAdd?: number | null) {
  const target = cd ?? nextAdd
  if (typeof target !== 'number') return '—'
  return secondsToLabel(target)
}

function formatNextAdd(next?: number | null) {
  if (typeof next !== 'number' || next <= 0) return 'ready'
  return `next add ${secondsToLabel(next)}`
}
</script>

<style scoped>
.trades-view {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.head {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 1rem;
}

.meta {
  display: flex;
  gap: 0.6rem;
  align-items: center;
}

.pill {
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.4);
  font-size: 0.8rem;
}

.pill.ok {
  border-color: rgba(74, 222, 128, 0.5);
  color: var(--ok, #4ade80);
}

.pill.warn {
  border-color: rgba(251, 191, 36, 0.5);
  color: #fbbf24;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
}

.metric-card {
  border: 1px solid rgba(92, 122, 179, 0.4);
  border-radius: 18px;
  padding: 0.9rem 1rem;
  background: rgba(8, 14, 27, 0.85);
}

.metric-card .label {
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.08em;
  color: rgba(203, 213, 225, 0.85);
}

.metric-card .value {
  font-size: 1.8rem;
  font-weight: 600;
  margin: 0.15rem 0;
}

.metric-card .value.pos {
  color: #4ade80;
}

.metric-card .value.neg {
  color: #f87171;
}

.metric-card .hint {
  font-size: 0.83rem;
  color: rgba(148, 163, 184, 0.85);
}

.content-grid {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 1.5rem;
}

.table-card {
  border: 1px solid rgba(59, 130, 246, 0.35);
  border-radius: 20px;
  padding: 1.1rem 1.2rem;
  background: rgba(7, 11, 23, 0.9);
}

.table-card table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}

th,
td {
  padding: 0.5rem 0.4rem;
  text-align: left;
}

th {
  text-transform: uppercase;
  font-size: 0.7rem;
  letter-spacing: 0.06em;
  color: rgba(148, 163, 184, 0.75);
}

tbody tr {
  border-top: 1px solid rgba(148, 163, 184, 0.2);
}

.status {
  text-transform: uppercase;
  font-size: 0.7rem;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.4);
}

.status.open {
  border-color: rgba(74, 222, 128, 0.6);
  color: #4ade80;
}

.status.pending {
  border-color: rgba(251, 191, 36, 0.5);
  color: #fbbf24;
}

.status.closed {
  border-color: rgba(248, 113, 113, 0.6);
  color: #f87171;
}

.small {
  font-size: 0.78rem;
}

.pos {
  color: #4ade80;
}

.neg {
  color: #f87171;
}

.empty {
  text-align: center;
  padding: 1.5rem 0;
}

@media (max-width: 960px) {
  .content-grid {
    grid-template-columns: 1fr;
  }
}
</style>
