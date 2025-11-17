<template>
  <section class="trade-signal-card">
    <header>
      <h3>실시간 거래 신호</h3>
      <span class="muted">{{ wsConnected ? 'WS ONLINE' : 'HTTP FALLBACK' }}</span>
    </header>

    <div class="grid">
      <div>
        <label>오픈 포지션</label>
        <strong>{{ openTrades.length }}건</strong>
      </div>
      <div>
        <label>활성 스택 신호</label>
        <strong>{{ activeSignals }}개</strong>
      </div>
      <div>
        <label>최근 심볼</label>
        <strong>{{ latestSymbol }}</strong>
      </div>
      <div>
        <label>최근 PnL</label>
        <strong :class="pnlClass">{{ latestPnl }}</strong>
      </div>
    </div>

    <div class="latest" v-if="latest">
      <p>
        <span class="badge" :class="latest.status">{{ latest.status?.toUpperCase() }}</span>
        {{ latest.symbol?.toUpperCase() }} · {{ latestSide }}
      </p>
      <p class="muted">
        진입 {{ formatNumber(latest.entry_price) }} / 현가
        {{ latest.last_price != null ? formatNumber(latest.last_price) : 'n/a' }} ·
        레버리지 {{ latest.leverage }}x
      </p>
    </div>
    <p v-else class="muted">거래 데이터가 아직 없습니다.</p>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { formatSignedPct } from '../utils'

interface TradeRow {
  id: number | string
  symbol: string
  status: string
  side?: string
  direction?: string
  pnl_pct_snapshot?: number
  entry_price?: number
  last_price?: number | null
  leverage?: number
}

const props = defineProps<{
  latest: TradeRow | null
  openTrades: TradeRow[]
  activeSignals: number
  wsConnected: boolean
}>()

const latestSymbol = computed(() => props.latest?.symbol?.toUpperCase() || 'N/A')
const latestSide = computed(() => props.latest?.side || props.latest?.direction || 'side?')
const latestPnl = computed(() =>
  props.latest?.pnl_pct_snapshot != null ? formatSignedPct(props.latest.pnl_pct_snapshot) : '–'
)
const pnlClass = computed(() => {
  if (props.latest?.pnl_pct_snapshot == null) return ''
  if (props.latest.pnl_pct_snapshot > 0) return 'pos'
  if (props.latest.pnl_pct_snapshot < 0) return 'neg'
  return ''
})
const formatNumber = (value?: number) => {
  if (value == null || Number.isNaN(value)) return 'n/a'
  if (Math.abs(value) >= 100) return value.toFixed(2)
  if (Math.abs(value) >= 1) return value.toFixed(3)
  return value.toFixed(4)
}
</script>

<style scoped>
.trade-signal-card {
  border: 1px solid rgba(34, 197, 94, 0.4);
  border-radius: 20px;
  padding: 1.1rem;
  margin-bottom: 1.2rem;
  background: rgba(10, 17, 32, 0.92);
}
header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 0.8rem;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.8rem;
  margin-bottom: 0.8rem;
}
label {
  display: block;
  font-size: 0.75rem;
  color: var(--muted);
  margin-bottom: 0.15rem;
}
strong {
  font-size: 1.1rem;
}
strong.pos {
  color: var(--ok);
}
strong.neg {
  color: var(--bad);
}
.latest {
  border-top: 1px solid rgba(148, 163, 184, 0.25);
  padding-top: 0.8rem;
  font-size: 0.9rem;
}
.badge {
  display: inline-flex;
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  margin-right: 0.4rem;
  font-size: 0.7rem;
  border: 1px solid rgba(148, 163, 184, 0.4);
}
.badge.open {
  border-color: rgba(96, 165, 250, 0.5);
  color: #bfdbfe;
}
.badge.closed {
  border-color: rgba(248, 113, 113, 0.5);
  color: #fecaca;
}
</style>
