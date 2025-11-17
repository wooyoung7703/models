<template>
  <section class="trade-history-card">
    <header>
      <h3>실시간 거래 내역</h3>
      <span class="muted">최근 {{ trades.length }}건</span>
    </header>
    <ul>
      <li v-for="row in trades" :key="row.id">
        <div>
          <strong>{{ row.symbol?.toUpperCase() }}</strong>
          <span class="muted">{{ shortTime(row.created_at || row.open_time) }}</span>
        </div>
        <div class="meta">
          <span class="status" :class="row.status">{{ row.status }}</span>
          <span>{{ formatSignedPct(row.pnl_pct_snapshot ?? 0) }}</span>
        </div>
      </li>
      <li v-if="!trades.length" class="muted empty">표시할 거래가 없습니다.</li>
    </ul>
  </section>
</template>

<script setup lang="ts">
import { formatSignedPct } from '../utils'

interface TradeRow {
  id: number | string
  symbol: string
  status: string
  created_at?: string
  open_time?: string
  pnl_pct_snapshot?: number
}

defineProps<{ trades: TradeRow[] }>()

const shortTime = (value?: string) => {
  if (!value) return '—'
  try {
    return new Date(value).toLocaleTimeString()
  } catch {
    return value
  }
}
</script>

<style scoped>
.trade-history-card {
  border: 1px solid rgba(96, 165, 250, 0.35);
  border-radius: 20px;
  padding: 1rem 1.1rem;
  margin-bottom: 1.2rem;
  background: rgba(7, 12, 24, 0.92);
}
ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.45rem;
}
li {
  border: 1px solid rgba(148, 163, 184, 0.28);
  border-radius: 14px;
  padding: 0.55rem 0.7rem;
  display: flex;
  justify-content: space-between;
  font-size: 0.82rem;
}
.meta {
  display: flex;
  gap: 0.5rem;
  align-items: center;
}
.status {
  text-transform: uppercase;
  font-size: 0.72rem;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.4);
}
.status.open {
  border-color: rgba(34, 197, 94, 0.4);
  color: var(--ok);
}
.status.closed {
  border-color: rgba(248, 113, 113, 0.4);
  color: var(--bad);
}
.empty {
  justify-content: center;
}
</style>
