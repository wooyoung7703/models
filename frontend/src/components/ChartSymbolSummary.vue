<template>
  <aside class="symbol-summary" v-if="symbol">
    <header>
      <div>
        <p class="label">Watching</p>
        <h3>{{ symbol }}</h3>
        <p class="muted interval-hint">Interval · {{ interval }}</p>
      </div>
      <div class="header-actions">
        <button class="star" :class="{ active: favorite }" @click="toggleFavorite" aria-label="즐겨찾기 토글">
          {{ favorite ? '★' : '☆' }}
        </button>
        <span class="badge" :class="wsConnected ? 'ok' : 'warn'">
          {{ wsConnected ? 'WS live' : 'fallback' }}
        </span>
      </div>
    </header>

    <section class="metrics">
      <article>
        <p class="label">가격</p>
        <p class="value">{{ formatPrice(price) }}</p>
        <p class="hint">{{ priceSource }}</p>
      </article>
      <article>
        <p class="label">Stacking prob</p>
        <p class="value">{{ formatProb(stackProb) }}</p>
        <p class="hint">{{ postureLabel }}</p>
      </article>
      <article>
        <p class="label">Bottom score</p>
        <p class="value">{{ formatProb(bottomScore) }}</p>
        <p class="hint">{{ intervalLabel }}</p>
      </article>
    </section>

    <section class="details" v-if="components.length">
      <header>
        <h4>구성 요소</h4>
        <p class="muted">상위 {{ components.length }}개</p>
      </header>
      <ul>
        <li v-for="comp in components" :key="comp.key">
          <span>{{ comp.key }}</span>
          <strong :class="{ pos: comp.value >= 0, neg: comp.value < 0 }">{{ comp.value.toFixed(3) }}</strong>
        </li>
      </ul>
    </section>

    <section class="signals">
      <header>
        <h4>최근 신호</h4>
        <p class="muted">{{ recentSignals.length }}건</p>
      </header>
      <ul>
        <li v-for="sig in recentSignals" :key="sig.id">
          <span class="badge" :class="sig.side === 'sell' ? 'warn' : 'ok'">{{ sig.side }}</span>
          <div>
            <p>{{ sig.label || 'signal' }}</p>
            <small class="muted">{{ formatTime(sig.ts) }}</small>
          </div>
          <strong>{{ formatPrice(sig.price) }}</strong>
        </li>
        <li v-if="!recentSignals.length" class="muted empty">표시할 신호 없음</li>
      </ul>
    </section>
  </aside>
  <aside v-else class="symbol-summary empty">
    <p class="muted">표시할 심볼을 선택하세요.</p>
  </aside>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface ChartSignal {
  id: string | number
  ts: string
  side: 'buy' | 'sell'
  price?: number
  label?: string
}

type NowcastPayload = Record<string, any> | null

const emit = defineEmits<{ (e: 'toggle-favorite'): void }>()

const props = defineProps<{
  symbol: string
  nowcast: NowcastPayload
  signals: ChartSignal[]
  wsConnected: boolean
  interval: string
  favorite: boolean
}>()

function toggleFavorite() {
  emit('toggle-favorite')
}

const price = computed(() => props.nowcast?.price ?? null)
const priceSource = computed(() => props.nowcast?.price_source || 'spot')
const stackProb = computed(() => props.nowcast?.stacking?.prob_final ?? props.nowcast?.stacking?.prob ?? null)
const postureLabel = computed(() => {
  const decision = props.nowcast?.stacking?.decision
  if (decision === true) return 'long bias'
  if (decision === false) return 'flat bias'
  return 'pending'
})
const bottomScore = computed(() => props.nowcast?.bottom_score ?? null)
const intervalLabel = computed(() => props.nowcast?.interval || 'n/a')
const components = computed(() => {
  const entries = Object.entries(props.nowcast?.components || {})
    .sort((a, b) => Math.abs(b[1] as number) - Math.abs(a[1] as number))
    .slice(0, 5)
  return entries
    .map(([key, value]) => ({ key, value: typeof value === 'number' ? value : Number(value) }))
    .filter((entry) => Number.isFinite(entry.value))
})
const recentSignals = computed(() => props.signals.slice(-4).reverse())

function formatPrice(value?: number | null) {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'n/a'
  if (value >= 1000) return value.toFixed(1)
  if (value >= 10) return value.toFixed(2)
  return value.toFixed(4)
}

function formatProb(value?: number | null) {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'n/a'
  return value.toFixed(3)
}

function formatTime(iso: string) {
  if (!iso) return '—'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleTimeString()
}
</script>

<style scoped>
.symbol-summary {
  border: 1px solid rgba(79, 98, 143, 0.35);
  border-radius: 18px;
  padding: 1rem 1.2rem;
  background: rgba(7, 11, 23, 0.92);
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.symbol-summary.empty {
  align-items: center;
  justify-content: center;
  min-height: 200px;
}

header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

header .label {
  font-size: 0.75rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(148, 163, 184, 0.8);
}

.badge {
  padding: 0.25rem 0.7rem;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.4);
  font-size: 0.75rem;
  text-transform: uppercase;
}

.interval-hint {
  font-size: 0.78rem;
  margin-top: 0.2rem;
}

.star {
  background: transparent;
  border: 1px solid rgba(148, 163, 184, 0.35);
  border-radius: 8px;
  color: rgba(148, 163, 184, 0.8);
  padding: 0.2rem 0.45rem;
  cursor: pointer;
  font-size: 0.95rem;
}

.star.active {
  border-color: rgba(250, 204, 21, 0.6);
  color: #facc15;
}

.badge.ok {
  border-color: rgba(74, 222, 128, 0.5);
  color: #4ade80;
}

.badge.warn {
  border-color: rgba(248, 113, 113, 0.5);
  color: #f87171;
}

.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.75rem;
}

.metrics article {
  border: 1px solid rgba(99, 121, 173, 0.35);
  border-radius: 14px;
  padding: 0.65rem 0.75rem;
}

.metrics .value {
  font-size: 1.4rem;
  font-weight: 600;
}

.details ul,
.signals ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.details li,
.signals li {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border: 1px solid rgba(148, 163, 184, 0.25);
  border-radius: 12px;
  padding: 0.45rem 0.6rem;
}

.details li strong,
.signals li strong {
  font-variant-numeric: tabular-nums;
}

.signals li {
  gap: 0.5rem;
}

.signals li > div {
  flex: 1;
}

.pos {
  color: #4ade80;
}

.neg {
  color: #f87171;
}

.empty {
  text-align: center;
}
</style>
