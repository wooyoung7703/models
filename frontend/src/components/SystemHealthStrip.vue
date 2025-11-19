<template>
  <section class="health-strip">
    <article class="health-card">
      <header>WebSocket</header>
      <p :class="['value', wsConnected ? 'ok' : 'warn']">{{ wsConnected ? 'Connected' : 'Fallback' }}</p>
      <p class="hint">시도 {{ wsAttempts }}회</p>
    </article>

    <article class="health-card">
      <header>데이터 신선도</header>
      <p class="value">{{ lastUpdatedLabel }}</p>
      <p class="hint">{{ lastUpdated ? lastUpdated.toLocaleTimeString() : '미수신' }}</p>
    </article>

    <article class="health-card">
      <header>피처 커버리지</header>
      <p class="value">{{ healthySymbols }} / {{ totalSymbols }}</p>
      <p class="hint">정상 {{ coveragePct }}%</p>
    </article>

    <article class="health-card">
      <header>Trainer 상태</header>
      <p :class="['value', trainerHealthy ? 'ok' : 'warn']">{{ trainerLabel }}</p>
      <p class="hint">하트비트 {{ trainerHeartbeatAge }}</p>
    </article>

    <article v-if="entrySamples" class="health-card">
      <header>Entry Win Rate</header>
      <p class="value">{{ entryWinRateLabel }}</p>
      <p class="hint">samples {{ entrySamples }}</p>
    </article>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { TrainerMeta } from '../types/realtime'

const props = defineProps<{
  wsConnected: boolean
  wsAttempts: number
  lastUpdated: Date | null
  healthySymbols: number
  totalSymbols: number
  trainerMeta: TrainerMeta | null
  entryWinRate?: number | null
  entrySamples?: number | null
}>()

const coveragePct = computed(() => {
  if (!props.totalSymbols) return 0
  return Math.round((props.healthySymbols / props.totalSymbols) * 100)
})

const lastUpdatedLabel = computed(() => {
  if (!props.lastUpdated) return '미수신'
  const delta = Date.now() - props.lastUpdated.getTime()
  if (delta < 30_000) return '방금'
  if (delta < 60_000) return `${Math.round(delta / 1000)}초 전`
  if (delta < 3_600_000) return `${Math.round(delta / 60000)}분 전`
  return `${(delta / 3_600_000).toFixed(1)}시간`
})

const trainerHealthy = computed(() => Boolean(props.trainerMeta?.heartbeat_healthy))

const trainerHeartbeatAge = computed(() => {
  const age = props.trainerMeta?.heartbeat_age_seconds
  if (typeof age !== 'number') return 'n/a'
  if (age < 120) return '방금'
  const minutes = Math.round(age / 60)
  return `${minutes}분 전`
})

const trainerLabel = computed(() => (trainerHealthy.value ? 'Healthy' : 'Stale'))

const entryWinRateLabel = computed(() => {
  if (typeof props.entryWinRate !== 'number') return 'n/a'
  return `${(props.entryWinRate * 100).toFixed(1)}%`
})
</script>

<style scoped>
.health-strip {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.75rem;
}

.health-card {
  border: 1px solid rgba(96, 165, 250, 0.3);
  border-radius: 18px;
  padding: 0.85rem 1rem;
  background: rgba(8, 13, 24, 0.85);
}

header {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: rgba(148, 163, 184, 0.85);
  margin-bottom: 0.3rem;
}

.value {
  font-size: 1.35rem;
  font-weight: 600;
  margin: 0;
}

.value.ok {
  color: #34d399;
}

.value.warn {
  color: #fcd34d;
}

.hint {
  font-size: 0.78rem;
  color: rgba(148, 163, 184, 0.8);
}
</style>
