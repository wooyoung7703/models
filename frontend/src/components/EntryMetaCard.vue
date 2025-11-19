<template>
  <article class="meta-card" v-if="entryMetrics">
    <header>
      <div>
        <p class="label">Entry Meta</p>
        <h3>{{ winRateLabel }}</h3>
      </div>
      <span class="badge">samples {{ samplesLabel }}</span>
    </header>
    <ul>
      <li>
        <span class="muted">Target</span>
        <strong>{{ targetLabel }}</strong>
      </li>
      <li>
        <span class="muted">Effective</span>
        <strong>{{ effectiveLabel }}</strong>
      </li>
      <li>
        <span class="muted">Threshold</span>
        <div class="thresholds">
          <span v-if="entryMetrics.threshold?.env">env {{ entryMetrics.threshold.env.toFixed(2) }}</span>
          <span v-if="entryMetrics.threshold?.dynamic">dyn {{ entryMetrics.threshold.dynamic.toFixed(2) }}</span>
          <span v-if="!hasThreshold" class="muted">n/a</span>
        </div>
      </li>
    </ul>
  </article>
  <article v-else class="meta-card missing">
    <p class="label">Entry Meta</p>
    <p class="muted">데이터 없음</p>
  </article>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { EntryMetrics } from '../types/realtime'

const props = defineProps<{ entryMetrics: EntryMetrics | null }>()

const winRateLabel = computed(() => {
  const value = props.entryMetrics?.overall?.win_rate
  return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : 'n/a'
})

const samplesLabel = computed(() => props.entryMetrics?.overall?.samples ?? '0')

const targetLabel = computed(() => formatPercent(props.entryMetrics?.target))
const effectiveLabel = computed(() => formatPercent(props.entryMetrics?.target_effective))

const hasThreshold = computed(() => Boolean(props.entryMetrics?.threshold?.env || props.entryMetrics?.threshold?.dynamic))

function formatPercent(value?: number | null) {
  if (typeof value !== 'number') return 'n/a'
  return value >= 1 ? `${value.toFixed(2)}` : `${(value * 100).toFixed(1)}%`
}
</script>

<style scoped>
.meta-card {
  border: 1px solid rgba(148, 163, 184, 0.4);
  border-radius: 18px;
  padding: 0.9rem 1rem;
  background: rgba(12, 19, 32, 0.9);
}

header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.label {
  margin: 0;
  text-transform: uppercase;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  color: rgba(148, 163, 184, 0.85);
}

h3 {
  margin: 0.1rem 0 0;
  font-size: 1.35rem;
}

.badge {
  font-size: 0.75rem;
  border: 1px solid rgba(14, 165, 233, 0.5);
  border-radius: 999px;
  padding: 0.1rem 0.6rem;
  color: #7dd3fc;
}

ul {
  list-style: none;
  margin: 0.8rem 0 0;
  padding: 0;
  display: grid;
  gap: 0.35rem;
}

li {
  display: flex;
  justify-content: space-between;
  gap: 0.6rem;
  font-size: 0.9rem;
}

.muted {
  color: rgba(148, 163, 184, 0.8);
}

.thresholds {
  display: flex;
  gap: 0.35rem;
  flex-wrap: wrap;
}

.thresholds span {
  border: 1px solid rgba(148, 163, 184, 0.4);
  border-radius: 10px;
  padding: 0.05rem 0.45rem;
  font-size: 0.78rem;
}

.meta-card.missing {
  text-align: center;
}
</style>
