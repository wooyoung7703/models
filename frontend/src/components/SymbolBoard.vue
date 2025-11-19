<template>
  <section class="symbol-board">
    <header>
      <div>
        <h3>Symbol Board</h3>
        <p class="muted">Live stacking + feature freshness</p>
      </div>
      <div class="controls">
        <input
          v-model.trim="filterText"
          type="search"
          placeholder="Filter (e.g. BTC)"
        />
        <select v-model="sortKey">
          <option value="score">Score</option>
          <option value="margin">Margin</option>
          <option value="fresh">Freshness</option>
        </select>
      </div>
    </header>

    <TransitionGroup name="card-list" tag="div" class="card-grid">
      <SymbolCard
        v-for="row in sortedRows"
        :key="row.symbol"
        v-bind="row"
        :selected="row.symbol === selectedSymbol"
        @select="onSelect"
      />
    </TransitionGroup>

    <p v-if="!sortedRows.length" class="empty">No symbols available.</p>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import SymbolCard from './SymbolCard.vue'
import type { FeatureSnapshot, NowcastEntry } from '../types/realtime'

const props = defineProps<{
  nowcasts: Record<string, NowcastEntry>
  featureHealth: Record<string, FeatureSnapshot>
  selectedSymbol: string
}>()

const emit = defineEmits<{ (e: 'select', symbol: string): void }>()

const initialFilter = typeof window !== 'undefined' ? localStorage.getItem('symbol-board-filter') : ''
const filterText = ref(initialFilter ?? '')
const sortKey = ref<'score' | 'margin' | 'fresh'>('score')

watch(filterText, (value: string) => {
  if (typeof window === 'undefined') return
  localStorage.setItem('symbol-board-filter', value)
})

const normalizedRows = computed(() => {
  return Object.entries(props.nowcasts)
    .filter(([symbol]) => !symbol.startsWith('_'))
    .map(([symbol, entry]) => {
      const feature = props.featureHealth[symbol]
      const score = entry?.stacking?.prob_final ?? entry?.stacking?.prob ?? entry?.bottom_score ?? null
      const components = entry?.components ?? null
      const margin = entry?.stacking?.margin ?? null
      const decision = entry?.stacking?.decision ?? null
      return {
        symbol,
        interval: entry?.interval ?? '1m',
        updatedAt: entry?.timestamp ?? null,
        price: entry?.price ?? null,
        score,
        margin,
        decision,
        components,
        featureFreshSeconds: feature?.data_fresh_seconds ?? null,
      }
    })
})

const filteredRows = computed(() => {
  const query = filterText.value.toLowerCase()
  if (!query) return normalizedRows.value
  return normalizedRows.value.filter((row) => row.symbol.toLowerCase().includes(query))
})

const sortedRows = computed(() => {
  const rows = [...filteredRows.value]
  const key = sortKey.value
  const fallback = (value: number | null) => (typeof value === 'number' ? value : -Infinity)

  rows.sort((a, b) => {
    if (key === 'fresh') {
      const freshA = a.featureFreshSeconds ?? Infinity
      const freshB = b.featureFreshSeconds ?? Infinity
      return freshA - freshB
    }
    if (key === 'margin') {
      return fallback(b.margin) - fallback(a.margin)
    }
    return fallback(b.score) - fallback(a.score)
  })

  return rows
})

function onSelect(symbol: string) {
  emit('select', symbol)
}
</script>

<style scoped>
.symbol-board {
  background: rgba(2, 6, 23, 0.7);
  border: 1px solid rgba(15, 23, 42, 0.9);
  border-radius: 24px;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1.2rem;
}

header {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}

.controls {
  display: flex;
  gap: 0.6rem;
}

input,
select {
  background: rgba(15, 23, 42, 0.6);
  border: 1px solid rgba(51, 65, 85, 0.8);
  border-radius: 999px;
  padding: 0.35rem 0.9rem;
  color: #e2e8f0;
}

.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 1rem;
}

.empty {
  text-align: center;
  color: rgba(148, 163, 184, 0.8);
}

.card-list-enter-active,
.card-list-leave-active {
  transition: all 0.2s ease;
}

.card-list-enter-from,
.card-list-leave-to {
  opacity: 0;
  transform: scale(0.95);
}
</style>
