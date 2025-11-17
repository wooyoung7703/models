<template>
  <section class="chart-pane">
    <header class="chart-toolbar">
      <div>
        <h2>실시간 차트</h2>
        <p class="muted">WS 기반 캔들 + 거래 신호</p>
      </div>
      <label>
        <span>심볼</span>
        <select v-model="innerSymbol">
          <option v-for="sym in symbols" :key="sym" :value="sym">{{ sym.toUpperCase() }}</option>
        </select>
      </label>
    </header>
    <RealtimeChart
      v-if="innerSymbol && apiBase"
      :symbol="innerSymbol"
      :apiBase="apiBase"
      interval="1m"
      :signals="chartSignals"
      :liveNowcast="chartNowcast"
      :wsConnected="wsConnected"
    />
    <p v-else class="muted empty-note">표시할 심볼이 없습니다.</p>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
// @ts-ignore script-setup SFC default export
import RealtimeChart from '../RealtimeChart.vue'

interface ChartSignal {
  id: string | number
  ts: string
  side: 'buy' | 'sell'
  price?: number
  label?: string
}

const props = defineProps<{
  symbols: string[]
  chartSymbol: string
  apiBase: string
  chartSignals: ChartSignal[]
  chartNowcast: Record<string, any> | null
  wsConnected: boolean
}>()

const emit = defineEmits<{
  (e: 'update:chartSymbol', value: string): void
}>()

const innerSymbol = computed({
  get: () => props.chartSymbol,
  set: (value: string) => emit('update:chartSymbol', value)
})
</script>

<style scoped>
.chart-pane {
  border: 1px solid rgba(81, 110, 163, 0.45);
  border-radius: 24px;
  background: var(--card);
  padding: 1.25rem 1.35rem 1.6rem;
  min-height: 520px;
}

.chart-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 1rem;
  margin-bottom: 1rem;
}

.chart-toolbar select {
  padding: 0.4rem 0.7rem;
  border-radius: 12px;
  border: 1px solid rgba(91, 118, 170, 0.45);
  background: rgba(15, 23, 42, 0.85);
  color: var(--text);
}
</style>
