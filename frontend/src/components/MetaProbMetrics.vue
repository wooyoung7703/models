<template>
  <div class="meta-prob-metrics" v-if="metrics">
    <header class="metrics-header">메타 재학습 메트릭</header>
    <ul class="metrics-list">
      <li><span class="label">현재 Brier</span><strong>{{ fmt(metrics.brier_new) }}</strong></li>
      <li v-if="metrics.prev_brier_old != null"><span class="label">이전 Brier</span><strong>{{ fmt(metrics.prev_brier_old) }}</strong></li>
      <li v-if="metrics.prev_rel_improve != null"><span class="label">상대 개선율</span><strong>{{ fmt(metrics.prev_rel_improve * 100) }}%</strong></li>
      <li v-if="metrics.retrain_samples != null"><span class="label">샘플 수</span><strong>{{ metrics.retrain_samples }}</strong></li>
    </ul>
  </div>
  <p v-else class="muted small">메타 확률 계수 없음</p>
</template>
<script setup lang="ts">
import { computed } from 'vue'
interface Metrics {
  brier_new?: number|null
  prev_brier_old?: number|null
  prev_rel_improve?: number|null
  retrain_samples?: number|null
}
const props = defineProps<{ stacking: any }>()
const metrics = computed<Metrics|undefined>(() => {
  const m = props.stacking?.bvf_meta_metrics
  if (!m) return undefined
  return {
    brier_new: m.brier_new ?? null,
    prev_brier_old: m.prev_brier_old ?? null,
    prev_rel_improve: m.prev_rel_improve ?? null,
    retrain_samples: m.retrain_samples ?? null,
  }
})
function fmt(v: number|undefined|null) {
  if (v == null || isNaN(v)) return '–'
  return v.toFixed(4)
}
</script>
<style scoped>
.meta-prob-metrics { margin-top: 0.5rem; }
.metrics-header { font-weight: 600; font-size: 0.85rem; margin-bottom: 0.25rem; }
.metrics-list { list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: repeat(auto-fit,minmax(120px,1fr)); gap: 4px; }
.metrics-list li { background: var(--card-bg,var(--c-bg-alt,#1e1e1e)); padding: 4px 6px; border-radius: 4px; font-size: 0.75rem; display: flex; flex-direction: column; }
.label { opacity: 0.7; font-size: 0.65rem; }
.muted.small { font-size: 0.7rem; opacity: 0.7; }
</style>
<script lang="ts">export default {}</script>
