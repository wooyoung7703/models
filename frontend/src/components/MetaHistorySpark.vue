<template>
  <div class="meta-spark" :class="rootClass" v-if="points.length">
    <svg :width="svgWidth" :height="height" :viewBox="`0 0 ${svgWidth} ${height}`">
      <polyline :points="polyline" :stroke="lineColor" fill="none" stroke-width="1"/>
      <circle v-for="(pt,i) in points" :key="i" :cx="pt.x" :cy="pt.y" r="2" :class="pt.cls" />
    </svg>
    <div class="legend">
      <span v-if="last">최근 개선: <strong :class="last.rel_improve>0?'pos':'neg'">{{ formatRel(last.rel_improve) }}</strong></span>
      <span v-if="avgRel != null"> / 평균: <strong :class="avgRel>0?'pos':'neg'">{{ formatRel(avgRel) }}</strong></span>
      <span v-if="winRate != null" class="wr">승률: <strong :class="winRate>0.5?'pos':(winRate<0.5?'neg':'zero')">{{ (winRate*100).toFixed(1) }}%</strong> <small>({{ posCount }}/{{ posCount+negCount }} / Ø0={{ zeroCount }})</small></span>
      <span v-if="warnNeg" class="warn-tag">Neg Streak 경고</span>
      <span v-if="driftWarn" class="drift-tag">Coef Drift 경고</span>
    </div>
  </div>
  <div v-else class="meta-spark empty">메타 히스토리 없음</div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface MetaRec { ts?: string|null; rel_improve?: number|null; brier_new?: number|null; brier_old?: number|null; overwritten?: boolean; samples?: number|null }
const props = defineProps<{ history: MetaRec[] | undefined; warnNeg?: boolean; driftWarn?: boolean }>()
const avgRel = computed(() => {
  if (!relValues.value.length) return null
  const sum = relValues.value.reduce((a,b)=>a+b,0)
  return sum / relValues.value.length
})
const lineColor = computed(() => {
  if (props.driftWarn) return '#c0392b'
  if (props.warnNeg) return '#d35400'
  return 'var(--spark-line,#4a90e2)'
})
const rootClass = computed(() => ({ 'warn-neg': props.warnNeg, 'warn-drift': props.driftWarn }))

const height = 34
const pad = 4
const maxPoints = 40
const data = computed(() => (props.history || []).slice(-maxPoints))

const relValues = computed(() => data.value.map(d => typeof d.rel_improve === 'number' ? d.rel_improve : 0))
const maxAbs = computed(() => Math.max(0.01, ...relValues.value.map(v => Math.abs(v))))

const points = computed(() => {
  const n = data.value.length
  if (!n) return [] as {x:number;y:number;cls:string}[]
  const w = Math.max(60, n * 6)
  return data.value.map((d,i) => {
    const rel = typeof d.rel_improve === 'number' ? d.rel_improve : 0
    // y: center baseline height/2, move up for positive, down for negative
    const base = height/2
    const scale = (height/2 - pad) / maxAbs.value
    const y = base - (rel * scale)
    const x = (i/(n-1)) * (w - pad*2) + pad
    return { x, y, cls: rel > 0 ? 'pos' : (rel < 0 ? 'neg' : 'zero') }
  })
})

const svgWidth = computed(() => {
  const n = data.value.length
  return Math.max(60, n * 6)
})

const polyline = computed(() => points.value.map(p => `${p.x},${p.y}`).join(' '))
const last = computed(() => data.value[data.value.length-1])

const formatRel = (v?: number|null) => {
  if (v == null || isNaN(v)) return 'n/a'
  return (v*100).toFixed(2)+'%'
}

// Win-rate 계산: 양수 개선(>0) vs 음수 개선(<0) 비율
const posCount = computed(() => relValues.value.filter(v => v > 0).length)
const negCount = computed(() => relValues.value.filter(v => v < 0).length)
const zeroCount = computed(() => relValues.value.filter(v => v === 0).length)
const winRate = computed(() => {
  const p = posCount.value; const n = negCount.value
  const denom = p + n
  if (denom === 0) return null
  return p / denom
})
</script>

<style scoped>
.meta-spark { display:flex; flex-direction:column; gap:2px; }
.meta-spark svg { width:100%; max-width:100%; overflow:visible; }
circle.pos { fill: #2e8b57; }
circle.neg { fill: #c0392b; }
circle.zero { fill: #888; }
.legend { font-size: 11px; color: var(--text-muted,#666); }
.legend strong.pos { color:#2e8b57; }
.legend strong.neg { color:#c0392b; }
.empty { font-size:11px; color:#999; }
.legend .wr { margin-left:6px; }
.warn-tag, .drift-tag { margin-left:6px; font-size:10px; padding:2px 4px; border-radius:3px; }
.warn-tag { background:#d35400; color:#fff; }
.drift-tag { background:#c0392b; color:#fff; }
.warn-neg { background: rgba(211,84,0,0.08); border:1px solid rgba(211,84,0,0.3); padding:4px; border-radius:4px; }
.warn-drift { background: rgba(192,57,43,0.08); border:1px solid rgba(192,57,43,0.3); padding:4px; border-radius:4px; }
</style>