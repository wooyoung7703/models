<template>
  <div class="prob-bars">
    <!-- BASE vs ADJ (조정 전/후) -->
    <div class="bar-row" v-if="stacking.prob_final_base != null">
      <span>BASE</span>
      <div class="progress"><span class="base" :style="{ width: pctWidth(stacking.prob_final_base) }"></span></div>
      <strong>{{ formatPct(stacking.prob_final_base) }}</strong>
    </div>
    <div class="bar-row" v-if="stacking.prob_final != null">
      <span>ADJ</span>
      <div class="progress"><span class="adj" :style="{ width: pctWidth(stacking.prob_final) }"></span></div>
      <strong>{{ formatPct(stacking.prob_final) }}</strong>
    </div>
    <div class="bar-row">
      <span>RAW</span>
      <div class="progress"><span :style="{ width: pctWidth(stacking.prob_raw ?? stacking.raw_prob) }"></span></div>
      <strong>{{ formatPct(stacking.prob_raw ?? stacking.raw_prob ?? 0) }}</strong>
    </div>
    <div class="bar-row">
      <span>SMOOTH</span>
      <div class="progress"><span class="smooth" :style="{ width: pctWidth(stacking.prob_smoothed ?? stacking.prob) }"></span></div>
      <strong>{{ formatPct(stacking.prob_smoothed ?? stacking.prob) }}</strong>
    </div>
    <div class="bar-row">
      <span>FINAL</span>
      <div class="progress"><span class="final" :style="{ width: pctWidth(stacking.prob_final ?? stacking.prob) }"></span></div>
      <strong>{{ formatPct(stacking.prob_final ?? stacking.prob) }}</strong>
    </div>
    <!-- Delta 즉시값 및 롤링 평균 -->
    <div class="bar-row" v-if="stacking.delta != null">
      <span>Δ</span>
      <div class="progress delta-wrap"><span class="delta" :class="deltaClass(stacking.delta)" :style="{ width: deltaWidth(stacking.delta) }"></span></div>
      <strong :class="deltaClass(stacking.delta)">{{ formatSignedPct(stacking.delta) }}</strong>
    </div>
    <div class="bar-row" v-if="stacking.delta_mean_50 != null">
      <span>Δ50</span>
      <div class="progress delta-wrap"><span class="delta mean" :class="deltaClass(stacking.delta_mean_50)" :style="{ width: deltaWidth(stacking.delta_mean_50) }"></span></div>
      <strong :class="deltaClass(stacking.delta_mean_50)">{{ formatSignedPct(stacking.delta_mean_50) }}</strong>
    </div>
    <div class="bar-row" v-if="stacking.delta_mean_200 != null">
      <span>Δ200</span>
      <div class="progress delta-wrap"><span class="delta mean" :class="deltaClass(stacking.delta_mean_200)" :style="{ width: deltaWidth(stacking.delta_mean_200) }"></span></div>
      <strong :class="deltaClass(stacking.delta_mean_200)">{{ formatSignedPct(stacking.delta_mean_200) }}</strong>
    </div>
    <div class="bar-row" v-if="stacking.delta_abs_mean_200 != null">
      <span>|Δ|200</span>
      <div class="progress delta-wrap"><span class="delta abs" :style="{ width: deltaWidth(stacking.delta_abs_mean_200) }"></span></div>
      <strong>{{ formatSignedPct(stacking.delta_abs_mean_200) }}</strong>
    </div>
    <div class="bar-row threshold">
      <span>THRESH</span>
      <div class="progress"><span class="threshold" :style="{ width: pctWidth(threshold) }"></span></div>
      <strong>{{ formatThreshold(threshold) }}</strong>
    </div>
  </div>
</template>

<script setup lang="ts">
import { formatPct, formatSignedPct } from '../utils'

interface StackingLike {
  prob?: number
  prob_raw?: number | null
  raw_prob?: number | null
  prob_smoothed?: number | null
  prob_final?: number | null
  prob_final_base?: number | null
  threshold?: number | null
  prob_final_adjusted?: number | null
  delta?: number | null
  delta_mean_50?: number | null
  delta_mean_200?: number | null
  delta_abs_mean_200?: number | null
}

const props = defineProps<{ stacking: StackingLike; threshold: number | null | undefined }>()

const pctWidth = (value?: number | null) => {
  if (value == null || isNaN(Number(value))) return '0%'
  const pct = Math.max(0, Math.min(1, Number(value)))
  return `${(pct * 100).toFixed(0)}%`
}
const formatThreshold = (value?: number | null) => (value == null || Number.isNaN(Number(value))) ? 'auto' : Number(value).toFixed(2)
const deltaWidth = (value?: number | null) => {
  if (value == null || Number.isNaN(Number(value))) return '0%'
  const mag = Math.min(0.2, Math.abs(value)) / 0.2
  return `${(mag * 100).toFixed(0)}%`
}
const deltaClass = (value?: number | null) => {
  if (value == null || Number.isNaN(Number(value))) return 'delta-zero'
  if (value > 0.005) return 'delta-pos'
  if (value < -0.005) return 'delta-neg'
  return 'delta-flat'
}
</script>

<style scoped>
.prob-bars { display: grid; gap: 0.6rem; }
.bar-row { display: grid; grid-template-columns: 60px 1fr auto; gap: 0.75rem; align-items: center; font-size: 0.8rem; }
.progress { position: relative; height: 8px; background: rgba(129,161,214,0.18); border-radius: 999px; overflow: hidden; }
.progress span { position: absolute; inset: 0; background: linear-gradient(90deg, rgba(96,165,250,0.2), rgba(96,165,250,0.7)); }
.progress span.smooth { background: linear-gradient(90deg, rgba(129,140,248,0.25), rgba(129,140,248,0.7)); }
.progress span.final { background: linear-gradient(90deg, rgba(14,165,233,0.3), rgba(14,165,233,0.8)); }
.progress span.threshold { background: linear-gradient(90deg, rgba(245,158,11,0.3), rgba(245,158,11,0.7)); }
.progress span.base { background: linear-gradient(90deg, rgba(148,163,184,0.35), rgba(148,163,184,0.7)); }
.progress span.adj { background: linear-gradient(90deg, rgba(16,185,129,0.35), rgba(16,185,129,0.75)); }
.progress.delta-wrap { background: rgba(129,161,214,0.15); }
.progress span.delta { background: linear-gradient(90deg, rgba(59,130,246,0.25), rgba(59,130,246,0.6)); }
.progress span.delta.mean { background: linear-gradient(90deg, rgba(99,102,241,0.3), rgba(99,102,241,0.7)); }
.progress span.delta.abs { background: linear-gradient(90deg, rgba(14,165,233,0.28), rgba(14,165,233,0.75)); }
.delta-pos { color: var(--ok); }
.delta-neg { color: var(--bad); }
.delta-flat, .delta-zero { color: var(--muted); }
.delta-zero { opacity: 0.7; }
</style>