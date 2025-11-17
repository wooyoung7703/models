<template>
  <section class="model-status-card">
    <header>
      <h3>모델 학습 상태</h3>
      <p class="muted">각 모델의 최신 학습 결과와 품질 지표</p>
    </header>
    <div class="status-list">
      <article v-for="item in items" :key="item.key">
        <div class="row-head">
          <strong>{{ item.name }}</strong>
          <span class="pill" :class="stateClass(item.status)">{{ labelMap[item.status] || item.status }}</span>
        </div>
        <div class="meta-line">
          <span>마지막 실행: {{ item.lastRun || '정보 없음' }}</span>
          <span v-if="item.samples">샘플 {{ item.samples }}</span>
        </div>
        <div class="bar-line">
          <label>품질</label>
          <div class="bar">
            <span :style="{ width: qualityWidth(item.quality) }"></span>
          </div>
          <span class="metric">{{ qualityText(item.quality) }}</span>
        </div>
        <div class="bar-line" v-if="item.loss != null">
          <label>Loss</label>
          <div class="bar alt">
            <span :style="{ width: lossWidth(item.loss) }"></span>
          </div>
          <span class="metric">{{ item.loss.toFixed(4) }}</span>
        </div>
      </article>
    </div>
  </section>
</template>

<script setup lang="ts">
interface ModelStatus {
  key: string
  name: string
  status?: string
  lastRun?: string | null
  loss?: number | null
  samples?: number | null
  quality?: number | null
}
const props = defineProps<{ items: ModelStatus[] }>()
const labelMap: Record<string, string> = {
  ready: '준비',
  running: '학습 중',
  idle: '대기',
  error: '오류'
}
const stateClass = (status?: string) => ({
  running: 'warn',
  error: 'bad'
}[status || ''] || 'ok')
const qualityWidth = (value?: number | null) => {
  if (value == null || Number.isNaN(value)) return '0%'
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`
}
const qualityText = (value?: number | null) => {
  if (value == null || Number.isNaN(value)) return 'n/a'
  return `${(value * 100).toFixed(1)}%`
}
const lossWidth = (loss?: number | null) => {
  if (loss == null || loss <= 0) return '4%'
  const clamped = Math.min(loss, 2)
  return `${(1 - clamped / 2) * 100}%`
}
</script>

<style scoped>
.model-status-card {
  border: 1px solid rgba(129, 161, 214, 0.32);
  border-radius: 24px;
  padding: 1.4rem 1.2rem;
  background: linear-gradient(180deg, rgba(16, 24, 44, 0.95), rgba(9, 14, 28, 0.92));
  margin-bottom: 1.5rem;
  box-shadow: 0 30px 60px -40px rgba(6, 10, 20, 0.8);
}
.status-list {
  display: grid;
  gap: 1rem;
}
article {
  border: 1px solid rgba(99, 121, 173, 0.3);
  border-radius: 18px;
  padding: 0.9rem 1rem;
  background: rgba(8, 13, 25, 0.8);
}
.row-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.pill {
  padding: 0.2rem 0.7rem;
  border-radius: 999px;
  font-size: 0.72rem;
  border: 1px solid transparent;
}
.pill.ok { border-color: rgba(74, 222, 128, 0.35); color: #4ade80; }
.pill.warn { border-color: rgba(234, 179, 8, 0.4); color: #facc15; }
.pill.bad { border-color: rgba(248, 113, 113, 0.4); color: #f87171; }
.meta-line {
  display: flex;
  justify-content: space-between;
  margin: 0.4rem 0 0.6rem;
  font-size: 0.78rem;
  color: var(--muted);
}
.bar-line {
  display: grid;
  grid-template-columns: 60px 1fr auto;
  gap: 0.75rem;
  align-items: center;
  font-size: 0.78rem;
}
.bar {
  position: relative;
  height: 8px;
  border-radius: 999px;
  background: rgba(79, 98, 143, 0.35);
  overflow: hidden;
}
.bar.alt {
  background: rgba(248, 113, 113, 0.15);
}
.bar span {
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, rgba(14, 165, 233, 0.2), rgba(14, 165, 233, 0.8));
}
.bar.alt span {
  background: linear-gradient(90deg, rgba(248, 113, 113, 0.2), rgba(248, 113, 113, 0.7));
}
.metric {
  font-variant-numeric: tabular-nums;
}
</style>
