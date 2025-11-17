<template>
  <section class="low-price-card">
    <header>
      <h3>지금 저가 구간인가?</h3>
      <p class="muted">나우캐스트 신호와 활성 비율 기반</p>
    </header>
    <div class="status-summary" :class="statusClass">
      <h2>{{ headline }}</h2>
      <p>{{ description }}</p>
    </div>
    <div class="metrics">
      <div>
        <label>평균 바텀 점수</label>
        <strong>{{ avgBottomPct }}</strong>
      </div>
      <div>
        <label>스택 활성 비율</label>
        <strong>{{ activeShare }}%</strong>
      </div>
      <div>
        <label>고신뢰 비율</label>
        <strong>{{ highShare }}%</strong>
      </div>
      <div>
        <label>WS 상태</label>
        <strong :class="wsConnected ? 'ok' : 'bad'">{{ wsConnected ? '연결' : '폴백' }}</strong>
      </div>
    </div>
    <footer>
      마지막 업데이트: <span>{{ updatedAgo }}</span>
    </footer>
  </section>
</template>

<script setup lang="ts">
const props = defineProps<{
  avgBottom: number | null
  activeShare: number
  highShare: number
  updatedAgo: string
  wsConnected: boolean
}>()
const score = computed(() => {
  const bottom = props.avgBottom ?? 0
  return bottom * 0.6 + (props.activeShare / 100) * 0.25 + (props.highShare / 100) * 0.15
})
const headline = computed(() => {
  if (score.value >= 0.65) return '고저점 경보'
  if (score.value >= 0.5) return '저가 근처'
  if (score.value >= 0.35) return '중립'
  return '저가 아님'
})
const description = computed(() => {
  if (score.value >= 0.65) return '복수 모델이 적극적으로 저점을 시사합니다.'
  if (score.value >= 0.5) return '저가 가능성이 높으며 추이를 주시하세요.'
  if (score.value >= 0.35) return '확신 없는 혼조 구간입니다.'
  return '시장은 아직 저가 시그널을 주지 않았습니다.'
})
const statusClass = computed(() => {
  if (score.value >= 0.65) return 'hot'
  if (score.value >= 0.5) return 'warm'
  if (score.value >= 0.35) return 'neutral'
  return 'cold'
})
const avgBottomPct = computed(() => {
  if (props.avgBottom == null) return '데이터 없음'
  return `${(props.avgBottom * 100).toFixed(1)}%`
})
</script>

<style scoped>
.low-price-card {
  border: 1px solid rgba(129, 161, 214, 0.32);
  border-radius: 24px;
  padding: 1.4rem 1.2rem;
  background: linear-gradient(180deg, rgba(11, 18, 34, 0.95), rgba(6, 10, 22, 0.92));
  margin-bottom: 1.5rem;
}
.status-summary {
  padding: 0.9rem 1rem;
  border-radius: 18px;
  margin: 0.8rem 0 1rem;
  background: rgba(15, 23, 42, 0.8);
  border: 1px solid transparent;
}
.status-summary.hot { border-color: rgba(248, 113, 113, 0.5); }
.status-summary.warm { border-color: rgba(251, 191, 36, 0.5); }
.status-summary.neutral { border-color: rgba(96, 165, 250, 0.35); }
.status-summary.cold { border-color: rgba(148, 163, 184, 0.35); }
.status-summary h2 {
  margin: 0 0 0.3rem;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.8rem;
}
.metrics div {
  background: rgba(8, 13, 25, 0.7);
  border-radius: 16px;
  padding: 0.7rem 0.9rem;
  border: 1px solid rgba(79, 98, 143, 0.4);
}
.metrics label {
  display: block;
  font-size: 0.75rem;
  color: var(--muted);
  margin-bottom: 0.25rem;
}
.metrics strong {
  font-size: 1.05rem;
}
.metrics strong.ok { color: #4ade80; }
.metrics strong.bad { color: #f87171; }
footer {
  margin-top: 0.8rem;
  font-size: 0.78rem;
  color: var(--muted);
}
</style>
