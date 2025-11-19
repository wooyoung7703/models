<template>
  <section class="top-signals">
    <header>
      <div>
        <h3>상위 스택킹 시그널</h3>
        <p class="muted">실시간 정렬 · 최대 {{ entries.length }}건</p>
      </div>
      <button v-if="entries.length" class="ghost" @click="$emit('inspect', entries[0].symbol)">
        최상위 심볼 차트
      </button>
    </header>
    <ul>
      <li
        v-for="entry in entries"
        :key="entry.symbol"
        :class="{ active: entry.symbol === selectedSymbol }"
        @click="$emit('inspect', entry.symbol)"
      >
        <div class="symbol">
          <strong>{{ entry.symbol }}</strong>
          <span class="muted">{{ entry.posture }}</span>
        </div>
        <div class="score">{{ formatScore(entry.score) }}</div>
        <span class="pill">{{ entry.posture }}</span>
      </li>
      <li v-if="!entries.length" class="empty muted">데이터 수신 대기중…</li>
    </ul>
  </section>
</template>

<script setup lang="ts">
interface TopSignalEntry {
  symbol: string
  score: number
  posture: string
}

defineProps<{
  entries: TopSignalEntry[]
  selectedSymbol?: string
}>()

defineEmits<{ (e: 'inspect', symbol: string): void }>()

function formatScore(value: number) {
  if (Number.isNaN(value)) return '0.000'
  return value.toFixed(3)
}
</script>

<style scoped>
.top-signals {
  border: 1px solid rgba(79, 98, 143, 0.35);
  border-radius: 16px;
  padding: 1rem 1.1rem;
  background: rgba(10, 15, 26, 0.92);
}
header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.8rem;
}
ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}
li {
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: 0.5rem;
  padding: 0.65rem 0.75rem;
  border: 1px solid rgba(148, 163, 184, 0.25);
  border-radius: 12px;
  cursor: pointer;
  transition: border-color 0.2s ease;
}
li.active {
  border-color: rgba(59, 130, 246, 0.7);
}
.symbol {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
}
.score {
  font-variant-numeric: tabular-nums;
  font-weight: 600;
}
.pill {
  align-self: center;
  padding: 0.15rem 0.6rem;
  border-radius: 999px;
  font-size: 0.75rem;
  border: 1px solid rgba(148, 163, 184, 0.35);
  text-transform: uppercase;
}
.empty {
  text-align: center;
  padding: 1rem 0;
}
</style>
