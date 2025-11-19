<template>
  <article class="meta-card" v-if="meta">
    <header>
      <div>
        <p class="label">Stacking</p>
        <h3>{{ meta.method || 'unknown' }}</h3>
      </div>
      <span class="badge">threshold {{ thresholdLabel }}</span>
    </header>
    <ul>
      <li>
        <span class="muted">Source</span>
        <strong>{{ meta.threshold_source || 'n/a' }}</strong>
      </li>
      <li>
        <span class="muted">Models</span>
        <div class="models">
          <span v-for="model in modelList" :key="model">{{ model }}</span>
          <span v-if="!modelList.length" class="muted">(none)</span>
        </div>
      </li>
    </ul>
  </article>
  <article v-else class="meta-card missing">
    <p class="label">Stacking</p>
    <p class="muted">메타 정보 없음</p>
  </article>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface StackingMeta {
  method?: string
  threshold?: number
  threshold_source?: string
  models?: string[]
}

const props = defineProps<{ meta: StackingMeta | null }>()

const thresholdLabel = computed(() => {
  const value = props.meta?.threshold
  if (typeof value !== 'number') return 'n/a'
  return value >= 1 ? value.toFixed(2) : value.toFixed(3)
})

const modelList = computed(() => props.meta?.models || [])
</script>

<style scoped>
.meta-card {
  border: 1px solid rgba(148, 163, 184, 0.4);
  border-radius: 18px;
  padding: 0.9rem 1rem;
  background: rgba(11, 16, 28, 0.9);
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
  font-size: 1.25rem;
}

.badge {
  font-size: 0.75rem;
  border: 1px solid rgba(94, 234, 212, 0.5);
  border-radius: 999px;
  padding: 0.1rem 0.6rem;
  color: #5eead4;
}

ul {
  list-style: none;
  margin: 0.8rem 0 0;
  padding: 0;
  display: grid;
  gap: 0.4rem;
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

.models {
  display: flex;
  gap: 0.35rem;
  flex-wrap: wrap;
}

.models span {
  border: 1px solid rgba(148, 163, 184, 0.4);
  border-radius: 10px;
  padding: 0.05rem 0.45rem;
  font-size: 0.78rem;
}

.meta-card.missing {
  align-items: center;
  justify-content: center;
  text-align: center;
}
</style>
