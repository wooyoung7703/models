<template>
  <article :class="['symbol-card', { selected }]" @click="$emit('select', symbol)">
    <header>
      <div>
        <h4>{{ symbolLabel }}</h4>
        <p class="muted">{{ interval }} · {{ updatedLabel }}</p>
      </div>
      <span class="decision" :class="decisionClass">{{ decisionLabel }}</span>
    </header>

    <dl>
      <div>
        <dt>Price</dt>
        <dd>{{ priceLabel }}</dd>
      </div>
      <div>
        <dt>Score</dt>
        <dd>{{ scoreLabel }}</dd>
      </div>
      <div>
        <dt>Margin</dt>
        <dd>{{ marginLabel }}</dd>
      </div>
      <div>
        <dt>Fresh</dt>
        <dd :class="freshClass">{{ freshLabel }}</dd>
      </div>
    </dl>

    <footer>
      <div class="component-grid" v-if="componentEntries.length">
        <span v-for="item in componentEntries" :key="item.key">
          {{ item.key }}: <strong>{{ item.value }}</strong>
        </span>
      </div>
      <p v-else class="muted">구성 요소 없음</p>
    </footer>
  </article>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  symbol: string
  interval: string
  updatedAt: string | null
  price: number | null
  score: number | null
  margin: number | null
  decision: boolean | null
  components: Record<string, number> | null
  featureFreshSeconds: number | null
  selected?: boolean
}>()

defineEmits<{ (e: 'select', symbol: string): void }>()

const symbolLabel = computed(() => props.symbol?.toUpperCase())
const priceLabel = computed(() => (typeof props.price === 'number' ? props.price.toFixed(2) : 'n/a'))
const scoreLabel = computed(() => (typeof props.score === 'number' ? (props.score * 100).toFixed(1) + '%' : 'n/a'))
const marginLabel = computed(() => (typeof props.margin === 'number' ? props.margin.toFixed(2) : 'n/a'))

const decisionClass = computed(() => {
  if (props.decision === true) return 'long'
  if (props.decision === false) return 'flat'
  return 'pending'
})

const decisionLabel = computed(() => {
  if (props.decision === true) return 'LONG'
  if (props.decision === false) return 'FLAT'
  return 'WAIT'
})

const updatedLabel = computed(() => {
  if (!props.updatedAt) return '미수신'
  const parsed = Date.parse(props.updatedAt)
  if (Number.isNaN(parsed)) return props.updatedAt
  const delta = Date.now() - parsed
  if (delta < 30_000) return '방금'
  if (delta < 120_000) return `${Math.round(delta / 1000)}초 전`
  const minutes = Math.round(delta / 60000)
  if (minutes < 60) return `${minutes}분 전`
  const hours = (minutes / 60).toFixed(1)
  return `${hours}시간 전`
})

const freshLabel = computed(() => {
  if (props.featureFreshSeconds == null) return 'n/a'
  const minutes = props.featureFreshSeconds / 60
  if (minutes < 1) return `${props.featureFreshSeconds}s`
  return `${minutes.toFixed(1)}m`
})

const freshClass = computed(() => {
  if (props.featureFreshSeconds == null) return ''
  if (props.featureFreshSeconds > 900) return 'bad'
  if (props.featureFreshSeconds > 300) return 'warn'
  return 'ok'
})

const componentEntries = computed(() => {
  if (!props.components) return []
  return Object.entries(props.components)
    .slice(0, 4)
    .map(([key, value]) => ({ key, value: value.toFixed(2) }))
})
</script>

<style scoped>
.symbol-card {
  border: 1px solid rgba(71, 85, 105, 0.4);
  border-radius: 16px;
  padding: 0.8rem 1rem;
  background: rgba(10, 16, 26, 0.95);
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
  cursor: pointer;
  transition: border-color 0.2s ease;
}

.symbol-card.selected {
  border-color: rgba(96, 165, 250, 0.8);
}

header {
  display: flex;
  justify-content: space-between;
  gap: 0.6rem;
  align-items: flex-start;
}

h4 {
  margin: 0;
  font-size: 1.15rem;
}

.muted {
  color: rgba(148, 163, 184, 0.75);
  font-size: 0.78rem;
}

.decision {
  font-size: 0.75rem;
  border: 1px solid rgba(148, 163, 184, 0.4);
  border-radius: 999px;
  padding: 0.1rem 0.5rem;
  text-transform: uppercase;
}

.decision.long {
  border-color: rgba(34, 197, 94, 0.5);
  color: #4ade80;
}

.decision.flat {
  border-color: rgba(248, 113, 113, 0.5);
  color: #fca5a5;
}

.decision.pending {
  border-style: dashed;
  color: rgba(148, 163, 184, 0.9);
}

dl {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.4rem 0.8rem;
  margin: 0;
}

dl div {
  display: flex;
  justify-content: space-between;
  gap: 0.4rem;
  font-size: 0.86rem;
}

dt {
  color: rgba(148, 163, 184, 0.75);
}

dd {
  margin: 0;
  font-variant-numeric: tabular-nums;
}

dd.ok {
  color: #4ade80;
}

dd.warn {
  color: #facc15;
}

dd.bad {
  color: #fb7185;
}

.component-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 0.3rem 0.6rem;
  font-size: 0.78rem;
}
</style>
