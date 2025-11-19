<template>
  <section class="polling-control">
    <header>
      <div>
        <h3>HTTP 폴백 주기</h3>
        <p class="muted">WS 실패 시 데이터를 다시 가져오는 간격을 조정합니다.</p>
      </div>
      <span :class="['badge', wsConnected ? 'ws' : 'poll']">
        {{ wsConnected ? 'WS 우선' : '폴백 활성' }}
      </span>
    </header>
    <div class="slider-row">
      <input
        type="range"
        :min="resolvedMin"
        :max="resolvedMax"
        :step="step"
        :value="value"
        @input="onInput"
      />
      <span class="value">{{ value }}초</span>
    </div>
    <footer>
      <p class="hint">
        {{ wsConnected ? 'WS 연결이 끊길 때 적용될 값입니다.' : '현재 HTTP 폴백이 동작 중입니다.' }}
      </p>
      <p class="hint">범위: {{ resolvedMin }}초 ~ {{ resolvedMax }}초</p>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
const props = withDefaults(
  defineProps<{ value: number; min?: number; max?: number; step?: number; wsConnected?: boolean }>(),
  {
    min: 5,
    max: 120,
    step: 5,
    wsConnected: true,
  }
)

const emit = defineEmits<{ (e: 'update:value', value: number): void }>()

const resolvedMin = computed(() => Math.max(1, props.min))
const resolvedMax = computed(() => Math.max(resolvedMin.value, props.max))
const step = computed(() => Math.max(1, props.step || 1))

function onInput(event: Event) {
  const target = event.target as HTMLInputElement
  const next = Number(target.value)
  emit('update:value', next)
}
</script>

<style scoped>
.polling-control {
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
}

header {
  display: flex;
  justify-content: space-between;
  gap: 0.75rem;
  align-items: flex-start;
}

header h3 {
  margin: 0;
  font-size: 1.1rem;
}

.muted {
  color: rgba(148, 163, 184, 0.85);
  font-size: 0.85rem;
}

.badge {
  padding: 0.25rem 0.55rem;
  border-radius: 999px;
  font-size: 0.75rem;
  text-transform: uppercase;
  border: 1px solid rgba(148, 163, 184, 0.4);
}

.badge.ws {
  background: rgba(34, 197, 94, 0.1);
  border-color: rgba(34, 197, 94, 0.5);
  color: #4ade80;
}

.badge.poll {
  background: rgba(248, 113, 113, 0.08);
  border-color: rgba(248, 113, 113, 0.5);
  color: #fda4af;
}

.slider-row {
  display: flex;
  align-items: center;
  gap: 0.8rem;
}

input[type='range'] {
  flex: 1;
  accent-color: #22d3ee;
}

.value {
  min-width: 3.6rem;
  font-weight: 600;
  text-align: right;
}

footer .hint {
  font-size: 0.78rem;
  color: rgba(148, 163, 184, 0.75);
}
</style>
