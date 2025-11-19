<template>
  <section class="notification-panel">
    <header>
      <div>
        <h3>실시간 알림</h3>
        <span class="muted">{{ filteredItems.length }} / {{ items.length }}</span>
      </div>
      <span :class="['status-pill', wsConnected ? 'ws' : 'fallback']">
        {{ wsConnected ? 'WS live' : 'HTTP fallback' }}
      </span>
    </header>

    <NotificationSourceChips
      :sources="sourceOptions"
      :model-value="sourceState"
      @update:modelValue="handleSourceChange"
    />

    <ul v-if="filteredItems.length">
      <li v-for="item in filteredItems" :key="item.id" :class="item.level">
        <div>
          <strong>[{{ item.level.toUpperCase() }}]</strong>
          <span>{{ item.message }}</span>
        </div>
        <div class="meta">
          <span class="source-tag">{{ (item.source || 'system').toUpperCase() }}</span>
          <time>{{ item.ts }}</time>
          <button class="link" @click="$emit('dismiss', item.id)">닫기</button>
        </div>
      </li>
    </ul>
    <p v-else class="muted empty">선택한 범주의 알림이 없습니다.</p>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
// @ts-ignore script-setup default export shim
import NotificationSourceChips from './NotificationSourceChips.vue'

interface NotificationItem {
  id: number
  ts: string
  level: 'info' | 'warn' | 'error'
  message: string
  source?: string
}

const sourceOptions = [
  { key: 'all', label: '전체' },
  { key: 'ws', label: 'WS' },
  { key: 'admin', label: 'Admin' },
  { key: 'system', label: 'System' },
]

const props = withDefaults(
  defineProps<{ items: NotificationItem[]; source?: string; wsConnected?: boolean }>(),
  { source: 'all', wsConnected: true }
)

const emit = defineEmits<{
  (e: 'dismiss', id: number): void
  (e: 'update:source', value: string): void
}>()

const sourceState = computed(() => props.source || 'all')

const filteredItems = computed(() => {
  const active = sourceState.value
  if (active === 'all') return props.items
  return props.items.filter((item) => (item.source || 'system') === active)
})

function handleSourceChange(value: string) {
  emit('update:source', value)
}
</script>

<style scoped>
.notification-panel {
  border: 1px solid rgba(96, 165, 250, 0.35);
  border-radius: 20px;
  padding: 1rem 1.1rem;
  margin-bottom: 1.2rem;
  background: rgba(11, 18, 34, 0.92);
}

header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.8rem;
}

.status-pill {
  font-size: 0.72rem;
  border-radius: 999px;
  padding: 0.15rem 0.6rem;
  border: 1px solid;
}

.status-pill.ws {
  border-color: rgba(34, 197, 94, 0.5);
  color: #4ade80;
}

.status-pill.fallback {
  border-color: rgba(248, 113, 113, 0.6);
  color: #fda4af;
}

ul {
  list-style: none;
  margin: 0.8rem 0 0;
  padding: 0;
  display: grid;
  gap: 0.4rem;
}

li {
  border: 1px solid rgba(148, 163, 184, 0.3);
  border-radius: 14px;
  padding: 0.55rem 0.75rem;
  display: flex;
  justify-content: space-between;
  gap: 0.8rem;
  font-size: 0.82rem;
}

li.info { border-color: rgba(96, 165, 250, 0.4); }
li.warn { border-color: rgba(250, 204, 21, 0.45); }
li.error { border-color: rgba(248, 113, 113, 0.45); }

.meta {
  text-align: right;
  font-size: 0.76rem;
  color: var(--muted);
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  align-items: flex-end;
}

.source-tag {
  font-size: 0.7rem;
  border: 1px solid rgba(148, 163, 184, 0.4);
  border-radius: 999px;
  padding: 0.05rem 0.35rem;
}

button.link {
  background: none;
  border: none;
  color: var(--accent);
  cursor: pointer;
  font-size: 0.76rem;
}

.empty {
  text-align: center;
  margin-top: 0.6rem;
}
</style>
