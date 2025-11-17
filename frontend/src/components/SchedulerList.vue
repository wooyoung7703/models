<template>
  <section class="scheduler-list">
    <header class="scheduler-list__header">
      <h3>등록된 스케줄러</h3>
      <p class="muted">실행 상태와 간격을 확인하고 제어하세요.</p>
    </header>
    <table>
      <thead>
        <tr>
          <th>이름</th>
          <th>위치</th>
          <th>간격</th>
          <th>상태</th>
          <th>최근 실행</th>
          <th>비고</th>
          <th>제어</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in items" :key="row.key">
          <td>
            <div class="col-name">
              <strong>{{ row.name }}</strong>
              <span class="tags" v-if="row.tags && row.tags.length">{{ row.tags.join(', ') }}</span>
            </div>
          </td>
          <td>
            <span class="type-chip" :class="row.type === 'frontend' ? 'frontend' : 'backend'">
              {{ row.type === 'frontend' ? '프론트엔드' : '백엔드' }}
            </span>
          </td>
          <td>{{ row.interval || '미설정' }}</td>
          <td>
            <span class="status-chip" :class="row.running ? 'ok' : 'warn'">
              {{ row.running ? 'RUNNING' : (row.enabled ? 'IDLE' : 'DISABLED') }}
            </span>
          </td>
          <td>{{ row.lastRun || '—' }}</td>
          <td class="muted">{{ row.desc || '—' }}</td>
          <td class="controls">
            <button
              class="btn ghost"
              :disabled="loadingKey === `${row.key}:start` || row.running"
              @click="$emit('command', { key: row.key, action: 'start' })"
            >
              시작
            </button>
            <button
              class="btn ghost"
              :disabled="loadingKey === `${row.key}:stop` || !row.running"
              @click="$emit('command', { key: row.key, action: 'stop' })"
            >
              중지
            </button>
          </td>
        </tr>
        <tr v-if="!items.length">
          <td colspan="7" class="muted empty">등록된 스케줄러가 없습니다.</td>
        </tr>
      </tbody>
    </table>
  </section>
</template>

<script setup lang="ts">
interface SchedulerRow {
  key: string
  name: string
  desc?: string
  interval?: string
  lastRun?: string | null
  running?: boolean
  enabled?: boolean
  tags?: string[]
  type?: string
}
defineProps<{
  items: SchedulerRow[]
  loadingKey: string
}>()
defineEmits<{
  (e: 'command', payload: { key: string; action: 'start' | 'stop' }): void
}>()
</script>

<style scoped>
.scheduler-list {
  border: 1px solid rgba(129, 161, 214, 0.32);
  border-radius: 20px;
  padding: 1.4rem 1.2rem;
  background: linear-gradient(180deg, rgba(16, 23, 42, 0.95), rgba(8, 13, 26, 0.92));
  margin-bottom: 1.5rem;
}
.scheduler-list__header {
  margin-bottom: 1rem;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}
th,
td {
  padding: 0.6rem 0.5rem;
  border-bottom: 1px solid rgba(99, 121, 173, 0.25);
  text-align: left;
}
.col-name strong {
  display: block;
}
.tags {
  font-size: 0.75rem;
  color: var(--muted);
}
.controls {
  display: flex;
  gap: 0.4rem;
}
.status-chip {
  padding: 0.2rem 0.75rem;
  border-radius: 999px;
  font-size: 0.72rem;
  border: 1px solid transparent;
}
.status-chip.ok {
  background: rgba(74, 222, 128, 0.15);
  border-color: rgba(74, 222, 128, 0.35);
  color: var(--ok);
}
.status-chip.warn {
  background: rgba(251, 191, 36, 0.15);
  border-color: rgba(251, 191, 36, 0.35);
  color: var(--warn);
}
.type-chip {
  display: inline-flex;
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  font-size: 0.72rem;
  border: 1px solid transparent;
}
.type-chip.backend {
  background: rgba(59, 130, 246, 0.18);
  border-color: rgba(59, 130, 246, 0.4);
  color: var(--accent);
}
.type-chip.frontend {
  background: rgba(234, 179, 8, 0.18);
  border-color: rgba(234, 179, 8, 0.4);
  color: #facc15;
}
.empty {
  text-align: center;
  padding: 1.4rem 0;
}
</style>
