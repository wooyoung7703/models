<template>
  <section class="notification-panel" v-if="items.length">
    <header>
      <h3>실시간 알림</h3>
      <span class="muted">{{ items.length }}개</span>
    </header>
    <ul>
      <li v-for="item in items" :key="item.id" :class="item.level">
        <div>
          <strong>[{{ item.level.toUpperCase() }}]</strong>
          <span>{{ item.message }}</span>
        </div>
        <div class="meta">
          <time>{{ item.ts }}</time>
          <button class="link" @click="$emit('dismiss', item.id)">닫기</button>
        </div>
      </li>
    </ul>
  </section>
</template>

<script setup lang="ts">
interface NotificationItem {
  id: number
  ts: string
  level: 'info' | 'warn' | 'error'
  message: string
}
defineProps<{ items: NotificationItem[] }>()
defineEmits<{ (e: 'dismiss', id: number): void }>()
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
  align-items: baseline;
  margin-bottom: 0.6rem;
}
ul {
  list-style: none;
  margin: 0;
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
}
button.link {
  background: none;
  border: none;
  color: var(--accent);
  cursor: pointer;
  font-size: 0.76rem;
}
</style>
