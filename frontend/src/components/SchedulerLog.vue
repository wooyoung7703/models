<template>
  <section class="scheduler-log-card" v-if="items.length">
    <header>
      <h3>스케줄러 로그</h3>
      <span class="muted">최근 {{ items.length }}건</span>
    </header>
    <ul>
      <li v-for="item in items" :key="item.id">
        <span class="muted">{{ item.ts }}</span>
        <strong>{{ item.message }}</strong>
      </li>
    </ul>
  </section>
</template>

<script setup lang="ts">
interface SchedulerLogItem {
  id: number
  ts: string
  message: string
}
defineProps<{ items: SchedulerLogItem[] }>()
</script>

<style scoped>
.scheduler-log-card {
  border: 1px solid rgba(96, 165, 250, 0.35);
  border-radius: 20px;
  padding: 1rem 1.1rem;
  background: rgba(7, 12, 24, 0.9);
  margin-top: 1.2rem;
}
ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.45rem;
  font-size: 0.82rem;
}
li {
  display: flex;
  justify-content: space-between;
  gap: 0.8rem;
  border-bottom: 1px solid rgba(148, 163, 184, 0.2);
  padding-bottom: 0.3rem;
}
li:last-child {
  border-bottom: none;
  padding-bottom: 0;
}
strong {
  font-weight: 600;
}
</style>
