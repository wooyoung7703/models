<template>
  <section class="job-log-card" v-if="items.length">
    <header>
      <h3>WS · 배치 로그</h3>
      <p class="muted">최신 30건</p>
    </header>
    <table>
      <thead>
        <tr>
          <th>시각</th>
          <th>출처</th>
          <th>메시지</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in items" :key="row.id">
          <td>{{ row.ts }}</td>
          <td>{{ row.source }}</td>
          <td>{{ row.message }}</td>
        </tr>
      </tbody>
    </table>
  </section>
</template>

<script setup lang="ts">
interface JobLogEntry {
  id: number
  ts: string
  source: string
  message: string
}
defineProps<{ items: JobLogEntry[] }>()
</script>

<style scoped>
.job-log-card {
  border: 1px solid rgba(96, 165, 250, 0.35);
  border-radius: 20px;
  padding: 1.1rem;
  margin-bottom: 1.5rem;
  background: rgba(8, 13, 26, 0.9);
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
}
th,
td {
  padding: 0.45rem;
  border-bottom: 1px solid rgba(99, 121, 173, 0.3);
  text-align: left;
}
th {
  color: var(--muted);
  font-weight: 500;
}
</style>
