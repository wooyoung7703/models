<template>
  <section class="preset-card">
    <header>
      <h3>레이아웃 프리셋</h3>
      <p class="muted">현재 필터/정렬 구성을 저장하거나 불러오세요.</p>
    </header>
    <form @submit.prevent="handleSave">
      <input v-model="name" placeholder="프리셋 이름" />
      <button class="btn ghost" type="submit">현재 상태 저장</button>
    </form>
    <ul>
      <li v-for="preset in presets" :key="preset.id">
        <div>
          <strong>{{ preset.name }}</strong>
          <span class="muted">필터 '{{ preset.filter || '전체' }}' · {{ preset.sortKey }}-{{ preset.sortDir }}</span>
        </div>
        <div class="actions">
          <button class="btn ghost" type="button" @click="$emit('apply', preset.id)">적용</button>
          <button class="btn ghost danger" type="button" @click="$emit('delete', preset.id)">삭제</button>
        </div>
      </li>
      <li v-if="!presets.length" class="muted empty">저장된 프리셋이 없습니다.</li>
    </ul>
  </section>
</template>

<script setup lang="ts">
import { ref } from 'vue'
interface LayoutPreset {
  id: string
  name: string
  filter: string
  sortKey: string
  sortDir: string
}
defineProps<{
  presets: LayoutPreset[]
  currentFilter: string
  sortKey: string
  sortDir: string
}>()
const emit = defineEmits<{
  (e: 'save', name: string): void
  (e: 'apply', id: string): void
  (e: 'delete', id: string): void
}>()
const name = ref('')
const handleSave = () => {
  emit('save', name.value)
  name.value = ''
}
</script>

<style scoped>
.preset-card {
  border: 1px solid rgba(129, 161, 214, 0.35);
  border-radius: 20px;
  padding: 1.2rem;
  margin-bottom: 1.2rem;
  background: rgba(9, 14, 28, 0.9);
}
form {
  display: flex;
  gap: 0.6rem;
  margin: 0.8rem 0 1rem;
}
input {
  flex: 1;
  border-radius: 10px;
  border: 1px solid rgba(148, 163, 184, 0.35);
  padding: 0.4rem 0.6rem;
  background: rgba(15, 23, 42, 0.8);
  color: var(--text);
}
ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.5rem;
}
li {
  border: 1px solid rgba(148, 163, 184, 0.3);
  border-radius: 14px;
  padding: 0.6rem 0.7rem;
  display: flex;
  justify-content: space-between;
  gap: 0.8rem;
  font-size: 0.82rem;
}
.actions {
  display: flex;
  gap: 0.4rem;
}
.btn.ghost.danger {
  border-color: rgba(248, 113, 113, 0.5);
  color: #f87171;
}
.empty {
  justify-content: center;
}
</style>
