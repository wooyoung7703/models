<template>
  <div class="source-chips">
    <button
      v-for="source in sources"
      :key="source.key"
      type="button"
      :class="['chip', { active: source.key === modelValue }]"
      @click="$emit('update:modelValue', source.key)"
    >
      {{ source.label }}
    </button>
  </div>
</template>

<script setup lang="ts">
interface SourceOption {
  key: string
  label: string
}

defineProps<{ sources: SourceOption[]; modelValue: string }>()

defineEmits<{ (e: 'update:modelValue', value: string): void }>()
</script>

<style scoped>
.source-chips {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.chip {
  border: 1px solid rgba(148, 163, 184, 0.35);
  border-radius: 999px;
  background: transparent;
  color: inherit;
  font-size: 0.78rem;
  padding: 0.2rem 0.7rem;
  cursor: pointer;
}

.chip.active {
  border-color: rgba(59, 130, 246, 0.6);
  color: #93c5fd;
}
</style>
