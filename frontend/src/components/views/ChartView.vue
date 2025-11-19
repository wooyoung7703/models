<template>
  <section class="chart-pane">
    <header class="chart-toolbar">
      <div>
        <h2>실시간 차트</h2>
        <p class="muted">WS 기반 캔들 + 거래 신호</p>
      </div>
      <div class="toolbar-controls">
        <label class="filter-field">
          <span>필터</span>
          <input
            v-model="filterText"
            type="text"
            placeholder="예: BTC"
          />
        </label>
        <label>
          <span>심볼</span>
          <select v-model="innerSymbol">
            <option v-for="sym in filteredSymbols" :key="sym" :value="sym">
              {{ sym.toUpperCase() }}
            </option>
            <option v-if="!filteredSymbols.length" disabled value="">
              매칭 없음
            </option>
          </select>
        </label>
        <div class="interval-toggle">
          <button
            v-for="option in intervalOptions"
            :key="option"
            type="button"
            :class="['chip', { active: interval === option }]"
            @click="interval = option"
          >
            {{ option }}
          </button>
        </div>
        <label class="autoplay-toggle">
          <input type="checkbox" v-model="autoAdvance" />
          <span>자동 순회</span>
          <span :class="['autoplay-status', autoAdvance ? 'on' : 'off']">
            {{ autoAdvance ? 'ON' : 'OFF' }}
          </span>
          <select
            v-model.number="autoAdvanceSpeed"
            :disabled="!autoAdvance"
            class="autoplay-speed"
          >
            <option
              v-for="option in autoAdvanceSpeedOptions"
              :key="option.value"
              :value="option.value"
            >
              {{ option.label }}
            </option>
          </select>
          <span
            v-if="autoAdvance && autoAdvanceCountdown !== null"
            class="autoplay-countdown"
          >
            다음 이동 {{ autoAdvanceCountdown }}초
          </span>
        </label>
        <button type="button" class="chip ghost" @click="showShortcutHelp = !showShortcutHelp">
          단축키
        </button>
      </div>
    </header>
    <div v-if="favoriteChips.length" class="favorite-chips">
      <span class="muted">즐겨찾기</span>
      <button
        v-for="sym in favoriteChips"
        :key="sym"
        type="button"
        :class="['chip', { active: sym === innerSymbol }]"
        @click="emit('update:chartSymbol', sym)"
      >
        {{ sym }}
      </button>
      <button
        type="button"
        class="chip ghost"
        @click="clearFavorites"
      >
        전체 해제
      </button>
    </div>
    <div v-if="innerSymbol && apiBase" class="chart-grid">
      <RealtimeChart
        class="chart-area"
        :symbol="innerSymbol"
        :apiBase="apiBase"
        :interval="interval"
        :signals="chartSignals"
        :liveNowcast="chartNowcast"
        :wsConnected="wsConnected"
      />
      <ChartSymbolSummary
        class="summary-area"
        :symbol="innerSymbol"
        :nowcast="chartNowcast"
        :signals="chartSignals"
        :wsConnected="wsConnected"
        :interval="interval"
        :favorite="isFavorite(innerSymbol)"
        @toggle-favorite="toggleFavorite(innerSymbol)"
      />
    </div>
    <p v-else class="muted empty-note">표시할 심볼이 없습니다.</p>

    <section v-if="showShortcutHelp" class="shortcut-panel">
      <header>
        <strong>단축키</strong>
        <button type="button" class="ghost" @click="showShortcutHelp = false">닫기</button>
      </header>
      <ul>
        <li><code>← / →</code><span>심볼 순환</span></li>
        <li><code>[ / ]</code><span>차트 간격 변경</span></li>
        <li><code>1-4</code><span>즐겨찾기 심볼 바로가기</span></li>
        <li><code>0</code><span>첫 심볼로 이동</span></li>
        <li><code>A</code><span>자동 순회 토글</span></li>
        <li><code>Shift + A</code><span>순회 속도 빠르게</span></li>
        <li><code>Alt + A</code><span>순회 속도 느리게</span></li>
        <li><code>?</code><span>이 패널 토글</span></li>
      </ul>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, onBeforeUnmount, ref, watch } from 'vue'
// @ts-ignore script-setup SFC default export
import RealtimeChart from '../RealtimeChart.vue'
// @ts-ignore script-setup SFC default export
import ChartSymbolSummary from '../ChartSymbolSummary.vue'

interface ChartSignal {
  id: string | number
  ts: string
  side: 'buy' | 'sell'
  price?: number
  label?: string
}

const props = defineProps<{
  symbols: string[]
  chartSymbol: string
  apiBase: string
  chartSignals: ChartSignal[]
  chartNowcast: Record<string, any> | null
  wsConnected: boolean
}>()

const emit = defineEmits<{
  (e: 'update:chartSymbol', value: string): void
  (e: 'interval-change', value: string): void
}>()

const innerSymbol = computed({
  get: () => props.chartSymbol,
  set: (value: string) => emit('update:chartSymbol', value)
})

const intervalOptions = ['1m', '5m', '15m'] as const

onMounted(() => {
  window.addEventListener('keydown', handleKeydown)
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', handleKeydown)
  stopAutoAdvance()
})
function handleKeydown(event: KeyboardEvent) {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return
  if (event.code === 'KeyA') {
    event.preventDefault()
    handleAToggle(event)
    return
  }
  const handler = keyMap[event.key]
  if (handler) {
    event.preventDefault()
    handler(event)
  }
}

function cycleSymbol(direction: 1 | -1) {
  const symbols = filteredSymbols.value
  if (!symbols.length) return
  const currentIndex = symbols.indexOf(innerSymbol.value)
  const nextIndex = (currentIndex + direction + symbols.length) % symbols.length
  emit('update:chartSymbol', symbols[nextIndex])
}

function cycleInterval(direction: 1 | -1) {
  const currentIndex = intervalOptions.indexOf(interval.value as any)
  const nextIndex = (currentIndex + direction + intervalOptions.length) % intervalOptions.length
  interval.value = intervalOptions[nextIndex]
}

function jumpToFavorite(index: number) {
  if (index === 0) {
    emit('update:chartSymbol', filteredSymbols.value[0] || favorites.value[0] || props.symbols[0])
    return
  }
  const target = favoriteChips.value[index - 1]
  if (target) emit('update:chartSymbol', target)
}
const intervalStorageKey = 'chart.interval'
const filterStorageKey = 'chart.symbolFilter'
const favoriteStorageKey = 'chart.favorites'
const autoAdvanceStorageKey = 'chart.autoAdvance'
const autoAdvanceSpeedStorageKey = 'chart.autoAdvanceSpeed'
const autoAdvanceSpeedOptions = [
  { label: '5초', value: 5000 },
  { label: '12초', value: 12000 },
  { label: '30초', value: 30000 },
] as const
const interval = ref(readStoredInterval())
const filterText = ref(readStoredFilter())
const favorites = ref<string[]>(readStoredFavorites())
const autoAdvance = ref(readStoredAutoAdvance())
const autoAdvanceSpeed = ref(readStoredAutoAdvanceSpeed())
const autoAdvanceCountdown = ref<number | null>(null)
const showShortcutHelp = ref(false)
let autoAdvanceTimer: ReturnType<typeof setInterval> | null = null
let autoAdvanceCountdownTimer: ReturnType<typeof setInterval> | null = null

const keyMap: Record<string, (event?: KeyboardEvent) => void> = {
  ArrowLeft: () => cycleSymbol(-1),
  ArrowRight: () => cycleSymbol(1),
  '[': () => cycleInterval(-1),
  ']': () => cycleInterval(1),
  '?': () => (showShortcutHelp.value = !showShortcutHelp.value),
  '0': () => jumpToFavorite(0),
  '1': () => jumpToFavorite(0),
  '2': () => jumpToFavorite(1),
  '3': () => jumpToFavorite(2),
  '4': () => jumpToFavorite(3),
}

const filteredSymbols = computed(() => {
  if (!filterText.value.trim()) return props.symbols
  const needle = filterText.value.trim().toUpperCase()
  return props.symbols.filter((sym) => sym.toUpperCase().includes(needle))
})

const favoriteChips = computed(() => favorites.value.filter((sym) => props.symbols.includes(sym)))

watch(
  interval,
  (value) => {
    if (typeof window !== 'undefined') {
      window.localStorage?.setItem(intervalStorageKey, value)
    }
    emit('interval-change', value)
  },
  { immediate: true }
)

watch(filterText, (value) => {
  if (typeof window === 'undefined') return
  window.localStorage?.setItem(filterStorageKey, value)
})

watch(
  favorites,
  (value) => {
    if (typeof window === 'undefined') return
    window.localStorage?.setItem(favoriteStorageKey, JSON.stringify(value))
  },
  { deep: true }
)

watch(
  autoAdvance,
  (value) => {
    if (typeof window !== 'undefined') {
      window.localStorage?.setItem(autoAdvanceStorageKey, value ? '1' : '0')
    }
    if (value) {
      scheduleAutoAdvance()
    } else {
      stopAutoAdvance()
    }
  },
  { immediate: true }
)

watch(filteredSymbols, () => {
  if (autoAdvance.value) {
    scheduleAutoAdvance()
  }
})

watch(
  autoAdvanceSpeed,
  (value) => {
    if (typeof window !== 'undefined') {
      window.localStorage?.setItem(autoAdvanceSpeedStorageKey, String(value))
    }
    if (autoAdvance.value) {
      scheduleAutoAdvance()
    }
  },
  { immediate: true }
)

watch(
  filteredSymbols,
  (symbols) => {
    if (!symbols.length) return
    if (!symbols.includes(innerSymbol.value)) {
      emit('update:chartSymbol', symbols[0])
    }
  },
  { immediate: true }
)

watch(
  () => props.symbols,
  (symbols) => {
    favorites.value = favorites.value.filter((sym) => symbols.includes(sym))
  },
  { immediate: true }
)

function isFavorite(symbol?: string) {
  if (!symbol) return false
  return favorites.value.includes(symbol)
}

function toggleFavorite(symbol?: string) {
  if (!symbol) return
  const next = new Set(favorites.value)
  if (next.has(symbol)) {
    next.delete(symbol)
  } else {
    next.add(symbol)
  }
  favorites.value = Array.from(next)
}

function clearFavorites() {
  favorites.value = []
}

function toggleAutoAdvance() {
  autoAdvance.value = !autoAdvance.value
}

function handleAToggle(event?: KeyboardEvent) {
  if (!event) {
    toggleAutoAdvance()
    return
  }
  if (event.shiftKey) {
    adjustAutoAdvanceSpeed(1)
    return
  }
  if (event.altKey) {
    adjustAutoAdvanceSpeed(-1)
    return
  }
  toggleAutoAdvance()
}

function adjustAutoAdvanceSpeed(direction: 1 | -1) {
  const currentIndex = autoAdvanceSpeedOptions.findIndex((option) => option.value === autoAdvanceSpeed.value)
  const safeIndex = currentIndex === -1 ? 1 : currentIndex
  const nextIndex = Math.min(
    autoAdvanceSpeedOptions.length - 1,
    Math.max(0, safeIndex + direction)
  )
  autoAdvanceSpeed.value = autoAdvanceSpeedOptions[nextIndex].value
}

function scheduleAutoAdvance() {
  stopAutoAdvanceTimers()
  if (!autoAdvance.value) {
    autoAdvanceCountdown.value = null
    return
  }
  if (filteredSymbols.value.length <= 1) {
    autoAdvanceCountdown.value = null
    return
  }
  if (typeof window === 'undefined') return
  resetAutoAdvanceCountdown()
  startAutoAdvanceCountdown()
  autoAdvanceTimer = window.setInterval(() => {
    if (!autoAdvance.value) return
    if (filteredSymbols.value.length <= 1) {
      autoAdvanceCountdown.value = null
      return
    }
    cycleSymbol(1)
    resetAutoAdvanceCountdown()
  }, autoAdvanceSpeed.value)
}

function stopAutoAdvance() {
  stopAutoAdvanceTimers()
  autoAdvanceCountdown.value = null
}

function stopAutoAdvanceTimers() {
  if (autoAdvanceTimer) {
    clearInterval(autoAdvanceTimer)
    autoAdvanceTimer = null
  }
  if (autoAdvanceCountdownTimer) {
    clearInterval(autoAdvanceCountdownTimer)
    autoAdvanceCountdownTimer = null
  }
}

function startAutoAdvanceCountdown() {
  if (typeof window === 'undefined') return
  if (autoAdvanceCountdownTimer) {
    clearInterval(autoAdvanceCountdownTimer)
  }
  autoAdvanceCountdownTimer = window.setInterval(() => {
    if (!autoAdvance.value) return
    if (autoAdvanceCountdown.value === null) return
    autoAdvanceCountdown.value = Math.max(0, autoAdvanceCountdown.value - 1)
    if (autoAdvanceCountdown.value === 0) {
      resetAutoAdvanceCountdown()
    }
  }, 1000)
}

function resetAutoAdvanceCountdown() {
  autoAdvanceCountdown.value = Math.max(1, Math.round(autoAdvanceSpeed.value / 1000))
}

function readStoredInterval(): (typeof intervalOptions)[number] {
  if (typeof window === 'undefined') return '1m'
  const stored = window.localStorage?.getItem(intervalStorageKey) || '1m'
  return intervalOptions.includes(stored as any) ? (stored as any) : '1m'
}

function readStoredFilter() {
  if (typeof window === 'undefined') return ''
  return window.localStorage?.getItem(filterStorageKey) || ''
}

function readStoredFavorites() {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage?.getItem(favoriteStorageKey)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed.filter((item) => typeof item === 'string') : []
  } catch {
    return []
  }
}

function readStoredAutoAdvance() {
  if (typeof window === 'undefined') return false
  return window.localStorage?.getItem(autoAdvanceStorageKey) === '1'
}

function readStoredAutoAdvanceSpeed() {
  if (typeof window === 'undefined') return autoAdvanceSpeedOptions[1].value
  const stored = Number(window.localStorage?.getItem(autoAdvanceSpeedStorageKey) || autoAdvanceSpeedOptions[1].value)
  const matched = autoAdvanceSpeedOptions.find((option) => option.value === stored)
  return matched ? matched.value : autoAdvanceSpeedOptions[1].value
}
</script>

<style scoped>
.chart-pane {
  border: 1px solid rgba(81, 110, 163, 0.45);
  border-radius: 24px;
  background: var(--card);
  padding: 1.25rem 1.35rem 1.6rem;
  min-height: 520px;
}

.chart-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 1rem;
  margin-bottom: 1rem;
}

.toolbar-controls {
  display: flex;
  align-items: flex-end;
  gap: 0.75rem;
}

.autoplay-toggle {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.82rem;
  opacity: 0.85;
}

.autoplay-toggle input {
  accent-color: #38bdf8;
}

.autoplay-status {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  border-radius: 999px;
  padding: 0.05rem 0.6rem;
  border: 1px solid rgba(148, 163, 184, 0.3);
  font-size: 0.7rem;
  transition: color 0.2s ease, border-color 0.2s ease, background 0.2s ease;
}

.autoplay-status::before {
  content: '';
  width: 0.35rem;
  height: 0.35rem;
  border-radius: 50%;
  background: currentColor;
}

.autoplay-status.on {
  color: #22d3ee;
  border-color: rgba(34, 211, 238, 0.5);
  background: rgba(16, 185, 129, 0.1);
}

.autoplay-status.off {
  color: rgba(148, 163, 184, 0.8);
  background: rgba(148, 163, 184, 0.08);
}

.autoplay-speed {
  padding: 0.25rem 0.35rem;
  border-radius: 10px;
  border: 1px solid rgba(91, 118, 170, 0.45);
  background: rgba(15, 23, 42, 0.85);
  color: var(--text);
  font-size: 0.76rem;
  min-width: 4.7rem;
  transition: opacity 0.2s ease;
}

.autoplay-speed:disabled {
  opacity: 0.4;
}

.autoplay-countdown {
  font-size: 0.74rem;
  color: rgba(148, 163, 184, 0.85);
  padding: 0.1rem 0.4rem;
  border-radius: 999px;
  border: 1px dashed rgba(148, 163, 184, 0.3);
  background: rgba(15, 23, 42, 0.7);
}

.toolbar-controls .filter-field input {
  padding: 0.35rem 0.55rem;
  border-radius: 10px;
  border: 1px solid rgba(91, 118, 170, 0.45);
  background: rgba(15, 23, 42, 0.85);
  color: var(--text);
}

.chart-toolbar select {
  padding: 0.4rem 0.7rem;
  border-radius: 12px;
  border: 1px solid rgba(91, 118, 170, 0.45);
  background: rgba(15, 23, 42, 0.85);
  color: var(--text);
}

.interval-toggle {
  display: flex;
  gap: 0.25rem;
}

.chip {
  border: 1px solid rgba(148, 163, 184, 0.35);
  border-radius: 10px;
  padding: 0.25rem 0.6rem;
  background: transparent;
  color: inherit;
  font-size: 0.82rem;
  cursor: pointer;
}

.chip.active {
  border-color: rgba(96, 165, 250, 0.6);
  color: #93c5fd;
}

.chip.ghost {
  border-style: dashed;
  opacity: 0.7;
}

.shortcut-panel {
  margin-top: 1rem;
  border: 1px solid rgba(148, 163, 184, 0.3);
  border-radius: 16px;
  padding: 0.8rem 1rem;
  background: rgba(7, 11, 23, 0.9);
}

.shortcut-panel header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.6rem;
}

.shortcut-panel ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.4rem;
}

.shortcut-panel li {
  display: flex;
  justify-content: space-between;
  font-size: 0.85rem;
}

.shortcut-panel code {
  background: rgba(15, 23, 42, 0.8);
  border: 1px solid rgba(148, 163, 184, 0.3);
  border-radius: 6px;
  padding: 0.1rem 0.35rem;
  font-size: 0.8rem;
}

.favorite-chips {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-bottom: 1rem;
}

.favorite-chips .muted {
  margin-right: 0.3rem;
}

.chart-grid {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 1.2rem;
  align-items: start;
}

.chart-area {
  min-height: 460px;
}

.summary-area {
  min-height: 460px;
}

@media (max-width: 1100px) {
  .chart-grid {
    grid-template-columns: 1fr;
  }
  .summary-area {
    min-height: auto;
  }
}
</style>
