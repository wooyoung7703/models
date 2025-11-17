<template>
  <article class="metric-card">
    <header>스케줄러 제어</header>
    <div class="metric-card__body controls">
      <div class="btn-row">
        <button class="btn" @click="startScheduler" :disabled="busy">시작</button>
        <button class="btn warn" @click="stopScheduler" :disabled="busy">중지</button>
      </div>
      <div class="btn-row">
        <button class="btn" @click="triggerMonthly" :disabled="busy">월간 학습 즉시 실행</button>
        <button class="btn" @click="triggerDrift" :disabled="busy">드리프트 재학습</button>
        <button class="btn ghost" @click="reloadModels" :disabled="busy">모델 리로드</button>
        <button class="btn ghost" @click="fetchStatus" :disabled="busy">상태 갱신</button>
      </div>
      <p class="muted" v-if="msg">{{ msg }}</p>
    </div>
    <div class="metric-card__body" v-if="nextRuns && Object.keys(nextRuns).length">
      <p class="history-title">다음 실행 예정</p>
      <ul class="list">
        <li v-for="(v, k) in nextRuns" :key="k">
          <span>{{ k }}</span>
          <span class="muted">{{ v.next || 'n/a' }}</span>
          <strong>{{ formatEta(v.eta_seconds) }}</strong>
        </li>
      </ul>
    </div>
  </article>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'

const props = defineProps<{ apiBase: string; trainingStatus: any | null; wsConnected?: boolean; useWs?: boolean; sendWs?: (action: string) => void; adminAck?: string }>()
const busy = ref(false)
const msg = ref('')

const nextRuns = computed(() => props.trainingStatus?.next_runs || {})

const post = async (path: string) => {
  msg.value = ''
  busy.value = true
  try {
    const res = await fetch(`${props.apiBase}${path}`, { method: 'POST' })
    const ok = res.ok
    let body: any = null
    try { body = await res.json() } catch {}
    if (!ok) throw new Error(`HTTP ${res.status}`)
    msg.value = body?.status ? `완료: ${body.status}` : '완료'
  } catch (e: any) {
    msg.value = `실패: ${e?.message || '에러'}`
  } finally {
    busy.value = false
  }
}

const preferWs = () => Boolean(props.useWs && props.wsConnected && typeof props.sendWs === 'function')
const startScheduler = () => preferWs() ? (props.sendWs as any)('scheduler_start') : post('/admin/scheduler/start')
const stopScheduler = () => preferWs() ? (props.sendWs as any)('scheduler_stop') : post('/admin/scheduler/stop')
const triggerMonthly = () => preferWs() ? (props.sendWs as any)('trigger_monthly') : post('/admin/trigger_monthly_training')
const triggerDrift = () => preferWs() ? (props.sendWs as any)('trigger_prob_drift') : post('/admin/trigger_prob_drift_retrain')
const reloadModels = () => preferWs() ? (props.sendWs as any)('reload_models') : post('/admin/reload_models')
const fetchStatus = async () => {
  if (preferWs()) {
    (props.sendWs as any)('get_status')
    return
  }
  msg.value = ''
  busy.value = true
  try {
    const res = await fetch(`${props.apiBase}/admin/scheduler/status`, { method: 'GET' })
    const ok = res.ok
    const body = await res.json().catch(() => null)
    if (!ok) throw new Error(`HTTP ${res.status}`)
    msg.value = body?.status === 'ok' ? '상태 갱신 완료' : '상태 수신'
  } catch (e: any) {
    msg.value = `상태 실패: ${e?.message || '에러'}`
  } finally {
    busy.value = false
  }
}

const formatEta = (sec: any) => {
  const v = Number(sec)
  if (!isFinite(v) || v < 0) return '–'
  if (v < 120) return `${v}s`
  const m = Math.round(v / 60)
  if (m < 120) return `${m}m`
  const h = Math.round(m / 60)
  if (h < 48) return `${h}h`
  const d = Math.round(h / 24)
  return `${d}d`
}
watch(() => props.adminAck, (val) => {
  if (typeof val === 'string' && val) msg.value = val
})

</script>

<style scoped>
.controls {
  display: grid;
  gap: 0.6rem;
}
.btn-row {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.btn {
  padding: 0.45rem 0.8rem;
  border-radius: 10px;
  border: 1px solid rgba(129, 161, 214, 0.4);
  background: rgba(255, 255, 255, 0.8);
  color: #10254a;
  font-size: 0.85rem;
}
.btn.warn {
  background: rgba(248, 113, 113, 0.16);
  border-color: rgba(248, 113, 113, 0.32);
  color: #d13a3a;
}
.btn.ghost {
  background: rgba(255,255,255,0.6);
}
.history-title {
  margin: 0.2rem 0 0.4rem;
}
.list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.4rem;
}
.list li {
  display: grid;
  grid-template-columns: 1fr minmax(0, 1fr) auto;
  gap: 0.6rem;
  align-items: center;
}
</style>