<template>
  <section class="training-view">
    <header class="head">
      <div>
        <h2>훈련 & 스케줄러 현황</h2>
        <p class="muted">메타 재학습, 트레이너 하트비트, 예약 작업을 한눈에 봅니다.</p>
      </div>
      <div class="head-actions">
        <span class="badge" :class="heartbeatClass">{{ heartbeatLabel }}</span>
        <button class="ghost" :disabled="refreshing" @click="handleRefresh">
          {{ refreshing ? '갱신중…' : '상태 새로고침' }}
        </button>
      </div>
    </header>

    <div class="summary-grid">
      <article class="summary-card">
        <p class="label">Meta Retrain</p>
        <p class="value">{{ metaStateLabel }}</p>
        <p class="hint">{{ metaStateHint }}</p>
      </article>
      <article class="summary-card">
        <p class="label">Neg Streak</p>
        <p class="value" :class="{ warn: negWarn }">{{ negStreak }}</p>
        <p class="hint">{{ negHint }}</p>
      </article>
      <article class="summary-card">
        <p class="label">Next Run</p>
        <p class="value">{{ nextRunLabel }}</p>
        <p class="hint">{{ nextRunEta }}</p>
      </article>
      <article class="summary-card">
        <p class="label">Entry Win Rate</p>
        <p class="value">{{ entryWinRate }}</p>
        <p class="hint">{{ entrySamples }}</p>
      </article>
    </div>

    <div class="meta-grid">
      <MetaHistorySpark
        :history="metaHistory"
        :warnNeg="trainingStatus?.meta_neg_streak_warn"
        :driftWarn="trainingStatus?.meta_coef_drift_warn"
      />
      <MetaProbMetrics :stacking="stackingMeta" />
    </div>

    <ModelTrainingStatus :items="modelStatusItems" />

    <div class="scheduler-grid">
      <SchedulerPanel
        :apiBase="apiBase"
        :trainingStatus="trainingStatus"
        :wsConnected="wsConnected"
        :useWs="true"
        :sendWs="sendWsProxy"
        :adminAck="adminAck"
      />
      <SchedulerList
        :items="schedulerItems"
        :loadingKey="schedulerLoading"
        @command="handleSchedulerCommand"
      />
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
// @ts-ignore script-setup default export shim
import MetaHistorySpark from '../MetaHistorySpark.vue'
// @ts-ignore script-setup default export shim
import MetaProbMetrics from '../MetaProbMetrics.vue'
// @ts-ignore script-setup default export shim
import ModelTrainingStatus from '../ModelTrainingStatus.vue'
// @ts-ignore script-setup default export shim
import SchedulerPanel from '../SchedulerPanel.vue'
// @ts-ignore script-setup default export shim
import SchedulerList from '../SchedulerList.vue'
import type {
  EntryMetrics,
  TrainerMeta,
  TrainingScheduleMeta,
  TrainingStatus,
} from '../../types/realtime'

const props = defineProps<{
  apiBase: string
  trainingStatus: TrainingStatus | null
  trainerMeta: TrainerMeta | null
  entryMetrics: EntryMetrics | null
  wsConnected: boolean
  adminAck: string
  manualRefresh?: () => Promise<void>
  sendAdminCommand?: (action: string) => Promise<void>
}>()

const refreshing = ref(false)
const schedulerLoading = ref('')

async function handleRefresh() {
  if (!props.manualRefresh) return
  refreshing.value = true
  try {
    await props.manualRefresh()
  } finally {
    refreshing.value = false
  }
}

const heartbeatLabel = computed(() => {
  if (!props.trainerMeta) return 'Heartbeat 수신 안됨'
  const age = props.trainerMeta.heartbeat_age_seconds
  const ageLabel = typeof age === 'number' ? `${Math.round(age / 60)}m ago` : 'n/a'
  if (props.trainerMeta.heartbeat_healthy === false) return `Stale · ${ageLabel}`
  return `Healthy · ${ageLabel}`
})
const heartbeatClass = computed(() =>
  props.trainerMeta?.heartbeat_healthy === false ? 'warn' : 'ok'
)

const metaStateLabel = computed(() => {
  if (!props.trainingStatus) return 'idle'
  if (props.trainingStatus.pending_meta_retrain) return '대기'
  if (props.trainingStatus.meta_neg_streak_warn) return '점검 필요'
  return '안정'
})
const metaStateHint = computed(() => {
  if (!props.trainingStatus) return '스냅샷 미수신'
  return (
    props.trainingStatus.pending_meta_retrain_reason ||
    props.trainingStatus.last_meta_retrain_reason ||
    '최근 재학습 정상'
  )
})
const negWarn = computed(() => Boolean(props.trainingStatus?.meta_neg_streak_warn))
const negStreak = computed(() => props.trainingStatus?.meta_neg_streak ?? 0)
const negHint = computed(() =>
  negWarn.value ? '부정 개선 연속 · 조정 필요' : '연속 경고 없음'
)

const nextRunEntry = computed(() => {
  const runs = props.trainingStatus?.next_runs
  if (!runs) return null
  const [first] = Object.entries(runs)
  if (!first) return null
  return { key: first[0], meta: first[1] as TrainingScheduleMeta }
})
const nextRunLabel = computed(() => nextRunEntry.value?.meta?.label || nextRunEntry.value?.key || '없음')
const nextRunEta = computed(() => {
  const eta = nextRunEntry.value?.meta?.eta_seconds
  if (typeof eta !== 'number') return 'ETA 정보 없음'
  if (eta < 120) return `${Math.round(eta)}초 후`
  if (eta < 3600) return `${Math.round(eta / 60)}분 후`
  return `${(eta / 3600).toFixed(1)}시간 후`
})

const entryWinRate = computed(() => {
  const wr = props.entryMetrics?.overall?.win_rate
  if (typeof wr !== 'number') return 'n/a'
  return `${(wr * 100).toFixed(1)}%`
})
const entrySamples = computed(() => {
  const samples = props.entryMetrics?.overall?.samples
  if (!samples) return '샘플 없음'
  return `${samples} samples`
})

const metaHistory = computed(() => props.trainingStatus?.meta_history || [])
const stackingMeta = computed(() => {
  const status = props.trainingStatus as Record<string, any> | null
  if (!status) return null
  if (status.stacking_meta) return status.stacking_meta
  if (Array.isArray(status.model_training)) {
    return status.model_training.find((row: any) => row?.type === 'stacking') || null
  }
  return null
})

const modelStatusItems = computed(() => {
  const rows =
    (props.trainingStatus?.model_training_rows as Array<Record<string, any>> | undefined) ||
    (props.trainingStatus?.model_training as Array<Record<string, any>> | undefined) ||
    []
  return rows.map((row, index) => ({
    key: String(row.key || row.name || row.model || index),
    name: String(row.name || row.model || row.key || `Model ${index + 1}`),
    status: row.state || row.status || (row.running ? 'running' : 'idle'),
    lastRun: row.last_run || row.lastRun || row.last_trained_at || null,
    loss:
      typeof row.loss === 'number'
        ? row.loss
        : typeof row.metrics?.loss === 'number'
          ? row.metrics.loss
          : null,
    samples: row.samples ?? row.sample_count ?? row.count ?? null,
    quality:
      typeof row.quality === 'number'
        ? row.quality
        : typeof row.metrics?.quality === 'number'
          ? row.metrics.quality
          : typeof row.accuracy === 'number'
            ? row.accuracy
            : null,
  }))
})

const schedulerItems = computed(() => {
  const runs = props.trainingStatus?.next_runs
  if (!runs) return []
  return Object.entries(runs).map(([key, meta]) => formatSchedulerRow(key, meta as TrainingScheduleMeta))
})

const sendWsProxy = (action: string) => {
  if (!props.sendAdminCommand) return
  return props.sendAdminCommand(action)
}

async function handleSchedulerCommand(payload: { key: string; action: 'start' | 'stop' }) {
  if (!props.sendAdminCommand) return
  const cmd = `scheduler:${payload.action}:${payload.key}`
  schedulerLoading.value = `${payload.key}:${payload.action}`
  try {
    await props.sendAdminCommand(cmd)
  } finally {
    schedulerLoading.value = ''
  }
}

function formatSchedulerRow(key: string, meta: TrainingScheduleMeta) {
  const interval = inferInterval(meta)
  return {
    key,
    name: meta.label || key,
    desc: meta.description || '',
    interval,
    lastRun: meta.last_run || null,
    running: Boolean(meta.running),
    enabled: meta.enabled !== false,
    tags: meta.tags,
    type: meta.scope === 'frontend' ? 'frontend' : 'backend',
  }
}

function inferInterval(meta: TrainingScheduleMeta) {
  if (typeof meta.every_minutes === 'number' && meta.every_minutes > 0) {
    return `${meta.every_minutes}m`
  }
  if (typeof meta.every_seconds === 'number' && meta.every_seconds > 0) {
    return `${meta.every_seconds}s`
  }
  return '미설정'
}
</script>

<style scoped>
.training-view {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.head {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}

.head-actions {
  display: flex;
  gap: 0.75rem;
  align-items: center;
}

.badge {
  padding: 0.35rem 0.8rem;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.4);
  font-size: 0.85rem;
}

.badge.ok {
  border-color: rgba(74, 222, 128, 0.5);
  color: #4ade80;
}

.badge.warn {
  border-color: rgba(248, 113, 113, 0.5);
  color: #f87171;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
}

.summary-card {
  border: 1px solid rgba(96, 165, 250, 0.35);
  border-radius: 18px;
  padding: 0.9rem 1rem;
  background: rgba(8, 14, 27, 0.85);
}

.summary-card .label {
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.08em;
  color: rgba(203, 213, 225, 0.85);
}

.summary-card .value {
  font-size: 1.6rem;
  font-weight: 600;
  margin: 0.15rem 0;
}

.summary-card .value.warn {
  color: #facc15;
}

.summary-card .hint {
  font-size: 0.83rem;
  color: rgba(148, 163, 184, 0.85);
}

.meta-grid {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 1rem;
  align-items: start;
}

.scheduler-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1.2rem;
}

@media (min-width: 1100px) {
  .scheduler-grid {
    grid-template-columns: 1fr 1fr;
  }
}

@media (max-width: 900px) {
  .meta-grid {
    grid-template-columns: 1fr;
  }
}
</style>
