import { ref, computed, onMounted, onBeforeUnmount, getCurrentInstance } from 'vue'
import type {
  AdminAckMessage,
  EntryMetrics,
  EntryMetricsMessage,
  FeatureSnapshot,
  FeaturesMessage,
  NowcastEntry,
  NowcastMap,
  NowcastMessage,
  SnapshotPayload,
  TradeRow,
  TradesMessage,
  TrainerMessage,
  TrainerMeta,
  TrainingStatus,
  TrainingStatusMessage,
  UiNotification,
  UseRealtimeHandle,
  UseRealtimeOptions,
  WsInboundMessage,
} from '../types/realtime'

const DEFAULT_API_BASE = 'http://127.0.0.1:8000'
const DEFAULT_WS_PORT = 8022
const MAX_NOTIFICATIONS = 8

interface PollHandles {
  nowcast: ReturnType<typeof setInterval> | null
  trades: ReturnType<typeof setInterval> | null
  features: ReturnType<typeof setInterval> | null
  training: ReturnType<typeof setInterval> | null
}

export function useRealtimeData(options: UseRealtimeOptions = {}): UseRealtimeHandle {
  const apiBase = ref(options.apiBase || inferInitialApiBase())
  const wsConnected = ref(false)
  const wsAttempts = ref(0)
  const wsConnectedSince = ref<number | null>(null)
  const lastUpdated = ref<Date | null>(null)
  const error = ref('')

  const nowcasts = ref<NowcastMap>({})
  const features = ref<Record<string, FeatureSnapshot>>({})
  const trades = ref<TradeRow[]>([])
  const trainingStatus = ref<TrainingStatus | null>(null)
  const trainerMeta = ref<TrainerMeta | null>(null)
  const entryMetrics = ref<EntryMetrics | null>(null)
  const notifications = ref<UiNotification[]>([])
  const adminAck = ref('')

  const symbolList = computed(() =>
    Object.keys(nowcasts.value).filter((sym) => !sym.startsWith('_') && nowcasts.value[sym])
  )

  let ws: WebSocket | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let pollers: PollHandles = {
    nowcast: null,
    trades: null,
    features: null,
    training: null,
  }

  let notificationSeq = 0
  let started = false
  let destroying = false
  let queueIdleSleep = Math.max(0.05, options.featuresIntervalSeconds ?? 0.05)
  let pollIntervalSeconds = Math.max(5, options.pollIntervalSeconds ?? 10)
  let tradesIntervalSeconds = Math.max(15, options.tradesIntervalSeconds ?? 45)
  let featuresIntervalSeconds = Math.max(30, options.featuresIntervalSeconds ?? 45)
  let trainingIntervalSeconds = Math.max(45, options.trainingStatusIntervalSeconds ?? 60)

  const autoStart = options.autoStart !== false && !!getCurrentInstance()

  function pushNotification(message: string, level: UiNotification['level'] = 'info') {
    const entry: UiNotification = {
      id: ++notificationSeq,
      ts: new Date().toISOString(),
      level,
      message,
    }
    notifications.value = [entry, ...notifications.value].slice(0, MAX_NOTIFICATIONS)
  }

  function handleSnapshot(payload: SnapshotPayload) {
    if (payload.nowcast) nowcasts.value = payload.nowcast
    if (payload.trades) trades.value = payload.trades
    if (payload.features) features.value = payload.features
    if (payload.training_status) trainingStatus.value = payload.training_status
    if (payload.trainer) trainerMeta.value = payload.trainer
    if (payload.entry_metrics) entryMetrics.value = payload.entry_metrics
    lastUpdated.value = new Date()
  }

  function mergeNowcast(symbol: string, data: NowcastEntry) {
    nowcasts.value = {
      ...nowcasts.value,
      [symbol]: {
        ...(nowcasts.value[symbol] || {}),
        ...data,
      },
    }
  }

  function handleNowcastMessage(payload: NowcastMessage) {
    mergeNowcast(payload.symbol, payload.data)
    lastUpdated.value = new Date()
  }

  function handleTradesMessage(payload: TradesMessage) {
    trades.value = payload.data
  }

  function handleFeaturesMessage(payload: FeaturesMessage) {
    features.value = { ...features.value, ...payload.data }
  }

  function handleTrainerMessage(payload: TrainerMessage) {
    trainerMeta.value = payload.data
  }

  function handleTrainingStatusMessage(payload: TrainingStatusMessage) {
    trainingStatus.value = payload.data
  }

  function handleEntryMetricsMessage(payload: EntryMetricsMessage) {
    entryMetrics.value = payload.data
  }

  function handleAdminAckMessage(payload: AdminAckMessage) {
    const action = payload.action || 'command'
    if (payload.ok) {
      adminAck.value = `WS: ${action} 완료`
      pushNotification(`Admin ${action} succeeded`, 'info')
    } else {
      const msg = payload.error ? `${action} 실패 · ${payload.error}` : `${action} 실패`
      adminAck.value = `WS: ${msg}`
      pushNotification(msg, 'warn')
    }
  }

  function handleWsMessage(evt: MessageEvent) {
    try {
      const payload = JSON.parse(evt.data) as WsInboundMessage
      switch (payload.type) {
        case 'snapshot':
          if (isSnapshotMessage(payload)) {
            handleSnapshot(payload)
            stopPollingLoops()
          }
          break
        case 'nowcast':
          if (isNowcastMessage(payload)) handleNowcastMessage(payload)
          break
        case 'trades':
          if (isTradesMessage(payload)) handleTradesMessage(payload)
          break
        case 'features':
          if (isFeaturesMessage(payload)) handleFeaturesMessage(payload)
          break
        case 'trainer':
          if (isTrainerMessage(payload)) handleTrainerMessage(payload)
          break
        case 'training_status':
          if (isTrainingStatusMessage(payload)) handleTrainingStatusMessage(payload)
          break
        case 'entry_metrics':
          if (isEntryMetricsMessage(payload)) handleEntryMetricsMessage(payload)
          break
        case 'admin_ack':
          if (isAdminAckMessage(payload)) handleAdminAckMessage(payload)
          break
        default:
          break
      }
    } catch (err) {
      console.warn('[ws] message parse failed', err)
    }
  }

  function deriveWsUrl() {
    if (options.wsUrl) return options.wsUrl
    if (typeof window === 'undefined') return ''
    const forced = (window as any).VITE_WS_URL || (window as any).WS_URL_OVERRIDE
    if (forced) return forced
    const host = window.location?.hostname || '127.0.0.1'
    const proto = window.location?.protocol === 'https:' ? 'wss' : 'ws'
    return `${proto}://${host}:${DEFAULT_WS_PORT}`
  }

  async function connectWs() {
    if (typeof window === 'undefined') return
    const url = deriveWsUrl()
    if (!url) return
    if (ws && ws.readyState === WebSocket.OPEN) return
    if (ws && ws.readyState === WebSocket.CONNECTING) return
    await disconnectWs()

    try {
      ws = new WebSocket(url)
    } catch (err) {
      scheduleReconnect()
      pushNotification('WebSocket 초기화 실패', 'warn')
      return
    }

    wsAttempts.value += 1
    ws.onopen = () => {
      wsConnected.value = true
      wsConnectedSince.value = Date.now()
      pushNotification('WS 연결됨')
      stopPollingLoops()
    }
    ws.onmessage = handleWsMessage
    ws.onerror = () => {
      pushNotification('WS 에러 감지 · HTTP 폴백 유지', 'warn')
    }
    ws.onclose = () => {
      wsConnected.value = false
      wsConnectedSince.value = null
      if (!destroying) {
        pushNotification('WS 해제 · 폴백 시작', 'warn')
        if (!options.wsOnly) startPollingLoops()
        scheduleReconnect()
      }
    }
  }

  function scheduleReconnect() {
    if (options.wsOnly || destroying) return
    if (reconnectTimer) clearTimeout(reconnectTimer)
    const attempt = Math.min(wsAttempts.value, 5)
    const delay = Math.min(60000, 4000 * Math.pow(2, attempt))
    reconnectTimer = setTimeout(() => {
      connectWs()
    }, delay)
  }

  async function disconnectWs() {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    if (ws) {
      try {
        ws.close()
      } catch {
        // ignore
      }
      ws = null
    }
  }

  async function fetchJson<T>(path: string): Promise<T | null> {
    try {
      const response = await fetch(path)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const contentType = response.headers.get('content-type') || ''
      if (!contentType.toLowerCase().includes('application/json')) {
        throw new Error('non-json response')
      }
      return (await response.json()) as T
    } catch (err) {
      error.value = (err as Error).message
      return null
    }
  }

  async function fetchNowcast() {
    const base = apiBase.value
    if (!base) return
    const data = await fetchJson<NowcastMap>(`${base}/nowcast`)
    if (data && typeof data === 'object' && !Array.isArray(data)) {
      nowcasts.value = data
      lastUpdated.value = new Date()
    }
  }

  async function fetchTrades() {
    const base = apiBase.value
    if (!base) return
    const data = await fetchJson<TradeRow[]>(`${base}/trades?limit=50`)
    if (Array.isArray(data)) trades.value = data
  }

  async function fetchTrainingStatus() {
    const base = apiBase.value
    if (!base) return
    const data = await fetchJson<TrainingStatus>(`${base}/training/status`)
    if (data) {
      trainingStatus.value = data
      if ((data as any).trainer) trainerMeta.value = (data as any).trainer
      if ((data as any).entry_metrics) entryMetrics.value = (data as any).entry_metrics
    }
  }

  async function fetchFeatureSnapshots(symbols?: string[]) {
    const base = apiBase.value
    if (!base) return
    const targetSymbols = symbols || symbolList.value
    await Promise.all(
      targetSymbols.map(async (sym) => {
        const snapshot = await fetchJson<FeatureSnapshot>(
          `${base}/health/features?symbol=${encodeURIComponent(sym)}`
        )
        if (snapshot) {
          features.value = {
            ...features.value,
            [sym]: snapshot,
          }
        }
      })
    )
  }

  const pollControls = {
    startNowcast() {
      if (options.wsOnly) return
      if (pollers.nowcast) clearInterval(pollers.nowcast)
      pollers.nowcast = setInterval(fetchNowcast, pollIntervalSeconds * 1000)
    },
    startTrades() {
      if (options.wsOnly) return
      if (pollers.trades) clearInterval(pollers.trades)
      pollers.trades = setInterval(fetchTrades, tradesIntervalSeconds * 1000)
    },
    startFeatures() {
      if (options.wsOnly) return
      if (pollers.features) clearInterval(pollers.features)
      pollers.features = setInterval(() => fetchFeatureSnapshots(), featuresIntervalSeconds * 1000)
    },
    startTraining() {
      if (options.wsOnly) return
      if (pollers.training) clearInterval(pollers.training)
      pollers.training = setInterval(fetchTrainingStatus, trainingIntervalSeconds * 1000)
    },
    stopAll() {
      Object.keys(pollers).forEach((key) => {
        const handle = pollers[key as keyof PollHandles]
        if (handle) clearInterval(handle)
        pollers[key as keyof PollHandles] = null
      })
    },
  }

  function startPollingLoops() {
    if (options.wsOnly) return
    pollControls.startNowcast()
    pollControls.startTrades()
    pollControls.startFeatures()
    pollControls.startTraining()
  }

  function stopPollingLoops() {
    pollControls.stopAll()
  }

  async function autoDetectApiBase() {
    if (apiBase.value && apiBase.value !== DEFAULT_API_BASE) return
    if (typeof window === 'undefined') return
    const candidates = new Set<string>()
    if (options.apiBase) candidates.add(options.apiBase)
    const sameOrigin = `${window.location.protocol}//${window.location.host}`
    candidates.add(sameOrigin)
    candidates.add(DEFAULT_API_BASE)

    for (const candidate of candidates) {
      try {
        const controller = new AbortController()
        const timeout = setTimeout(() => controller.abort(), 1200)
        const res = await fetch(`${candidate}/health`, {
          signal: controller.signal,
          headers: { Accept: 'application/json' },
        })
        clearTimeout(timeout)
        if (res.ok) {
          apiBase.value = candidate
          ;(window as any).VITE_API_BASE = candidate
          return
        }
      } catch {
        // try next
      }
    }
  }

  async function manualRefresh() {
    await Promise.all([fetchNowcast(), fetchTrades(), fetchTrainingStatus(), fetchFeatureSnapshots()])
  }

  async function start() {
    if (started) return
    started = true
    if (!options.wsOnly) {
      await autoDetectApiBase()
      await Promise.all([fetchNowcast(), fetchTrades(), fetchTrainingStatus(), fetchFeatureSnapshots()])
      startPollingLoops()
    }
    await connectWs()
  }

  function stop() {
    destroying = true
    stopPollingLoops()
    disconnectWs()
  }

  async function sendAdminCommand(action: string, token?: string) {
    try {
      if (!ws || ws.readyState !== WebSocket.OPEN) throw new Error('WS not connected')
      const payload = { type: 'admin', action, token: token || (window as any).WS_ADMIN_TOKEN || '' }
      ws.send(JSON.stringify(payload))
      adminAck.value = `WS: ${action} 전송`
      pushNotification(`Admin ${action} 전송`, 'info')
    } catch (err) {
      const msg = (err as Error).message || 'admin send failed'
      adminAck.value = `WS: ${msg}`
      pushNotification(msg, 'warn')
    }
  }

  if (autoStart) {
    onMounted(() => {
      start()
    })
  }

  onBeforeUnmount(() => {
    stop()
  })

  return {
    apiBase,
    wsConnected,
    wsAttempts,
    wsConnectedSince,
    lastUpdated,
    error,
    nowcasts,
    features,
    trades,
    trainingStatus,
    trainerMeta,
    entryMetrics,
    notifications,
    adminAck,
    start,
    stop,
    connectWs,
    disconnectWs,
    manualRefresh,
    sendAdminCommand,
  }
}

function inferInitialApiBase() {
  if (typeof window === 'undefined') return DEFAULT_API_BASE
  return (window as any).VITE_API_BASE || DEFAULT_API_BASE
}

function isSnapshotMessage(msg: WsInboundMessage): msg is SnapshotPayload {
  return msg.type === 'snapshot'
}

function isNowcastMessage(msg: WsInboundMessage): msg is NowcastMessage {
  return msg.type === 'nowcast' && typeof (msg as NowcastMessage).symbol === 'string'
}

function isTradesMessage(msg: WsInboundMessage): msg is TradesMessage {
  return msg.type === 'trades' && Array.isArray((msg as TradesMessage).data)
}

function isFeaturesMessage(msg: WsInboundMessage): msg is FeaturesMessage {
  return msg.type === 'features' && typeof (msg as FeaturesMessage).data === 'object'
}

function isTrainerMessage(msg: WsInboundMessage): msg is TrainerMessage {
  return msg.type === 'trainer' && typeof (msg as TrainerMessage).data === 'object'
}

function isTrainingStatusMessage(msg: WsInboundMessage): msg is TrainingStatusMessage {
  return msg.type === 'training_status' && typeof (msg as TrainingStatusMessage).data === 'object'
}

function isEntryMetricsMessage(msg: WsInboundMessage): msg is EntryMetricsMessage {
  return msg.type === 'entry_metrics' && typeof (msg as EntryMetricsMessage).data === 'object'
}

function isAdminAckMessage(msg: WsInboundMessage): msg is AdminAckMessage {
  return msg.type === 'admin_ack'
}
