/*
 * Shared types for realtime dashboard data coming from the backend WS/HTTP APIs.
 * These interfaces intentionally keep optional fields to accommodate partial payloads
 * while still providing structure for the frontend components/composables.
 */

export type NotificationLevel = 'info' | 'warn' | 'error'

export interface UiNotification {
  id: number
  ts: string
  level: NotificationLevel
  message: string
}

export interface LayoutPreset {
  id: string
  name: string
  filter: string
  sortKey: string
  sortDir: 'asc' | 'desc'
}

export interface TradeFillRow {
  t: string
  price: number
  qty: number
}

export interface TradeRow {
  id: number
  symbol: string
  status: 'open' | 'closed' | 'pending' | string
  leverage?: number
  quantity: number
  entry_price: number
  avg_price: number
  last_price?: number
  take_profit_pct: number
  stop_loss_pct: number
  pnl_pct_snapshot: number
  adds_done: number
  next_add_in_seconds?: number
  created_at: string
  closed_at?: string | null
  fills?: TradeFillRow[]
  last_fill_at?: string
  cooldown_seconds?: number
}

export interface EntryMetaHistoryRow {
  ts: string
  rel_improve?: number | null
  brier_old?: number | null
  brier_new?: number | null
  samples?: number | null
  overwritten?: boolean
}

export interface EntryMetrics {
  window?: number
  target?: number
  target_effective?: number
  overall?: {
    win_rate?: number
    samples?: number
  }
  threshold?: {
    env?: number
    sidecar?: number
    dynamic?: number
  }
  history_overall?: Array<{ ts: string; wr?: number | null }>
  by_symbol?: Record<string, { win_rate?: number; samples?: number }>
}

export interface TrainerMeta {
  heartbeat_healthy?: boolean
  heartbeat_age_seconds?: number
  last_run?: {
    start?: string
    end?: string
    base_days?: number
    stacking_days?: number
  }
  last_run_age_seconds?: number
}

export interface TrainingScheduleMeta {
  label?: string
  description?: string
  every_minutes?: number
  every_seconds?: number
  eta_seconds?: number
  state?: 'running' | 'idle' | 'disabled'
  last_run?: string | null
  running?: boolean
  enabled?: boolean
  tags?: string[]
  models?: string[]
  scope?: 'backend' | 'frontend' | string
}

export interface TrainingStatus {
  pending_meta_retrain?: boolean
  last_retrain_utc?: string
  last_retrain_reason?: string
  last_meta_retrain_utc?: string
  last_meta_retrain_reason?: string
  last_meta_retrain_overwritten?: boolean
  pending_meta_retrain_reason?: string
  meta_history?: EntryMetaHistoryRow[]
  meta_corr_rel_improve_samples?: number | null
  meta_neg_streak?: number
  meta_neg_streak_warn?: boolean
  meta_reg_slope?: number | null
  meta_reg_pvalue?: number | null
  meta_coef_cosine?: number | null
  meta_coef_drift_warn?: boolean
  next_runs?: Record<string, TrainingScheduleMeta>
  model_training_rows?: Array<Record<string, unknown>>
  model_training?: Array<Record<string, unknown>>
}

export interface FeatureSnapshot extends Record<string, string | number | boolean | null | undefined> {
  data_fresh_seconds?: number
  missing_minutes_24h?: number
  ['5m_latest_open_time']?: string
  ['15m_latest_open_time']?: string
}

export interface StackingEntry {
  ready?: boolean
  prob?: number
  prob_final?: number | null
  confidence?: number | null
  decision?: boolean
  margin?: number | null
  threshold?: number | null
  used_models?: string[]
  entry_meta?: {
    entry_decision?: boolean
    entry_prob?: number
  }
}

export interface NowcastEntry {
  symbol?: string
  interval?: string
  timestamp?: string
  price?: number
  price_source?: string
  bottom_score?: number
  components?: Record<string, number>
  base_probs?: Record<string, number | null>
  base_info?: Record<string, Record<string, number | boolean | null>>
  stacking?: StackingEntry
  signals?: ChartSignal[]
  chart_signals?: ChartSignal[]
}

export type NowcastMap = Record<string, NowcastEntry>

export interface ChartSignal {
  id: string | number
  ts: string
  side: 'buy' | 'sell'
  price?: number
  label?: string
}

export interface SnapshotPayload {
  type: 'snapshot'
  nowcast?: NowcastMap
  trades?: TradeRow[]
  features?: Record<string, FeatureSnapshot>
  training_status?: TrainingStatus
  trainer?: TrainerMeta
  entry_metrics?: EntryMetrics
}

export interface NowcastMessage {
  type: 'nowcast'
  symbol: string
  data: NowcastEntry
}

export interface TradesMessage {
  type: 'trades'
  data: TradeRow[]
}

export interface FeaturesMessage {
  type: 'features'
  data: Record<string, FeatureSnapshot>
}

export interface TrainerMessage {
  type: 'trainer'
  data: TrainerMeta
}

export interface TrainingStatusMessage {
  type: 'training_status'
  data: TrainingStatus
}

export interface EntryMetricsMessage {
  type: 'entry_metrics'
  data: EntryMetrics
}

export interface AdminAckMessage {
  type: 'admin_ack'
  action?: string
  ok?: boolean
  error?: string
}

export type WsInboundMessage =
  | SnapshotPayload
  | NowcastMessage
  | TradesMessage
  | FeaturesMessage
  | TrainerMessage
  | TrainingStatusMessage
  | EntryMetricsMessage
  | AdminAckMessage
  | { type: string; [key: string]: unknown }

export interface UseRealtimeOptions {
  wsUrl?: string
  apiBase?: string
  wsOnly?: boolean
  autoStart?: boolean
  pollIntervalSeconds?: number
  tradesIntervalSeconds?: number
  featuresIntervalSeconds?: number
  trainingStatusIntervalSeconds?: number
}

export interface UseRealtimeHandle {
  // reactive state
  apiBase: Ref<string>
  wsConnected: Ref<boolean>
  wsAttempts: Ref<number>
  wsConnectedSince: Ref<number | null>
  lastUpdated: Ref<Date | null>
  error: Ref<string>
  nowcasts: Ref<NowcastMap>
  features: Ref<Record<string, FeatureSnapshot>>
  trades: Ref<TradeRow[]>
  trainingStatus: Ref<TrainingStatus | null>
  trainerMeta: Ref<TrainerMeta | null>
  entryMetrics: Ref<EntryMetrics | null>
  notifications: Ref<UiNotification[]>
  adminAck: Ref<string>

  // lifecycle helpers
  start: () => Promise<void>
  stop: () => void
  connectWs: () => Promise<void>
  disconnectWs: () => Promise<void>
  manualRefresh: () => Promise<void>
  sendAdminCommand: (action: string, token?: string) => Promise<void>
}

// Vue's Ref type import helper (allows tree-shaking when used outside of *.vue files)
export type Ref<T> = import('vue').Ref<T>
