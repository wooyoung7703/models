<template>
  <div class="app-shell">
    <header class="masthead">
      <div class="masthead__titles">
        <h1>Real-Time Bottom Detector</h1>
        <p class="muted">실시간 바텀 탐지 대시보드</p>
      </div>
      <div class="masthead__status">
        <div class="api-pill">
          <span>API</span>
          <input class="api-input" v-model="apiBase" spellcheck="false" />
        </div>
        <span class="status-chip" :class="healthDot">{{ healthText }}</span>
        <span class="muted">업데이트 {{ lastUpdatedRelative }}</span>
        <button class="btn ghost" @click="manualRefresh" :disabled="loading">동기화</button>
      </div>
    </header>

    <section class="system-strip">
      <article class="metric-card">
        <header>시스템 상태</header>
        <div class="metric-card__body">
          <p class="health-line">
            <span class="dot" :class="healthDot"></span>
            <span>{{ healthText }}</span>
          </p>
          <p class="muted">최근 {{ lastUpdatedRelative }}</p>
          <p class="muted" v-if="error">{{ error }}</p>
        </div>
      </article>

      <article class="metric-card">
        <header>신호 현황</header>
        <div class="metric-card__body distribution">
          <div class="dist-row">
            <span>ACTIVE</span>
            <div class="bar"><span :style="{ width: activeShare + '%' }"></span></div>
            <strong>{{ activeSignals.length }}</strong>
          </div>
          <div class="dist-row">
            <span>HIGH</span>
            <div class="bar warn"><span :style="{ width: highConfidenceShare + '%' }"></span></div>
            <strong>{{ highConfidenceSignals.length }}</strong>
          </div>
          <p class="muted">총 {{ displayedSymbols.length }}개 심볼</p>
        </div>
      </article>

      <article class="metric-card" v-if="stackingMeta">
        <header>스태킹 설정</header>
        <div class="metric-card__body meta-grid">
          <p><span class="label">방법</span><span>{{ stackingMeta.method || 'auto' }}</span></p>
          <p v-if="stackingMeta.method_override"><span class="label">Override</span><span>{{ stackingMeta.method_override }}</span></p>
          <p><span class="label">Threshold</span><span>{{ formatThreshold(stackingMeta.threshold) }}</span></p>
          <p><span class="label">Source</span><span>{{ stackingMeta.threshold_source || 'auto' }}</span></p>
          <p><span class="label">Models</span><span>{{ stackingMeta.used_models?.length || stackingMeta.models?.length || 0 }}</span></p>
        </div>
      </article>

      <article class="metric-card">
        <header>폴링 주기</header>
        <div class="metric-card__body interval-picker">
          <input type="range" min="5" max="45" step="5" v-model.number="pollInterval" />
          <span>{{ pollInterval }}초</span>
        </div>
      </article>
    </section>

    <main class="main-grid" v-if="displayedSymbols.length">
      <section class="symbol-board">
        <header class="board-header">
          <input class="filter-box" v-model="symbolFilter" placeholder="심볼 검색" />
          <div class="sort-controls">
            <label>
              <span>정렬</span>
              <select v-model="sortKey">
                <option value="stack_prob">스택 확률</option>
                <option value="bottom_score">바텀 점수</option>
                <option value="symbol">심볼</option>
              </select>
            </label>
            <label>
              <span>방향</span>
              <select v-model="sortDir">
                <option value="desc">내림차순</option>
                <option value="asc">오름차순</option>
              </select>
            </label>
          </div>
        </header>
        <div class="symbol-grid">
          <article
            v-for="sym in sortedAndFilteredSymbols"
            :key="sym"
            :class="['symbol-card', sym === selectedSymbol ? 'selected' : '', nowcasts[sym].stacking?.decision ? 'decision-on' : 'decision-off']"
            @click="selectedSymbol = sym"
          >
            <header class="symbol-card__top">
              <div>
                <h2>{{ sym.toUpperCase() }}</h2>
                <p class="muted">{{ formatInterval(nowcasts[sym].interval) }}</p>
              </div>
              <time :datetime="nowcasts[sym].timestamp">{{ formatTimestamp(nowcasts[sym].timestamp) }}</time>
            </header>
            <div class="symbol-card__metrics">
              <div class="metric">
                <span class="label">Price</span>
                <strong>{{ formatPrice(nowcasts[sym].price) }}</strong>
              </div>
              <div class="metric">
                <span class="label">Bottom</span>
                <strong :class="scoreClass(nowcasts[sym].bottom_score)">{{ formatPct(nowcasts[sym].bottom_score) }}</strong>
              </div>
              <div class="metric" v-if="nowcasts[sym].stacking">
                <span class="label">Stack</span>
                <strong>{{ formatPct(nowcasts[sym].stacking?.prob) }}</strong>
              </div>
              <div class="metric" v-if="nowcasts[sym].stacking">
                <span class="label">Margin</span>
                <strong :class="marginClass(nowcasts[sym].stacking?.margin)">{{ formatSignedPct(nowcasts[sym].stacking?.margin) }}</strong>
              </div>
            </div>
            <div class="symbol-card__spark">
              <svg viewBox="0 0 100 42" preserveAspectRatio="none">
                <polyline :points="sparklinePoints(sym)" />
              </svg>
            </div>
            <footer class="symbol-card__badges">
              <span class="badge" :class="freshnessBadgeClass(sym)">{{ freshnessBadgeText(sym) }}</span>
              <span class="badge" :class="gapBadgeClass(sym)">{{ gapBadgeText(sym) }}</span>
              <span class="badge" :class="tfBadgeClass(sym)">{{ tfBadgeText(sym) }}</span>
              <span class="badge" :class="strengthClass(sym)">{{ shortStrength(sym) }}</span>
            </footer>
          </article>
        </div>
      </section>

      <section class="detail-panel" v-if="selectedSymbol && nowcasts[selectedSymbol]">
        <header class="detail-header">
          <div>
            <h2>{{ selectedSymbol.toUpperCase() }} 상세</h2>
            <p class="muted">{{ detailSubtitle }}</p>
          </div>
          <div class="pill-group" v-if="selectedStacking">
            <span class="pill" :class="selectedStacking.decision ? 'on' : 'off'">
              {{ selectedStacking.decision ? 'ON' : 'OFF' }}
            </span>
            <span class="pill muted">Conf {{ formatPct((selectedStacking.confidence ?? 0)) }}</span>
            <span class="pill muted">Margin {{ formatSignedPct(selectedStacking.margin) }}</span>
          </div>
        </header>

        <section class="detail-grid">
          <article class="detail-card">
            <header>확률 구조</header>
            <div class="prob-bars" v-if="selectedStacking">
              <div class="bar-row">
                <span>RAW</span>
                <div class="progress">
                  <span :style="{ width: pctWidth(selectedStacking.prob_raw ?? selectedStacking.raw_prob) }"></span>
                </div>
                <strong>{{ formatPct((selectedStacking.prob_raw ?? selectedStacking.raw_prob ?? 0)) }}</strong>
              </div>
              <div class="bar-row">
                <span>SMOOTH</span>
                <div class="progress">
                  <span class="smooth" :style="{ width: pctWidth(selectedStacking.prob_smoothed ?? selectedStacking.prob) }"></span>
                </div>
                <strong>{{ formatPct(selectedStacking.prob_smoothed ?? selectedStacking.prob) }}</strong>
              </div>
              <div class="bar-row">
                <span>FINAL</span>
                <div class="progress">
                  <span class="final" :style="{ width: pctWidth(selectedStacking.prob_final ?? selectedStacking.prob) }"></span>
                </div>
                <strong>{{ formatPct(selectedStacking.prob_final ?? selectedStacking.prob) }}</strong>
              </div>
              <div class="bar-row threshold">
                <span>THRESH</span>
                <div class="progress">
                  <span class="threshold" :style="{ width: pctWidth(selectedStacking.threshold) }"></span>
                </div>
                <strong>{{ formatThreshold(selectedStacking.threshold) }}</strong>
              </div>
            </div>
            <p v-else class="muted empty-note">Staking block unavailable.</p>
          </article>

          <article class="detail-card">
            <header>기저 모델</header>
            <ul class="list" v-if="baseProbEntries.length">
              <li v-for="[model, value] in baseProbEntries" :key="model">
                <span>{{ model }}</span>
                <div class="progress compact">
                  <span :style="{ width: pctWidth(value) }"></span>
                </div>
                <strong>{{ formatBaseProb(value) }}</strong>
              </li>
            </ul>
            <p v-else class="muted empty-note">모델 확률 데이터 없음</p>
          </article>

          <article class="detail-card">
            <header>특징 기여도</header>
            <ul class="list components">
              <li v-for="(value, key) in filteredComponents(selectedNowcast?.components || {})" :key="key">
                <span>{{ key }}</span>
                <strong>{{ formatComponent(value) }}</strong>
              </li>
            </ul>
          </article>

          <article class="detail-card">
            <header>데이터 품질</header>
            <div class="quality" v-if="featureSnapshot">
              <p><span class="label">Freshness</span><span>{{ freshnessBadgeText(selectedSymbol) }}</span></p>
              <p><span class="label">누락 분</span><span>{{ featureSnapshot.missing_minutes_24h ?? 0 }}</span></p>
              <p><span class="label">TF</span><span>{{ tfBadgeText(selectedSymbol) }}</span></p>
              <p><span class="label">Source</span><span>{{ selectedNowcast?.price_source }}</span></p>
            </div>
            <p v-else class="muted empty-note">헬스 데이터를 기다리는 중…</p>
          </article>
        </section>
      </section>
    </main>

    <section class="trades-pane" v-if="trades.length">
      <header class="pane-header">
        <h2>실거래 내역</h2>
        <button class="btn ghost" @click="manualTradesRefresh" :disabled="tradesLoading">새로고침</button>
      </header>
      <table class="trades-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>심볼</th>
            <th>상태</th>
            <th>레버리지</th>
            <th>수량</th>
            <th>진입</th>
            <th>평단</th>
            <th>현재</th>
            <th>익절/손절</th>
            <th>진행 손익</th>
            <th>추가 매수</th>
            <th>다음 추가</th>
            <th>생성</th>
            <th>종료</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="row in tradeRows"
            :key="row.kind === 'trade' ? `trade-${row.trade.id}` : `fills-${row.trade.id}`"
            :class="[row.trade.status, row.kind === 'fills' ? 'fills-row' : '']"
          >
            <template v-if="row.kind === 'trade'">
              <td>
                <div class="id-cell">
                  <span>{{ row.trade.id }}</span>
                  <button class="link" @click.prevent="toggleExpanded(row.trade.id)">
                    {{ expanded[row.trade.id] ? '접기' : '체결' }}
                  </button>
                </div>
              </td>
              <td>{{ row.trade.symbol.toUpperCase() }}</td>
              <td><span class="status-badge" :class="row.trade.status">{{ row.trade.status }}</span></td>
              <td>{{ row.trade.leverage }}x</td>
              <td>{{ row.trade.quantity.toFixed(3) }}</td>
              <td>{{ formatNumber(row.trade.entry_price) }}</td>
              <td>{{ formatNumber(row.trade.avg_price) }}</td>
              <td>{{ row.trade.last_price != null ? formatNumber(row.trade.last_price) : '-' }}</td>
              <td>{{ formatSignedPct(row.trade.take_profit_pct) }} / {{ formatSignedPct(row.trade.stop_loss_pct) }}</td>
              <td :class="row.trade.pnl_pct_snapshot > 0 ? 'pos' : row.trade.pnl_pct_snapshot < 0 ? 'neg' : ''">
                {{ formatSignedPct(row.trade.pnl_pct_snapshot) }}
              </td>
              <td>{{ row.trade.adds_done }}/{{ row.trade.max_adds }}</td>
              <td>
                <span v-if="row.trade.next_add_in_seconds != null && row.trade.status === 'open'">
                  <span v-if="row.trade.next_add_in_seconds > 0">{{ formatDuration(row.trade.next_add_in_seconds) }}</span>
                  <span v-else class="pos">가능</span>
                </span>
                <span v-else>-</span>
              </td>
              <td>{{ formatShortDt(row.trade.created_at) }}</td>
              <td>{{ row.trade.closed_at ? formatShortDt(row.trade.closed_at) : '-' }}</td>
            </template>
            <template v-else>
              <td colspan="14">
                <div class="fills-wrap">
                  <div class="fills-meta">
                    <span>마지막 체결 {{ row.trade.last_fill_at ? formatShortDt(row.trade.last_fill_at) : '-' }}</span>
                    <span v-if="row.trade.cooldown_seconds">쿨다운 {{ Math.floor(row.trade.cooldown_seconds / 60) }}분</span>
                  </div>
                  <table class="fills-table">
                    <thead>
                      <tr>
                        <th>시간</th>
                        <th>가격</th>
                        <th>수량</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="(fill, idx) in row.trade.fills" :key="idx">
                        <td>{{ formatShortDt(fill.t) }}</td>
                        <td class="num">{{ formatNumber(fill.price) }}</td>
                        <td class="num">{{ Number(fill.qty).toFixed(3) }}</td>
                      </tr>
                      <tr v-if="!row.trade.fills || !row.trade.fills.length">
                        <td colspan="3" class="muted">체결 기록 없음</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </td>
            </template>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="empty-state" v-else>
      <p>나우캐스트 데이터를 기다리고 있습니다. 백엔드가 온라인인지 확인해주세요.</p>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue'
import { formatPct, formatSignedPct, scoreClass, marginClass } from './utils'

interface NowcastComponents {
  [key: string]: number
}

interface StackingBlock {
  ready: boolean
  prob?: number
  method?: string
  used_models?: string[]
  threshold?: number | null
  threshold_source?: string | null
  threshold_source_final?: string | null
  decision?: boolean | null
  confidence?: number | null
  margin?: number | null
  prob_raw?: number | null
  raw_prob?: number | null
  prob_smoothed?: number | null
  prob_final?: number | null
  threshold_adaptive?: number | null
}

interface NowcastResult {
  symbol: string
  interval: string
  timestamp: string
  price: number
  price_source: string
  bottom_score: number
  components: NowcastComponents
  base_probs?: Record<string, number | null>
  stacking?: StackingBlock
}

type NowcastMap = Record<string, NowcastResult & { [key: string]: any }>

const nowcasts = ref<NowcastMap>({})
const loading = ref<boolean>(false)
const error = ref<string>('')
const pollInterval = ref<number>(10)
const lastUpdated = ref<Date | null>(null)
let pollTimer: number | null = null

const apiBase = ref<string>((window as any).VITE_API_BASE || 'http://127.0.0.1:8000')

const features = ref<Record<string, any>>({})
let featuresTimer: number | null = null
const trades = ref<any[]>([])
const tradesLoading = ref<boolean>(false)
let tradesTimer: number | null = null
const expanded = ref<Record<number, boolean>>({})

const symbolFilter = ref<string>('')
const sortKey = ref<'stack_prob' | 'bottom_score' | 'symbol'>('stack_prob')
const sortDir = ref<'asc' | 'desc'>('desc')
const selectedSymbol = ref<string>('')

const displayedSymbols = computed<string[]>(() =>
  Object.keys(nowcasts.value)
    .filter((key) => !key.startsWith('_') && !!nowcasts.value[key])
)

const sortedAndFilteredSymbols = computed<string[]>(() => {
  let list = displayedSymbols.value
  if (symbolFilter.value.trim()) {
    const q = symbolFilter.value.trim().toLowerCase()
    list = list.filter((sym) => sym.toLowerCase().includes(q))
  }

  const sorted = [...list]
  sorted.sort((a, b) => {
    if (sortKey.value === 'symbol') {
      const cmp = a.localeCompare(b)
      return sortDir.value === 'asc' ? cmp : -cmp
    }
    const getValue = (sym: string) => {
      const nc = nowcasts.value[sym]
      if (!nc) return 0
      if (sortKey.value === 'bottom_score') return Number(nc.bottom_score ?? 0)
      return Number(nc.stacking?.prob ?? 0)
    }
    const diff = getValue(a) - getValue(b)
    return sortDir.value === 'asc' ? diff : -diff
  })
  return sorted
})

watch(displayedSymbols, (syms) => {
  if (!selectedSymbol.value && syms.length) {
    selectedSymbol.value = syms[0]
  } else if (selectedSymbol.value && !syms.includes(selectedSymbol.value) && syms.length) {
    selectedSymbol.value = syms[0]
  } else if (!syms.length) {
    selectedSymbol.value = ''
  }
})

const stackingMeta = computed(() => (nowcasts.value._stacking_meta as { method?: string; method_override?: string | null; threshold?: number | null; threshold_source?: string | null; used_models?: string[]; models?: string[] }) || null)

const activeSignals = computed(() =>
  displayedSymbols.value.filter((sym) => nowcasts.value[sym]?.stacking?.decision)
)

const highConfidenceSignals = computed(() =>
  displayedSymbols.value.filter((sym) => {
    const conf = Number(nowcasts.value[sym]?.stacking?.confidence ?? 0)
    return conf >= 0.2
  })
)

const activeShare = computed(() => {
  if (!displayedSymbols.value.length) return 0
  return Math.round((activeSignals.value.length / displayedSymbols.value.length) * 100)
})

const highConfidenceShare = computed(() => {
  if (!displayedSymbols.value.length) return 0
  return Math.round((highConfidenceSignals.value.length / displayedSymbols.value.length) * 100)
})

const lastUpdatedRelative = computed(() => {
  if (!lastUpdated.value) return '미동기화'
  const diffMs = Date.now() - lastUpdated.value.getTime()
  const seconds = Math.max(0, Math.round(diffMs / 1000))
  if (seconds < 60) return `${seconds}초 전`
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes}분 전`
  const hours = Math.round(minutes / 60)
  return `${hours}시간 전`
})

const healthDot = computed(() => {
  if (error.value) return 'bad'
  if (!displayedSymbols.value.length) return 'warn'
  if (activeSignals.value.length) return 'ok'
  return 'warn'
})

const healthText = computed(() => {
  if (error.value) return `에러: ${error.value}`
  if (!displayedSymbols.value.length) return '데이터 없음 · 백엔드 확인 필요'
  if (activeSignals.value.length) return '정상 작동 중 · 신호 감지'
  return '정상 작동 중 · 대기'
})

const selectedNowcast = computed(() => (selectedSymbol.value ? nowcasts.value[selectedSymbol.value] : null))
const selectedStacking = computed(() => selectedNowcast.value?.stacking && selectedNowcast.value.stacking.ready ? selectedNowcast.value.stacking : null)
const baseProbEntries = computed<[string, number][]>(() => {
  if (!selectedNowcast.value?.base_probs) return []
  const rows: Array<[string, number]> = []
  for (const [k, v] of Object.entries(selectedNowcast.value.base_probs)) {
    rows.push([k.toUpperCase(), Number(v ?? 0)])
  }
  return rows
})

const featureSnapshot = computed(() => (selectedSymbol.value ? features.value[selectedSymbol.value] : null))

const detailSubtitle = computed(() => {
  if (!selectedNowcast.value) return ''
  return `${formatTimestamp(selectedNowcast.value.timestamp)} · ${selectedNowcast.value.price_source}`
})

const pctWidth = (value?: number | null) => {
  if (value == null || isNaN(Number(value))) return '0%'
  const pct = Math.max(0, Math.min(1, Number(value)))
  return `${(pct * 100).toFixed(0)}%`
}

const sparklinePoints = (sym: string) => {
  const comp = nowcasts.value[sym]?.components || {}
  const values = Object.values(comp).map((v) => Number(v)).filter((v) => !Number.isNaN(v))
  if (!values.length) return '0,21 100,21'
  const slice = values.slice(0, 18)
  const max = Math.max(...slice)
  const min = Math.min(...slice)
  const range = max - min || 1
  return slice
    .map((val, idx) => {
      const x = slice.length === 1 ? 100 : (idx / (slice.length - 1)) * 100
      const norm = (val - min) / range
      const y = 36 - norm * 32
      return `${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')
}

const shortStrength = (sym: string) => {
  const cls = strengthClass(sym)
  if (cls === 'strength-high') return '강'
  if (cls === 'strength-mid') return '중'
  return '약'
}

const formatThreshold = (value?: number | null) => {
  if (value == null || Number.isNaN(Number(value))) return 'auto'
  return Number(value).toFixed(2)
}

const formatTimestamp = (ts: string) => {
  try {
    return new Date(ts).toLocaleString()
  } catch {
    return ts
  }
}

const formatInterval = (interval: string) => interval?.toUpperCase?.() || interval

const formatPrice = (price: number) => {
  if (Math.abs(price) >= 100) return price.toFixed(2)
  return price.toFixed(4)
}

const formatNumber = (value: number) => {
  if (value == null || Number.isNaN(Number(value))) return 'n/a'
  if (Math.abs(value) >= 100) return Number(value).toFixed(2)
  if (Math.abs(value) >= 1) return Number(value).toFixed(3)
  return Number(value).toFixed(4)
}

const formatDuration = (seconds: number) => {
  const sec = Math.max(0, Math.floor(seconds))
  const minutes = Math.floor(sec / 60)
  const rest = sec % 60
  return `${minutes}:${String(rest).padStart(2, '0')}`
}

const formatShortDt = (value: string) => {
  try {
    return new Date(value).toLocaleString()
  } catch {
    return value
  }
}

const formatComponent = (value: number) => {
  if (Math.abs(value) >= 1000) return value.toFixed(0)
  if (Math.abs(value) >= 10) return value.toFixed(2)
  return value.toFixed(4)
}

const formatBaseProb = (value?: number | null) => {
  if (value == null || Number.isNaN(Number(value))) return 'n/a'
  const percent = Number(value) * 100
  const abs = Math.abs(percent)
  if (abs >= 1) return `${percent.toFixed(1)}%`
  if (abs >= 0.1) return `${percent.toFixed(2)}%`
  if (abs >= 0.01) return `${percent.toFixed(3)}%`
  return `${percent.toFixed(4)}%`
}

const filteredComponents = (components: NowcastComponents) => {
  const hidden = ['logit']
  const items: Record<string, number> = {}
  Object.entries(components || {}).forEach(([key, value]) => {
    if (!hidden.includes(key)) items[key] = value
  })
  return items
}

const strengthClass = (sym: string) => {
  const stack = nowcasts.value[sym]?.stacking
  if (!stack?.ready) return 'strength-low'
  let score = 0
  const margin = Number(stack.margin ?? 0)
  const conf = Number(stack.confidence ?? 0)
  const bottom = Number(nowcasts.value[sym]?.bottom_score ?? 0)
  if (margin >= 0.05) score += 2
  else if (margin >= 0.02) score += 1
  if (conf >= 0.2) score += 2
  else if (conf >= 0.1) score += 1
  if (bottom >= 0.7) score += 2
  else if (bottom >= 0.55) score += 1
  if (score >= 5) return 'strength-high'
  if (score >= 3) return 'strength-mid'
  return 'strength-low'
}

const freshnessBadgeClass = (sym: string) => {
  const snapshot = features.value[sym]
  if (!snapshot) return 'stale'
  const sec = Number(snapshot.data_fresh_seconds ?? 999)
  if (sec < 90) return 'fresh'
  if (sec < 300) return 'stale'
  return 'cold'
}

const freshnessBadgeText = (sym: string) => {
  const cls = freshnessBadgeClass(sym)
  if (cls === 'fresh') return 'Fresh'
  if (cls === 'cold') return 'Cold'
  return 'Stale'
}

const gapBadgeClass = (sym: string) => {
  const snapshot = features.value[sym]
  const missing = Number(snapshot?.missing_minutes_24h ?? 0)
  if (missing === 0) return 'no-gaps'
  if (missing <= 10) return 'minor-gaps'
  return 'gaps'
}

const gapBadgeText = (sym: string) => {
  const snapshot = features.value[sym]
  const missing = Number(snapshot?.missing_minutes_24h ?? 0)
  return missing === 0 ? 'No Gaps' : `Gaps:${missing}`
}

const tfBadgeClass = (sym: string) => {
  const snapshot = features.value[sym]
  if (!snapshot) return 'tf-missing'
  const has5 = Boolean(snapshot['5m_latest_open_time'])
  const has15 = Boolean(snapshot['15m_latest_open_time'])
  if (has5 && has15) return 'tf-ok'
  if (has5 || has15) return 'tf-partial'
  return 'tf-missing'
}

const tfBadgeText = (sym: string) => {
  const cls = tfBadgeClass(sym)
  if (cls === 'tf-ok') return 'TF OK'
  if (cls === 'tf-partial') return 'TF Partial'
  return 'TF Missing'
}

const toggleExpanded = (id: number) => {
  expanded.value[id] = !expanded.value[id]
}

const tradeRows = computed(() => {
  const rows: Array<{ kind: 'trade' | 'fills'; trade: any }> = []
  trades.value.forEach((trade) => {
    rows.push({ kind: 'trade', trade })
    if (expanded.value[trade.id]) rows.push({ kind: 'fills', trade })
  })
  return rows
})

const fetchNowcast = async () => {
  loading.value = true
  error.value = ''
  try {
    const response = await fetch(`${apiBase.value}/nowcast`)
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const data = await response.json()
    if (data && typeof data === 'object' && !Array.isArray(data)) {
      if (data.symbol && Object.keys(data).length <= 12) {
        const sym = String(data.symbol).toLowerCase()
        nowcasts.value = { [sym]: data }
      } else {
        nowcasts.value = data as NowcastMap
      }
      lastUpdated.value = new Date()
    } else {
      nowcasts.value = {}
    }
  } catch (err: any) {
    error.value = err?.message || '데이터를 불러오는 데 실패했습니다'
  } finally {
    loading.value = false
  }
}

const fetchFeaturesForSymbol = async (sym: string) => {
  try {
    const response = await fetch(`${apiBase.value}/health/features?symbol=${encodeURIComponent(sym)}`)
    if (!response.ok) return
    const data = await response.json()
    features.value = { ...features.value, [sym]: data }
  } catch {
    /* swallow */
  }
}

const fetchTrades = async () => {
  tradesLoading.value = true
  try {
    const response = await fetch(`${apiBase.value}/trades?limit=50`)
    if (!response.ok) return
    const data = await response.json()
    if (Array.isArray(data)) trades.value = data
  } catch {
    /* swallow */
  } finally {
    tradesLoading.value = false
  }
}

const manualRefresh = () => {
  fetchNowcast()
  fetchTrades()
}

const manualTradesRefresh = () => fetchTrades()

const stopPolling = () => {
  if (pollTimer) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
}

const startPolling = () => {
  stopPolling()
  pollTimer = window.setInterval(fetchNowcast, pollInterval.value * 1000)
}

const stopFeaturesPolling = () => {
  if (featuresTimer) {
    window.clearInterval(featuresTimer)
    featuresTimer = null
  }
}

const startFeaturesPolling = () => {
  stopFeaturesPolling()
  const runner = () => displayedSymbols.value.forEach((sym) => fetchFeaturesForSymbol(sym))
  runner()
  featuresTimer = window.setInterval(runner, 45_000)
}

const stopTradesPolling = () => {
  if (tradesTimer) {
    window.clearInterval(tradesTimer)
    tradesTimer = null
  }
}

const startTradesPolling = () => {
  stopTradesPolling()
  const runner = () => fetchTrades()
  runner()
  tradesTimer = window.setInterval(runner, 45_000)
}

const autoDetectApiBase = async () => {
  if ((window as any).VITE_API_BASE) return
  const candidates: string[] = ['http://127.0.0.1:8000']
  for (let port = 8022; port <= 8040; port += 1) candidates.push(`http://127.0.0.1:${port}`)
  const sameOrigin = `${location.protocol}//${location.host}`
  if (!/:5173\b/.test(location.host)) candidates.unshift(sameOrigin)

  const tryHealth = async (base: string) => {
    const controller = new AbortController()
    const timeout = window.setTimeout(() => controller.abort(), 1200)
    try {
      const response = await fetch(`${base}/health`, {
        signal: controller.signal,
        headers: { Accept: 'application/json' }
      })
      if (!response.ok) return false
      const type = response.headers.get('content-type') || ''
      if (!type.toLowerCase().includes('application/json')) return false
      const payload = await response.json().catch(() => null)
      return payload && payload.status === 'ok'
    } catch {
      return false
    } finally {
      window.clearTimeout(timeout)
    }
  }

  for (const candidate of candidates) {
    const ok = await tryHealth(candidate)
    if (ok) {
      apiBase.value = candidate
      ;(window as any).VITE_API_BASE = candidate
      break
    }
  }
}

watch(pollInterval, () => {
  startPolling()
})

onMounted(async () => {
  await autoDetectApiBase()
  await Promise.all([fetchNowcast(), fetchTrades()])
  startPolling()
  startFeaturesPolling()
  startTradesPolling()
})

onBeforeUnmount(() => {
  stopPolling()
  stopFeaturesPolling()
  stopTradesPolling()
})
</script>

<style scoped>
:root {
  --bg: #dbe7ff;
  --card: linear-gradient(180deg, rgba(255, 255, 255, 0.85), rgba(243, 248, 255, 0.95));
  --card-alt: linear-gradient(180deg, rgba(248, 252, 255, 0.92), rgba(229, 239, 255, 0.92));
  --border: rgba(118, 142, 196, 0.28);
  --text: #10254a;
  --muted: #526a94;
  --ok: #1c8747;
  --warn: #d48b00;
  --bad: #d13a3a;
  --accent: #246bff;
  --glow-1: rgba(36, 107, 255, 0.2);
  --glow-2: rgba(20, 160, 227, 0.18);
}

body {
  background: radial-gradient(circle at 18% 12%, rgba(255, 255, 255, 0.75), transparent 65%),
    radial-gradient(circle at 78% -10%, rgba(144, 194, 255, 0.55), transparent 70%),
    linear-gradient(170deg, var(--bg), #f1f6ff);
  color: var(--text);
}

.app-shell {
  min-height: 100vh;
  padding: 2.4rem 3rem 4rem;
  position: relative;
  overflow: hidden;
  color: var(--text);
  font-family: 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
  background: linear-gradient(165deg, rgba(255, 255, 255, 0.7), rgba(229, 240, 255, 0.9));
}

.app-shell::before,
.app-shell::after {
  content: '';
  position: absolute;
  pointer-events: none;
  border-radius: 50%;
  filter: blur(140px);
  opacity: 0.55;
}

.app-shell::before {
  width: 520px;
  height: 520px;
  top: -200px;
  right: -220px;
  background: radial-gradient(circle, rgba(36, 107, 255, 0.35), transparent 72%);
}

.app-shell::after {
  width: 600px;
  height: 600px;
  bottom: -260px;
  left: -200px;
  background: radial-gradient(circle, rgba(20, 160, 227, 0.36), transparent 74%);
}

.app-shell > * {
  position: relative;
  z-index: 1;
}

.api-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.45rem 0.75rem;
  background: rgba(255, 255, 255, 0.75);
  border: 1px solid rgba(129, 161, 214, 0.4);
  border-radius: 9999px;
  font-size: 0.85rem;
  color: var(--muted);
  box-shadow: 0 8px 18px -12px rgba(60, 100, 180, 0.4);
}

.api-input {
  width: 220px;
  border: none;
  background: transparent;
  color: var(--text);
  font-weight: 600;
  font-size: 0.85rem;
}

.api-input:focus {
  outline: none;
}

.status-chip {
  padding: 0.4rem 0.8rem;
  border-radius: 9999px;
  font-size: 0.78rem;
  border: 1px solid transparent;
}

.status-chip.ok {
  background: rgba(34, 197, 94, 0.16);
  border-color: rgba(34, 197, 94, 0.3);
  color: var(--ok);
}

.status-chip.warn {
  background: rgba(250, 204, 21, 0.18);
  border-color: rgba(250, 204, 21, 0.3);
  color: #facc15;
}

.status-chip.bad {
  background: rgba(248, 113, 113, 0.16);
  border-color: rgba(248, 113, 113, 0.32);
  color: var(--bad);
}

.system-strip {
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  margin-bottom: 1.5rem;
}

.metric-card {
  border: 1px solid rgba(129, 161, 214, 0.32);
  border-radius: 20px;
  background: var(--card);
  box-shadow: 0 24px 48px -30px rgba(67, 108, 186, 0.35);
  display: flex;
  flex-direction: column;
  backdrop-filter: saturate(140%) blur(12px);
}

.metric-card header {
  padding: 0.85rem 1rem 0.35rem;
  font-size: 0.85rem;
  letter-spacing: 0.8px;
  color: var(--muted);
  text-transform: uppercase;
}

.metric-card__body {
  padding: 0 1rem 1rem;
  display: grid;
  gap: 0.4rem;
}

.health-line {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0;
  font-weight: 600;
}

.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--muted);
}

.dot.ok {
  background: var(--ok);
}

.dot.warn {
  background: var(--warn);
}

.dot.bad {
  background: var(--bad);
}

.muted {
  color: var(--muted);
  font-size: 0.85rem;
}

.meta-grid p {
  margin: 0;
  display: flex;
  justify-content: space-between;
  font-size: 0.85rem;
}

.meta-grid .label {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.4px;
}

.distribution {
  display: grid;
  gap: 0.6rem;
}

.dist-row {
  display: grid;
  grid-template-columns: 60px 1fr auto;
  gap: 0.75rem;
  align-items: center;
  font-size: 0.8rem;
}

.bar {
  position: relative;
  height: 7px;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 999px;
  overflow: hidden;
}

.bar span {
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, rgba(34, 197, 94, 0.1), rgba(34, 197, 94, 0.6));
}

.bar.warn span {
  background: linear-gradient(90deg, rgba(250, 204, 21, 0.1), rgba(250, 204, 21, 0.6));
}

.interval-picker {
  display: grid;
  gap: 0.5rem;
}

.interval-picker input[type='range'] {
  width: 100%;
}

.main-grid {
  display: grid;
  gap: 1.5rem;
  grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
  align-items: start;
}

.symbol-board {
  border: 1px solid rgba(129, 161, 214, 0.32);
  border-radius: 24px;
  background: var(--card-alt);
  padding: 1.35rem 1.45rem 1.6rem;
  backdrop-filter: saturate(150%) blur(16px);
  box-shadow: 0 32px 60px -34px rgba(67, 108, 186, 0.32);
  position: relative;
  overflow: hidden;
}

.symbol-board::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 50% -20%, rgba(173, 205, 255, 0.55), transparent 62%);
  opacity: 0.5;
}

.symbol-board::after {
  content: '';
  position: absolute;
  inset: 0;
  background-image: linear-gradient(rgba(208, 219, 255, 0.18) 1px, transparent 1px),
    linear-gradient(90deg, rgba(208, 219, 255, 0.18) 1px, transparent 1px);
  background-size: 52px 52px;
  opacity: 0.28;
  mask-image: radial-gradient(circle, rgba(255, 255, 255, 0.85), transparent 70%);
}

.symbol-board > * {
  position: relative;
  z-index: 1;
}

.board-header {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.filter-box {
  flex: 1;
  min-width: 180px;
  padding: 0.6rem 0.8rem;
  border-radius: 12px;
  border: 1px solid rgba(129, 161, 214, 0.35);
  background: rgba(255, 255, 255, 0.78);
  color: var(--text);
  box-shadow: 0 12px 26px -18px rgba(67, 108, 186, 0.4);
}

.sort-controls {
  display: flex;
  gap: 0.75rem;
  font-size: 0.8rem;
  color: var(--muted);
}

.sort-controls label {
  display: flex;
  gap: 0.4rem;
  align-items: center;
}

.sort-controls select {
  padding: 0.35rem 0.6rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: rgba(255, 255, 255, 0.85);
  color: var(--text);
  box-shadow: 0 12px 20px -16px rgba(67, 108, 186, 0.4);
}

.symbol-grid {
  display: grid;
  gap: 0.8rem;
  grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
}

.symbol-card {
  border: 1px solid rgba(129, 161, 214, 0.28);
  border-radius: 18px;
  padding: 0.95rem;
  cursor: pointer;
  transition: transform 0.28s ease, border-color 0.25s ease, box-shadow 0.28s ease;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(235, 243, 255, 0.94));
  box-shadow: 0 24px 44px -32px rgba(67, 108, 186, 0.38);
}

.symbol-card:hover {
  transform: translateY(-6px);
  border-color: rgba(36, 107, 255, 0.42);
  box-shadow: 0 32px 56px -30px rgba(36, 107, 255, 0.42);
}

.symbol-card.selected {
  transform: translateY(-4px);
  border-color: rgba(36, 107, 255, 0.48);
  box-shadow: 0 28px 52px -32px rgba(36, 107, 255, 0.45);
}

.symbol-card.decision-on {
  border-color: rgba(34, 197, 94, 0.36);
}

.symbol-card__top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}

.symbol-card__top h2 {
  margin: 0;
  font-size: 1.1rem;
}

.symbol-card__top time {
  font-size: 0.72rem;
  color: var(--muted);
}

.symbol-card__metrics {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.6rem;
  margin: 0.8rem 0 0.6rem;
}

.symbol-card__metrics .label {
  display: block;
  color: var(--muted);
  font-size: 0.75rem;
}

.symbol-card__metrics strong {
  display: block;
  font-size: 0.95rem;
}

.symbol-card__spark {
  height: 46px;
}

.symbol-card__spark svg {
  width: 100%;
  height: 100%;
  stroke: var(--accent);
  stroke-width: 1.8;
  fill: none;
  opacity: 0.9;
}

.symbol-card__badges {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
  margin-top: 0.6rem;
}

.badge {
  display: inline-flex;
  padding: 0.25rem 0.5rem;
  border-radius: 9999px;
  font-size: 0.68rem;
  letter-spacing: 0.4px;
  text-transform: uppercase;
  background: rgba(148, 163, 184, 0.15);
  color: var(--muted);
}

.badge.fresh {
  background: rgba(34, 197, 94, 0.2);
  color: var(--ok);
}

.badge.cold {
  background: rgba(248, 113, 113, 0.18);
  color: var(--bad);
}

.badge.no-gaps {
  background: rgba(34, 197, 94, 0.16);
  color: var(--ok);
}

.badge.minor-gaps {
  background: rgba(250, 204, 21, 0.2);
  color: var(--warn);
}

.badge.gaps {
  background: rgba(248, 113, 113, 0.16);
  color: var(--bad);
}

.badge.tf-ok {
  background: rgba(96, 165, 250, 0.2);
  color: #bfdbfe;
}

.badge.tf-partial {
  background: rgba(129, 140, 248, 0.22);
  color: #c7d2fe;
}


.badge.tf-missing {
  background: rgba(148, 163, 184, 0.28);
  color: var(--muted);
}

.badge.strength-high {
  background: rgba(16, 185, 129, 0.24);
  color: #6ee7b7;
}

.badge.strength-mid {
  background: rgba(250, 204, 21, 0.25);
  color: #fde68a;
}

.badge.strength-low {
  background: rgba(148, 163, 184, 0.24);
  color: var(--muted);
}

.detail-panel {
  border: 1px solid rgba(129, 161, 214, 0.32);
  border-radius: 24px;
  background: var(--card);
  padding: 1.6rem 1.5rem;
  display: grid;
  gap: 1.25rem;
  box-shadow: 0 36px 68px -38px rgba(67, 108, 186, 0.35);
  backdrop-filter: saturate(150%) blur(18px);
}

.detail-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
}

.detail-header h2 {
  margin: 0;
  font-size: 1.2rem;
}

.pill-group {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}

.pill {
  padding: 0.35rem 0.7rem;
  border-radius: 999px;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.4px;
}

.pill.on {
  background: rgba(34, 197, 94, 0.25);
  color: var(--ok);
}

.pill.off {
  background: rgba(148, 163, 184, 0.2);
  color: var(--muted);
}

.pill.muted {
  background: rgba(255, 255, 255, 0.6);
  color: var(--muted);
}

.detail-grid {
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
}

.detail-card {
  border: 1px solid rgba(129, 161, 214, 0.28);
  border-radius: 18px;
  background: linear-gradient(185deg, rgba(255, 255, 255, 0.92), rgba(233, 241, 255, 0.94));
  padding: 1rem 1.2rem;
  display: grid;
  gap: 0.85rem;
  box-shadow: 0 18px 36px -26px rgba(67, 108, 186, 0.28);
}

.detail-card header {
  font-size: 0.85rem;
  letter-spacing: 0.6px;
  text-transform: uppercase;
  color: var(--muted);
}

.prob-bars {
  display: grid;
  gap: 0.6rem;
}

.bar-row {
  display: grid;
  grid-template-columns: 60px 1fr auto;
  gap: 0.75rem;
  align-items: center;
  font-size: 0.8rem;
}

.progress {
  position: relative;
  height: 8px;
  background: rgba(129, 161, 214, 0.18);
  border-radius: 999px;
  overflow: hidden;
}

.progress span {
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, rgba(96, 165, 250, 0.2), rgba(96, 165, 250, 0.7));
}

.progress span.smooth {
  background: linear-gradient(90deg, rgba(129, 140, 248, 0.25), rgba(129, 140, 248, 0.7));
}

.progress span.final {
  background: linear-gradient(90deg, rgba(14, 165, 233, 0.3), rgba(14, 165, 233, 0.8));
}

.progress span.threshold {
  background: linear-gradient(90deg, rgba(245, 158, 11, 0.3), rgba(245, 158, 11, 0.7));
}

.progress.compact {
  height: 6px;
}

.list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.55rem;
  font-size: 0.82rem;
}

.list li {
  display: grid;
  grid-template-columns: 1fr minmax(0, 120px) auto;
  gap: 0.6rem;
  align-items: center;
}

.components li {
  grid-template-columns: 1fr auto;
}

.quality {
  display: grid;
  gap: 0.4rem;
  font-size: 0.82rem;
}

.quality .label {
  color: var(--muted);
  margin-right: 0.6rem;
}

.empty-note {
  margin: 0;
}

.trades-pane {
  margin-top: 2.4rem;
  border: 1px solid rgba(129, 161, 214, 0.3);
  border-radius: 26px;
  background: linear-gradient(185deg, rgba(255, 255, 255, 0.92), rgba(235, 243, 255, 0.95));
  padding: 1.7rem 1.5rem;
  box-shadow: 0 44px 80px -46px rgba(67, 108, 186, 0.38);
  backdrop-filter: saturate(150%) blur(18px);
  position: relative;
  overflow: hidden;
}

.trades-pane::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 85% 15%, rgba(173, 205, 255, 0.45), transparent 62%);
  opacity: 0.45;
}

.trades-pane > * {
  position: relative;
  z-index: 1;
}

.pane-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.trades-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.78rem;
}

.trades-table th,
.trades-table td {
  padding: 0.55rem 0.6rem;
  border-bottom: 1px solid rgba(148, 163, 184, 0.18);
  text-align: right;
}

.trades-table th:first-child,
.trades-table td:first-child,
.trades-table th:nth-child(2),
.trades-table td:nth-child(2) {
  text-align: left;
}

.id-cell {
  display: flex;
  gap: 0.45rem;
  align-items: center;
}

.status-badge {
  display: inline-flex;
  padding: 0.25rem 0.55rem;
  border-radius: 999px;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  font-size: 0.68rem;
}

.status-badge.open {
  background: rgba(96, 165, 250, 0.22);
  color: #bfdbfe;
}

.status-badge.closed {
  background: rgba(34, 197, 94, 0.18);
  color: #4ade80;
}

.status-badge.error {
  background: rgba(248, 113, 113, 0.2);
  color: #fda4af;
}

.fills-row td {
  background: rgba(255, 255, 255, 0.88);
}

.fills-wrap {
  padding: 0.75rem 0.5rem 0.2rem;
}

.fills-meta {
  display: flex;
  gap: 1rem;
  font-size: 0.8rem;
  color: var(--muted);
  margin-bottom: 0.5rem;
}

.fills-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.78rem;
}

.fills-table th,
.fills-table td {
  text-align: left;
  padding: 0.35rem 0.5rem;
  border-bottom: 1px dashed rgba(148, 163, 184, 0.25);
}

.fills-table td.num {
  text-align: right;
}

.empty-state {
  margin-top: 4rem;
  text-align: center;
  color: var(--muted);
}

.btn {
  padding: 0.55rem 0.95rem;
  border-radius: 999px;
  border: 1px solid rgba(125, 211, 252, 0.4);
  background: linear-gradient(90deg, rgba(59, 130, 246, 0.25), rgba(56, 189, 248, 0.3));
  color: #e0f2fe;
  font-weight: 600;
  cursor: pointer;
  box-shadow: 0 14px 28px -18px rgba(125, 211, 252, 0.9);
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn.ghost {
  background: transparent;
  border-color: rgba(148, 163, 184, 0.25);
  color: var(--muted);
}

.link {
  border: none;
  background: none;
  color: var(--accent);
  cursor: pointer;
  font-size: 0.72rem;
  padding: 0;
}

.pos {
  color: var(--ok);
}

.neg {
  color: var(--bad);
}

.symbol-card__metrics strong.high {
  color: var(--bad);
}

.symbol-card__metrics strong.mid {
  color: var(--warn);
}

.symbol-card__metrics strong.low {
  color: var(--accent);
}

.conf-high {
  color: var(--ok);
}

.conf-mid {
  color: var(--warn);
}

.conf-low {
  color: var(--muted);
}

@media (max-width: 1024px) {
  .main-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 720px) {
  .masthead__status {
    width: 100%;
    flex-wrap: wrap;
    justify-content: flex-start;
  }

  .api-input {
    width: 100%;
  }

  .sort-controls {
    width: 100%;
    justify-content: flex-start;
  }
}
</style>
