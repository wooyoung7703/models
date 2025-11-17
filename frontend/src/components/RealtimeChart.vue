<template>
  <section class="realtime-chart">
    <div ref="containerEl" class="chart-container"></div>
    <div class="chart-footer">
      <span class="muted">WS {{ wsConnected ? '온라인' : '폴백' }}</span>
      <span class="muted">실시간 추적 {{ followRealtime ? 'ON' : 'OFF' }}</span>
      <button class="btn ghost" @click="reload" :disabled="initialLoading">새로고침</button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, watch, computed, type PropType } from 'vue'
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type HistogramData,
  ColorType,
  CrosshairMode
} from 'lightweight-charts'

interface SignalMarker {
  id: string | number
  ts: string
  side: 'buy' | 'sell'
  price?: number
  label?: string
}

const props = defineProps({
  symbol: { type: String, required: true },
  apiBase: { type: String, required: true },
  interval: { type: String, default: '1m' },
  signals: { type: Array as PropType<SignalMarker[]>, default: () => [] },
  liveNowcast: { type: Object as PropType<any>, default: null },
  wsConnected: { type: Boolean, default: false }
})

const containerEl = ref<HTMLDivElement | null>(null)
let chart: IChartApi | null = null
let candleSeries: ISeriesApi<'Candlestick'> | null = null
let volumeSeries: ISeriesApi<'Histogram'> | null = null
let resizeObserver: ResizeObserver | null = null

const candles = ref<Array<CandlestickData & { volume?: number }>>([])
const earliestTime = ref<number | null>(null)
const initialLoading = ref(false)
const loadingOlder = ref(false)
const followRealtime = ref(true)
const intervalSeconds = computed(() => parseInterval(props.interval))

const createBaseChart = () => {
  if (!containerEl.value) return
  chart = createChart(containerEl.value, {
    layout: { background: { type: ColorType.Solid, color: 'rgba(6,11,23,0.9)' }, textColor: '#cbd5f5' },
    grid: { vertLines: { color: 'rgba(70,86,121,0.4)' }, horzLines: { color: 'rgba(70,86,121,0.4)' } },
    crosshair: { mode: CrosshairMode.Normal },
    width: containerEl.value.clientWidth,
    height: 420
  })
  candleSeries = chart.addCandlestickSeries({
    upColor: '#4ade80',
    downColor: '#f87171',
    wickUpColor: '#4ade80',
    wickDownColor: '#f87171',
    borderUpColor: '#4ade80',
    borderDownColor: '#f87171'
  })
  volumeSeries = chart.addHistogramSeries({
    color: '#a5b4fc',
    priceFormat: { type: 'volume' },
    priceScaleId: ''
  })
  volumeSeries?.priceScale().applyOptions({ scaleMargins: { top: 0.75, bottom: 0 } })
  chart.timeScale().subscribeVisibleTimeRangeChange(handleVisibleRange)
}

const disposeChart = () => {
  if (chart) {
    chart.timeScale().unsubscribeVisibleTimeRangeChange(handleVisibleRange)
    chart.remove()
  }
  chart = null
  candleSeries = null
  volumeSeries = null
}

const loadInitial = async () => {
  if (!props.apiBase || !props.symbol) return
  initialLoading.value = true
  const data = await fetchCandles()
  candles.value = data
  earliestTime.value = data[0]?.time as number | null
  candleSeries?.setData(candles.value)
  volumeSeries?.setData(toVolumeSeries(candles.value))
  chart?.timeScale().scrollToRealTime()
  syncMarkers()
  initialLoading.value = false
}

const reload = async () => {
  earliestTime.value = null
  candles.value = []
  candleSeries?.setData([])
  volumeSeries?.setData([])
  await loadInitial()
}

const fetchCandles = async (params: { end?: number } = {}) => {
  const url = new URL(`${props.apiBase}/chart/candles`)
  url.searchParams.set('symbol', props.symbol)
  url.searchParams.set('interval', props.interval)
  url.searchParams.set('limit', '500')
  if (params.end) url.searchParams.set('end', String(params.end))
  const res = await fetch(url.toString())
  if (!res.ok) return []
  const payload = await res.json()
  return (Array.isArray(payload) ? payload : []).map(toBar).sort((a, b) => (a.time as number) - (b.time as number))
}

const toBar = (row: any): CandlestickData & { volume?: number } => {
  const ts = Math.floor((row.open_time ? Date.parse(row.open_time) : row.time * 1000) / 1000)
  return {
    time: ts,
    open: Number(row.open),
    high: Number(row.high),
    low: Number(row.low),
    close: Number(row.close),
    volume: Number(row.volume ?? row.qty ?? 0)
  }
}

const toVolumeSeries = (bars: Array<CandlestickData & { volume?: number }>): HistogramData[] =>
  bars.map((bar) => ({
    time: bar.time,
    value: bar.volume ?? 0,
    color: (bar.close ?? 0) >= (bar.open ?? 0) ? '#4ade80' : '#f87171'
  }))

const handleVisibleRange = async (range: { from: number; to: number } | null) => {
  const latest = candles.value[candles.value.length - 1]
  if (latest && range?.to != null) {
    followRealtime.value = range.to >= (latest.time as number) - intervalSeconds.value
  } else {
    followRealtime.value = true
  }

  if (!range?.from || loadingOlder.value || !earliestTime.value) return
  const threshold = earliestTime.value + intervalSeconds.value * 2
  if (range.from > threshold) return
  loadingOlder.value = true
  const older = await fetchCandles({ end: earliestTime.value - intervalSeconds.value })
  if (older.length) {
    candles.value = [...older, ...candles.value]
    earliestTime.value = candles.value[0]?.time as number
    candleSeries?.setData(candles.value)
    volumeSeries?.setData(toVolumeSeries(candles.value))
    syncMarkers()
  }
  loadingOlder.value = false
}

const updateLiveBar = () => {
  const payload = props.liveNowcast
  if (!payload?.timestamp || payload.price == null) return
  const ts = Math.floor(Date.parse(payload.timestamp) / 1000)
  const last = candles.value[candles.value.length - 1]
  if (last && ts === last.time) {
    last.close = payload.price
    last.high = Math.max(last.high, payload.price)
    last.low = Math.min(last.low, payload.price)
    candleSeries?.update(last)
    volumeSeries?.update({
      time: last.time,
      value: last.volume ?? 0,
      color: last.close >= last.open ? '#4ade80' : '#f87171'
    })
  } else if (!last || ts > last.time) {
    const open = last ? last.close : payload.price
    const next = { time: ts, open, high: payload.price, low: payload.price, close: payload.price, volume: payload.volume ?? 0 }
    candles.value.push(next)
    candleSeries?.update(next)
    volumeSeries?.update({
      time: next.time,
      value: next.volume ?? 0,
      color: '#4ade80'
    })
  }
  if (followRealtime.value) chart?.timeScale().scrollToRealTime()
}

const syncMarkers = () => {
  if (!candleSeries) return
  const markers = props.signals.map((signal) => ({
    time: Math.floor(Date.parse(signal.ts) / 1000),
    position: signal.side === 'sell' ? 'aboveBar' : 'belowBar',
    color: signal.side === 'sell' ? '#f87171' : '#4ade80',
    shape: signal.side === 'sell' ? 'arrowDown' : 'arrowUp',
    text: signal.label || signal.side.toUpperCase()
  }))
  candleSeries.setMarkers(markers)
}

const parseInterval = (interval: string) => {
  if (!interval) return 60
  const m = interval.match(/^(\d+)([smhd])$/i)
  if (!m) return 60
  const value = Number(m[1])
  const unit = m[2].toLowerCase()
  if (unit === 's') return value
  if (unit === 'm') return value * 60
  if (unit === 'h') return value * 3600
  return value * 86400
}

onMounted(async () => {
  createBaseChart()
  if (containerEl.value) {
    resizeObserver = new ResizeObserver(() => {
      if (chart && containerEl.value) chart.resize(containerEl.value.clientWidth, 420)
    })
    resizeObserver.observe(containerEl.value)
  }
  await loadInitial()
})

onBeforeUnmount(() => {
  resizeObserver?.disconnect()
  disposeChart()
})

watch(
  () => [props.symbol, props.apiBase, props.interval],
  async () => {
    disposeChart()
    createBaseChart()
    await loadInitial()
  }
)

watch(
  () => props.liveNowcast?.timestamp,
  () => updateLiveBar()
)

watch(
  () => props.signals,
  () => syncMarkers(),
  { deep: true }
)
</script>

<style scoped>
.realtime-chart {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}
.chart-container {
  width: 100%;
  height: 420px;
}
.chart-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.82rem;
}
</style>
