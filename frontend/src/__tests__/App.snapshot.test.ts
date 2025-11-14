import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import App from '../App.vue'
import { nextTick } from 'vue'

// Fixed timestamps in mock payloads for stable rendering
const FIXED_TS = 1704067200000 // 2024-01-01T00:00:00Z

// Mock fetch for /nowcast and /health/features
const mockNowcast = {
  _stacking_meta: { method: 'logistic', threshold: 0.9, models: ['lstm','tf','xgb'] },
  BTCUSDT: {
    symbol: 'BTCUSDT', price: 50000, price_source: 'live', bottom_score: 0.42,
    stacking: { prob: 0.38, threshold: 0.9, threshold_source: 'sidecar', decision: false, above_threshold: false, margin: -0.52, confidence: 0.52 },
    base_probs: { lstm: 0.33, tf: 0.41, xgb: 0.55 },
    components: {}, base_info: {}, timestamp: FIXED_TS
  },
  ETHUSDT: {
    symbol: 'ETHUSDT', price: 3500, price_source: 'live', bottom_score: 0.75,
    stacking: { prob: 0.93, threshold: 0.9, threshold_source: 'sidecar', decision: true, above_threshold: true, margin: 0.03, confidence: 0.03 },
    base_probs: { lstm: 0.65, tf: 0.71, xgb: 0.69 },
    components: {}, base_info: {}, timestamp: FIXED_TS
  }
}
const mockFeaturesHealth = {
  data_fresh_seconds: 8,
  missing_minutes_24h: 0,
  latest_5m_open_time: FIXED_TS/1000,
  latest_15m_open_time: FIXED_TS/1000,
}

const fetchMock: (url: string) => Promise<any> = vi.fn((url: string) => {
  if (url.endsWith('/nowcast')) {
    return Promise.resolve({ ok: true, json: () => Promise.resolve(mockNowcast) })
  }
  if (url.endsWith('/health/features')) {
    return Promise.resolve({ ok: true, json: () => Promise.resolve(mockFeaturesHealth) })
  }
  return Promise.reject(new Error('unknown url ' + url))
}) as any
// Stub fetch and timers to avoid background polling during tests
(globalThis as any).fetch = fetchMock
const realSetInterval = window.setInterval
const realClearInterval = window.clearInterval
// no-op intervals
window.setInterval = ((cb: any, _ms?: number) => 0) as any
window.clearInterval = (() => {}) as any

// Provide API base without clobbering jsdom window
Object.defineProperty(window as any, 'VITE_API_BASE', { value: 'http://localhost:8000', configurable: true })

describe('App.vue snapshot', () => {
  it('renders stacking meta and symbols', async () => {
    const wrapper = mount(App)
    // explicitly trigger fetch to ensure data populated
    await (wrapper.vm as any).manualRefresh()
    await nextTick(); await nextTick()
    const text = wrapper.text()
    expect(text).toContain('logistic')
    expect(text).toContain('BTCUSDT')
    expect(text).toContain('ETHUSDT')
    expect(text).toContain('임계값')
  })
})
