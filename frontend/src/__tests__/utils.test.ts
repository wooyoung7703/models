import { describe, it, expect } from 'vitest'
import { formatPct, formatSignedPct, scoreClass, confidenceClass, marginClass } from '../utils'

describe('utils formatting', () => {
  it('formatPct works (Intl rounded)', () => {
    expect(formatPct(0.1234)).toBe('12.3%')
    expect(formatPct(undefined)).toBe('n/a')
  })
  it('formatSignedPct works (Intl)', () => {
    expect(formatSignedPct(0.05)).toBe('+5.0%')
    expect(formatSignedPct(-0.031)).toBe('-3.1%')
    expect(formatSignedPct(null)).toBe('n/a')
  })
  it('scoreClass buckets', () => {
    expect(scoreClass(0.8)).toBe('high')
    expect(scoreClass(0.5)).toBe('mid')
    expect(scoreClass(0.1)).toBe('low')
  })
  it('confidenceClass thresholds', () => {
    expect(confidenceClass(0.25)).toBe('conf-high')
    expect(confidenceClass(0.15)).toBe('conf-mid')
    expect(confidenceClass(0.02)).toBe('conf-low')
    expect(confidenceClass(undefined)).toBe('')
  })
  it('marginClass positive/negative', () => {
    expect(marginClass(0.001)).toBe('pos')
    expect(marginClass(-0.2)).toBe('neg')
    expect(marginClass(undefined)).toBe('')
  })
})
