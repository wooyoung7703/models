// Consistent, locale-stable percent formatting using Intl.
// We pin to 'en-US' to avoid snapshot diffs across environments.
const pctFmt = new Intl.NumberFormat('en-US', {
  style: 'percent',
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
})

export const formatPct = (p?: number) => {
  if (p == null || isNaN(p)) return 'n/a'
  const v = p
  // Extremely small but positive values: show a floor indicator
  if (v > 0 && v < 0.001) return '<0.1%'
  // Sub-1% values: increase precision to two decimals
  if (v < 0.01) return `${(v * 100).toFixed(2)}%`
  try {
    return pctFmt.format(v)
  } catch {
    return `${(v * 100).toFixed(1)}%`
  }
}

export const formatSignedPct = (p?: number | null) => {
  if (p == null || isNaN(Number(p))) return 'n/a'
  const raw = Number(p)
  const sign = raw > 0 ? '+' : raw < 0 ? '-' : ''
  const v = Math.abs(raw)
  if (v > 0 && v < 0.001) return `${sign}<0.1%`
  if (v < 0.01) return `${sign}${(v * 100).toFixed(2)}%`
  try {
    return `${sign}${pctFmt.format(v)}`
  } catch {
    return `${sign}${(v * 100).toFixed(1)}%`
  }
}

export const scoreClass = (s: number) => {
  if (s >= 0.66) return 'high'
  if (s >= 0.33) return 'mid'
  return 'low'
}

export const confidenceClass = (c?: number | null) => {
  if (c == null || isNaN(Number(c))) return ''
  const v = Number(c)
  if (v >= 0.2) return 'conf-high'
  if (v >= 0.1) return 'conf-mid'
  return 'conf-low'
}

export const marginClass = (m?: number | null) => {
  if (m == null || isNaN(Number(m))) return ''
  return Number(m) >= 0 ? 'pos' : 'neg'
}
