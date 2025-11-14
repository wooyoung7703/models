// Consistent, locale-stable percent formatting using Intl.
// We pin to 'en-US' to avoid snapshot diffs across environments.
const pctFmt = new Intl.NumberFormat('en-US', {
  style: 'percent',
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
})

export const formatPct = (p?: number) => {
  if (p == null || isNaN(p)) return 'n/a'
  try {
    return pctFmt.format(p)
  } catch {
    return `${(p * 100).toFixed(1)}%`
  }
}

export const formatSignedPct = (p?: number | null) => {
  if (p == null || isNaN(Number(p))) return 'n/a'
  const v = Number(p)
  const sign = v > 0 ? '+' : v < 0 ? '-' : ''
  try {
    return `${sign}${pctFmt.format(Math.abs(v))}`
  } catch {
    return `${sign}${(Math.abs(v) * 100).toFixed(1)}%`
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
