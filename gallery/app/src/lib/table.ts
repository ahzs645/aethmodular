/**
 * Long ("tidy") tables for the spec-driven figures (components/Figure): one
 * row per point, one column per variable. Exports may store a table as
 * columns ({col: [values]}) or as rows ([{col: value}]); `rowsOf` accepts
 * both. The summaries here are the few every meeting figure needs: a median
 * with an interval over repeated splits, and histogram densities.
 */
export type Value = string | number | boolean | null
export type Row = Record<string, Value>
export type Table = Row[] | Record<string, Value[]>

export function rowsOf(t: Table | null | undefined): Row[] {
  if (!t) return []
  if (Array.isArray(t)) return t
  const cols = Object.keys(t)
  const n = cols.length ? t[cols[0]].length : 0
  const out: Row[] = new Array(n)
  for (let i = 0; i < n; i++) { const r: Row = {}; for (const c of cols) r[c] = t[c][i]; out[i] = r }
  return out
}

/** Keep rows whose columns equal the given value (or one of the values in a list). */
export function where(rows: Row[], cond: Record<string, Value | Value[]> | undefined): Row[] {
  if (!cond) return rows
  const tests = Object.entries(cond).map(([c, v]) => Array.isArray(v) ? (r: Row) => v.includes(r[c]) : (r: Row) => r[c] === v)
  return rows.filter((r) => tests.every((t) => t(r)))
}

/** Distinct values of a column, in first-seen order. */
export const uniq = (rows: Row[], col: string): Value[] => [...new Set(rows.map((r) => r[col]))]

export const num = (v: Value | undefined): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

/** a summarised group: its `by` columns plus the statistic, interval and count */
export type Summary = Row & { mid: number | null; lo: number | null; hi: number | null; n: number }
export type Interval = 'iqr' | 'p10-p90' | 'p5-p95' | 'minmax' | 'none'
const BOUNDS: Record<Exclude<Interval, 'none'>, [number, number]> = { iqr: [0.25, 0.75], 'p10-p90': [0.1, 0.9], 'p5-p95': [0.05, 0.95], minmax: [0, 1] }
export const INTERVAL_LABEL: Record<Interval, string> = { iqr: 'interquartile range', 'p10-p90': '10–90 %', 'p5-p95': '5–95 %', minmax: 'full range', none: '' }

function q(sorted: number[], p: number) {
  const i = (sorted.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i)
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo)
}

/**
 * Median (or mean) of `value` per combination of the `by` columns, with an
 * interval, e.g. the median Addis intercept over 100 splits per season × baseline.
 * Groups keep first-seen order.
 */
export function summarize(rows: Row[], by: string[], value: string, { stat = 'median', interval = 'iqr' }: { stat?: 'median' | 'mean'; interval?: Interval } = {}) {
  const groups = new Map<string, { key: Row; vals: number[] }>()
  for (const r of rows) {
    const k = JSON.stringify(by.map((c) => r[c]))
    let g = groups.get(k)
    if (!g) { g = { key: Object.fromEntries(by.map((c) => [c, r[c]])), vals: [] }; groups.set(k, g) }
    const v = num(r[value])
    if (v != null) g.vals.push(v)
  }
  return [...groups.values()].map(({ key, vals }) => {
    const s = vals.sort((a, b) => a - b)
    const mid = !s.length ? null : stat === 'mean' ? s.reduce((a, b) => a + b, 0) / s.length : q(s, 0.5)
    const [plo, phi] = interval === 'none' ? [NaN, NaN] : BOUNDS[interval]
    return { ...key, mid, lo: s.length && interval !== 'none' ? q(s, plo) : null, hi: s.length && interval !== 'none' ? q(s, phi) : null, n: s.length } as Summary
  })
}

/**
 * Histogram densities on shared bin edges (log-spaced with `log`), so several
 * populations compare on one axis whatever their sizes: each series integrates to 1.
 */
export function densities(series: { key: string; values: number[] }[], { bins = 40, log = false, tail = 0 }: {
  bins?: number; log?: boolean
  /** set the range from each series' tail quantiles instead of its extremes; values outside fall in the end bins */
  tail?: number
} = {}) {
  const usable = series.map((s) => s.values.filter((v) => Number.isFinite(v) && (!log || v > 0)).sort((a, b) => a - b))
  if (!usable.some((u) => u.length)) return { edges: [] as number[], series: series.map((s) => ({ key: s.key, centers: [] as number[], density: [] as number[], median: null as number | null })) }
  let lo = Math.min(...usable.filter((u) => u.length).map((u) => q(u, tail))), hi = Math.max(...usable.filter((u) => u.length).map((u) => q(u, 1 - tail)))
  if (lo === hi) { lo = log ? lo / 2 : lo - 0.5; hi = log ? hi * 2 : hi + 0.5 }
  const t = (v: number) => (log ? Math.log10(v) : v), inv = (v: number) => (log ? 10 ** v : v)
  const step = (t(hi) - t(lo)) / bins
  const edges = Array.from({ length: bins + 1 }, (_, i) => inv(t(lo) + i * step))
  return {
    edges,
    series: series.map((s) => {
      const counts = new Array(bins).fill(0)
      const vals = s.values.filter((v) => Number.isFinite(v) && (!log || v > 0))
      for (const v of vals) counts[Math.min(bins - 1, Math.max(0, Math.floor((t(v) - t(lo)) / step)))]++
      // the median of every finite value, not just those a log axis can show, so it matches the tables
      const sorted = s.values.filter(Number.isFinite).sort((a, b) => a - b)
      return {
        key: s.key,
        centers: counts.map((_, i) => inv(t(lo) + (i + 0.5) * step)),
        // density per unit of the (log) axis, so the area under each curve is 1
        density: counts.map((c) => (vals.length ? c / vals.length / step : 0)),
        median: sorted.length ? q(sorted, 0.5) : null,
      }
    }),
  }
}
