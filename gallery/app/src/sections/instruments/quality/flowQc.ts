/**
 * The maths of notebooks/qc/flow_qc_analysis that the Data quality page redoes in the browser: pandas' centred
 * rolling windows over the hourly rows, numpy's linear quantiles, matplotlib's Tukey box, scipy's Gaussian KDE (on
 * the exported histogram), the flow-ratio pass/warn/fail classes, and the all-sites figures' daily records (a
 * union of two records, lines that break across long gaps). No imports, so node tests load it directly.
 */

const ok = (v: number | null | undefined): v is number => v != null && Number.isFinite(v)

/** numpy.quantile (linear) of the finite values; null when there are none */
export function quantileLinear(values: readonly (number | null | undefined)[], q: number): number | null {
  const v = values.filter(ok).sort((a, b) => a - b)
  if (!v.length) return null
  const h = (v.length - 1) * q, lo = Math.floor(h)
  return lo + 1 < v.length ? v[lo] + (h - lo) * (v[lo + 1] - v[lo]) : v[lo]
}

/**
 * pandas Series.rolling(window, center=True) over the rows as they stand (missing hours are not rows), with its
 * default min_periods = window: row i reads rows i − ⌊w/2⌋ … i − ⌊w/2⌋ + w − 1 and is null unless all w are present
 * and finite (pandas counts NaN rows towards the window but not towards min_periods).
 */
export function rollingCentred(values: readonly (number | null)[], window: number, stat: 'median' | 'mean'): (number | null)[] {
  const half = Math.floor(window / 2)
  return values.map((_, i) => {
    const a = i - half, b = a + window
    if (a < 0 || b > values.length) return null
    const xs: number[] = []
    for (let j = a; j < b; j++) if (ok(values[j])) xs.push(values[j]!)
    if (xs.length < window) return null
    if (stat === 'mean') return xs.reduce((s, x) => s + x, 0) / xs.length
    xs.sort((p, q) => p - q)
    return window % 2 ? xs[half] : (xs[half - 1] + xs[half]) / 2
  })
}

/** the hourly table of <CODE>.flow.json: runs of consecutive hours from `start` */
export interface HourlyRuns { start: string; runs: [number, number][] }
/** each stored hour's local start as a UTC date of the local clock (as data.ts localDate), in order */
export function expandHours(h: HourlyRuns): Date[] {
  const t0 = Date.parse(`${h.start.slice(0, 10)}T${h.start.slice(11, 16)}:00Z`)
  const out: Date[] = []
  for (const [off, len] of h.runs) for (let k = 0; k < len; k++) out.push(new Date(t0 + (off + k) * 3600e3))
  return out
}

/** flow_qc_analysis's ratio flags: pass within 5 % of the target, warn within 10 %, else fail */
export type RatioClass = 'pass' | 'warn' | 'fail'
export function ratioClass(r: number, target: number): RatioClass {
  const dev = Math.abs(r - target)
  return dev <= 0.05 * target ? 'pass' : dev <= 0.1 * target ? 'warn' : 'fail'
}

/** matplotlib's box (whis = 1.5): numpy quartiles, whiskers at the furthest values within 1.5 IQR of the box */
export interface TukeyBox { n: number; q25: number; median: number; q75: number; lo: number; hi: number; min: number; max: number; mean: number; below: number; above: number }
export function tukeyBox(values: readonly (number | null | undefined)[]): TukeyBox | null {
  const v = values.filter(ok).sort((a, b) => a - b)
  if (!v.length) return null
  const q25 = quantileLinear(v, 0.25)!, q75 = quantileLinear(v, 0.75)!, iqr = q75 - q25
  const inside = v.filter((x) => x >= q25 - 1.5 * iqr && x <= q75 + 1.5 * iqr)
  return {
    n: v.length, q25, median: quantileLinear(v, 0.5)!, q75, lo: inside[0], hi: inside[inside.length - 1], min: v[0], max: v[v.length - 1],
    mean: v.reduce((s, x) => s + x, 0) / v.length, below: v.filter((x) => x < q25 - 1.5 * iqr).length, above: v.filter((x) => x > q75 + 1.5 * iqr).length,
  }
}

/**
 * scipy.stats.gaussian_kde with Scott's bandwidth (σ·n^−1/5), on a histogram instead of the raw values: each bin's
 * count sits at its centre. The values outside [lo, hi] (whose positions the histogram does not keep) count towards
 * n but add no kernel, so the curve is right inside the range and integrates to the share inside it. With bins
 * narrower than the bandwidth this is the notebook's curve to within the bin width. Returns density at `at`.
 */
export function kdeFromHist(h: { lo: number; hi: number; counts: number[]; below: number; above: number }, sd: number, at: readonly number[]): number[] {
  const nb = h.counts.length, w = (h.hi - h.lo) / nb
  const counts = h.counts
  const n = counts.reduce((s, c) => s + c, 0) + h.below + h.above
  if (!n || !(sd > 0)) return at.map(() => 0)
  const bw = sd * n ** -0.2, k = 1 / (n * bw * Math.sqrt(2 * Math.PI))
  return at.map((x) => {
    let s = 0
    for (let i = 0; i < nb; i++) if (counts[i]) { const z = (x - (h.lo + (i + 0.5) * w)) / bw; s += counts[i] * Math.exp(-0.5 * z * z) }
    return s * k
  })
}

/** one record's daily values by ISO date */
export type DayValues = Map<string, number>
/** the union of two records of one instrument, `primary`'s value wherever both have the day; sorted by day */
export function unionDays(primary: DayValues, secondary: DayValues): DayValues {
  const out = new Map(secondary)
  for (const [d, v] of primary) out.set(d, v)
  return new Map([...out].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)))
}

/** points in date order with a null inserted wherever consecutive days are more than `maxGap` days apart */
export function breakGaps(days: DayValues, maxGap: number): { x: Date; y: number | null }[] {
  const out: { x: Date; y: number | null }[] = []
  let prev: number | null = null
  for (const [d, v] of days) {
    const t = Date.parse(`${d}T00:00:00Z`)
    if (prev != null && t - prev > maxGap * 864e5) out.push({ x: new Date(prev + 864e5), y: null })
    out.push({ x: new Date(t), y: v })
    prev = t
  }
  return out
}

/** day-by-day points with a null inserted wherever consecutive days are more than one day apart */
export function dailyGaps<T extends { x: Date; y: number | null }>(points: T[]): (T | { x: Date; y: null })[] {
  const out: (T | { x: Date; y: null })[] = []
  points.forEach((p, i) => {
    if (i > 0 && +p.x - +points[i - 1].x > 1.5 * 864e5) out.push({ x: new Date(+points[i - 1].x + 864e5), y: null })
    out.push(p)
  })
  return out
}
