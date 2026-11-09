/**
 * The gallery's one statistics library: every mean, median, quantile, correlation, fit and test a chart or
 * page computes comes from here (AGENTS.md, "extend the shared pieces"). Inputs are null-tolerant: a value
 * that is not a finite number (null, undefined, NaN, ±Infinity) is skipped, and paired inputs keep only the
 * pairs where both are finite.
 *
 * - describe: `mean`, `sd`, `median`, `quantile`, `quantileSorted`, `meanSe`, `describe`, `boxStats`,
 *   `monthlyStats`, `monthlyMeans`, `kde`, `jitter`
 * - correlate: `pearson`, `spearman` (both with a `minN`), `ranks`, `corrMatrix`, `correlationP`, `steiger`
 * - fit: `regression` (OLS + optional Deming and York, the crossplots' fit), `ols` (scipy linregress with
 *   standard errors), `deming`, `york`, `theilSen`, `originFit`, `multiOls`, `quadratic`
 * - test: `twoSample`, `normalCdf`, `incompleteBeta`
 * - screen: `splitSweep`, `bestLowThreshold`, `splitScan`, `slidingWindow`, `influence`
 *
 * The research notebooks' numbers are the reference (node-tested against scipy / numpy in
 * src/lib/researchStats.test.mjs): OLS is scipy.stats.linregress, quantiles are numpy's linear default,
 * `median` is numpy.median, Theil–Sen's intercept is scipy's (median y − slope · median x), the through-origin
 * R² is uncentred. York follows the uncorrelated-error implementation in calibration_explorer/hips_lab.py.
 * Display preferences never refit EC calibrations or change their saved scores.
 */

/** a value that may be missing: only finite numbers count */
export type Num = number | null | undefined

/** a finite number (the guard every null-tolerant statistic here uses) */
export const isFiniteNumber = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

/** a statistic's NaN (no values to take it over) as null, for callers that draw "n/a" from null */
export const orNull = (v: number): number | null => (Number.isNaN(v) ? null : v)

/** the finite values, in order */
export const finite = (values: Iterable<unknown>): number[] => {
  const out: number[] = []
  for (const v of values) if (isFiniteNumber(v)) out.push(v)
  return out
}

/** the pairs where both values are finite, over the shorter of the two arrays */
export function finitePairs(xs: readonly Num[], ys: readonly Num[]): { x: number[]; y: number[] } {
  const x: number[] = [], y: number[] = []
  for (let i = 0; i < Math.min(xs.length, ys.length); i++) {
    const a = xs[i], b = ys[i]
    if (isFiniteNumber(a) && isFiniteNumber(b)) { x.push(a); y.push(b) }
  }
  return { x, y }
}

/** arithmetic mean of the finite values; NaN when there are none */
export function mean(values: readonly Num[]): number {
  let s = 0, n = 0
  for (const v of values) if (isFiniteNumber(v)) { s += v; n++ }
  return n ? s / n : NaN
}

/** standard deviation of the finite values with `ddof` (1: the sample SD, pandas' and d3.deviation's); NaN below ddof + 1 values */
export function sd(values: readonly Num[], ddof = 1): number {
  const v = finite(values)
  if (v.length <= ddof) return NaN
  const m = v.reduce((a, b) => a + b, 0) / v.length
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - ddof))
}

/**
 * The p-quantile of values already in ascending order, linear between order statistics (numpy's and pandas'
 * default, d3.quantileSorted); NaN when empty. A typed array works too (it sorts numerically and fast).
 */
export function quantileSorted(sorted: ArrayLike<number>, p: number): number {
  const n = sorted.length
  if (!n) return NaN
  const h = (n - 1) * p, lo = Math.floor(h)
  return lo + 1 < n ? sorted[lo] + (sorted[lo + 1] - sorted[lo]) * (h - lo) : sorted[lo]
}

/** numpy's default (linear) quantile of the finite values, unsorted; NaN when there are none */
export function quantile(values: readonly Num[], p: number): number {
  return quantileSorted(finite(values).sort((a, b) => a - b), p)
}

/** Right-continuous empirical CDF; tied observations contribute one combined jump. */
export function ecdf(values: readonly Num[], counts?: readonly Num[]): { x: number; y: number; n: number }[] {
  if (counts && counts.length !== values.length) throw new Error('ECDF values and counts must align')
  const sorted = values.flatMap((x, i) => {
    const count = counts ? counts[i] : 1
    return x != null && Number.isFinite(x) && count != null && Number.isFinite(count) && count > 0
      ? [{ x, count }] : []
  }).sort((a, b) => a.x - b.x)
  const n = sorted.reduce((sum, row) => sum + row.count, 0)
  const out: { x: number; y: number; n: number }[] = []
  let cumulative = 0
  for (let i = 0; i < sorted.length; i++) {
    cumulative += sorted[i].count
    if (i + 1 === sorted.length || sorted[i + 1].x !== sorted[i].x) out.push({ x: sorted[i].x, y: cumulative / n, n })
  }
  return out
}

/** the median of values already in ascending order: the middle one, or the mean of the middle two (numpy.median); NaN when empty */
export function medianSorted(sorted: ArrayLike<number>): number {
  const n = sorted.length, h = n >> 1
  return !n ? NaN : n % 2 ? sorted[h] : (sorted[h - 1] + sorted[h]) / 2
}

/** numpy.median of the finite values; NaN when there are none */
export function median(values: readonly Num[]): number {
  return medianSorted(finite(values).sort((a, b) => a - b))
}

/** mean and standard error of the finite values; null mean when there are none */
export function meanSe(values: readonly Num[]): { mean: number | null; se: number | null; n: number } {
  const v = finite(values)
  if (!v.length) return { mean: null, se: null, n: 0 }
  const m = v.reduce((a, b) => a + b, 0) / v.length
  if (v.length < 2) return { mean: m, se: null, n: 1 }
  const s = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1))
  return { mean: m, se: s / Math.sqrt(v.length), n: v.length }
}

/**
 * Percent effect and a normal interval from a log contrast: `effect = exp(beta) - 1`,
 * while `logSe` is the SE of beta. Transform the log endpoints before expressing them as
 * percentages; adding a log-scale SE directly to the fractional effect mixes scales.
 * The default z = 1.645 gives an approximate 90% interval. Saved rounded summaries yield
 * a reconstructed interval at that precision; this does not refit the underlying model.
 */
export function percentIntervalFromLogEffect(effect: Num, logSe: Num, z = 1.645): { v: number; lo: number; hi: number } | null {
  if (!isFiniteNumber(effect) || effect <= -1 || !isFiniteNumber(logSe) || logSe < 0 || !isFiniteNumber(z) || z <= 0) return null
  const beta = Math.log1p(effect), margin = z * logSe
  const v = effect * 100, lo = Math.expm1(beta - margin) * 100, hi = Math.expm1(beta + margin) * 100
  return [v, lo, hi].every(Number.isFinite) ? { v, lo, hi } : null
}

export interface Summary { n: number; mean: number; sd: number | null; median: number; q10: number; q25: number; q75: number; q90: number; min: number; max: number }

/** n, mean, sample SD and quantiles of the finite values; null when there are none */
export function describe(values: Iterable<Num>): Summary | null {
  const v = finite(values).sort((a, b) => a - b)
  if (!v.length) return null
  const m = v.reduce((s, x) => s + x, 0) / v.length
  const s = v.length > 1 ? Math.sqrt(v.reduce((a, x) => a + (x - m) ** 2, 0) / (v.length - 1)) : null
  const q = (p: number) => quantileSorted(v, p)
  return { n: v.length, mean: m, sd: s, median: q(0.5), q10: q(0.1), q25: q(0.25), q75: q(0.75), q90: q(0.9), min: v[0], max: v[v.length - 1] }
}

/** a group needs this many values before its interquartile range is drawn (monthly bands, the seasonal clock) */
export const IQR_MIN_N = 4

export interface MonthStats { month: number; n: number; median: number | null; mean: number | null; q1: number | null; q3: number | null }

/**
 * Per calendar month (1–12, all twelve returned): the count, median and mean of the finite values, and the
 * quartiles once a month has `iqrMinN` values (default IQR_MIN_N).
 */
export function monthlyStats<R>(rows: readonly R[], monthOf: (r: R) => number | null | undefined, valueOf: (r: R) => unknown, { iqrMinN = IQR_MIN_N }: { iqrMinN?: number } = {}): MonthStats[] {
  const by: number[][] = Array.from({ length: 12 }, () => [])
  for (const r of rows) {
    const m = monthOf(r), v = valueOf(r)
    if (m != null && Number.isInteger(m) && m >= 1 && m <= 12 && isFiniteNumber(v)) by[m - 1].push(v)
  }
  return by.map((vals, i) => {
    vals.sort((a, b) => a - b)
    const n = vals.length
    return {
      month: i + 1, n,
      median: n ? quantileSorted(vals, 0.5) : null,
      mean: n ? vals.reduce((a, b) => a + b, 0) / n : null,
      q1: n >= iqrMinN ? quantileSorted(vals, 0.25) : null,
      q3: n >= iqrMinN ? quantileSorted(vals, 0.75) : null,
    }
  })
}

/** month (1–12) → mean of the values on dates (YYYY-MM-…) in that month, with the count of days; null where none */
export function monthlyMeans(dates: readonly string[], values: readonly Num[]): { mean: number | null; n: number }[] {
  const sum = new Array(12).fill(0), n = new Array(12).fill(0)
  dates.forEach((d, i) => {
    const v = values[i]
    if (!isFiniteNumber(v)) return
    const m = Number(d.slice(5, 7)) - 1
    if (m >= 0 && m < 12) { sum[m] += v; n[m]++ }
  })
  return sum.map((s, m) => ({ mean: n[m] ? s / n[m] : null, n: n[m] }))
}

/** a deterministic pseudo-random number in [0, 1) from an integer seed (a sine hash); another `salt` gives another sequence */
export const hashUnit = (seed: number, salt = 12.9898) => {
  const t = Math.sin(seed * salt) * 43758.5453
  return t - Math.floor(t)
}

/** deterministic jitter in [−w/2, w/2) from an integer seed (hashUnit), so points do not dance on re-render */
export const jitter = (seed: number, w: number) => (hashUnit(seed) - 0.5) * w

/** deterministic, evenly spread jitter: the i-th point at the golden-ratio sequence (plus `offset`), in [−w/2, w/2) */
export const goldenJitter = (i: number, w: number, offset = 0) => (((i * 0.6180339 + offset) % 1) - 0.5) * w

export interface RegressionStats {
  n: number
  slope: number
  intercept: number
  r: number
  r2: number
  /** errors-in-variables slope; null when not requested */
  demingSlope: number | null
  demingIntercept: number | null
  demingLambda: number
  /** 100*(deming - ols)/ols — matches plotting.utils.calculate_regression_stats */
  slopeAttenuationPct: number | null
  rmse: number
  yorkSlope: number | null
  yorkIntercept: number | null
  yorkAssumption: string
}

export interface RegressionOptions { errorsInVariables?: boolean; lambda?: number; sigmaX?: number[]; sigmaY?: number[]; bootstrapGroups?: string[] }
export const regressionSamples = new WeakMap<RegressionStats, { x: number[]; y: number[]; opts: RegressionOptions }>()

export function regression(
  xs: number[],
  ys: number[],
  opts: RegressionOptions = {}
): RegressionStats | null {
  if (xs.length !== ys.length) return null
  const keep = xs.map((_, i) => i).filter(i => Number.isFinite(xs[i]) && Number.isFinite(ys[i]))
  const sx = opts.sigmaX && keep.map(i => opts.sigmaX![i])
  const sy = opts.sigmaY && keep.map(i => opts.sigmaY![i])
  xs = keep.map(i => xs[i]); ys = keep.map(i => ys[i])
  const n = xs.length
  if (n < 3 || ys.length !== n) return null

  const mx = xs.reduce((a, b) => a + b, 0) / n
  const my = ys.reduce((a, b) => a + b, 0) / n

  let sxx = 0
  let syy = 0
  let sxy = 0
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx
    const dy = ys[i] - my
    sxx += dx * dx
    syy += dy * dy
    sxy += dx * dy
  }
  if (sxx === 0 || syy === 0) return null

  const slope = sxy / sxx
  const intercept = my - slope * mx
  const r = sxy / Math.sqrt(sxx * syy)

  let rss = 0
  for (let i = 0; i < n; i++) {
    const resid = ys[i] - (slope * xs[i] + intercept)
    rss += resid * resid
  }
  const rmse = Math.sqrt(rss / n)

  let demingSlope: number | null = null
  let demingIntercept: number | null = null
  let slopeAttenuationPct: number | null = null
  const lambda = opts.lambda ?? 1.0

  if (opts.errorsInVariables && sxy !== 0 && Number.isFinite(lambda) && lambda > 0) {
    // Deming: solves for the slope minimising perpendicular-ish residuals,
    // weighted by lambda = var(err_y)/var(err_x). lambda = 1 is orthogonal.
    demingSlope = demingSlopeOf(sxx, syy, sxy, lambda)
    demingIntercept = my - demingSlope * mx
    // Same definition as the Python: relative to the OLS slope, so the web
    // gallery and the notebooks quote the same number for the same data.
    if (slope !== 0) {
      slopeAttenuationPct = ((demingSlope - slope) / slope) * 100
    }
  }

  const supplied = !!sx || !!sy
  const yorkFit = opts.errorsInVariables ? york(xs, ys, sx ?? xs.map(() => 1), sy ?? ys.map(() => Math.sqrt(lambda))) : null

  const result: RegressionStats = {
    n,
    slope,
    intercept,
    r,
    r2: r * r,
    demingSlope,
    demingIntercept,
    demingLambda: lambda,
    slopeAttenuationPct,
    rmse,
    yorkSlope: yorkFit?.slope ?? null,
    yorkIntercept: yorkFit?.intercept ?? null,
    yorkAssumption: supplied ? 'York: supplied point uncertainties; uncorrelated errors' : `York: constant errors, λ=${lambda}`,
  }
  regressionSamples.set(result, { x: xs, y: ys, opts: { ...opts, sigmaX: sx, sigmaY: sy, bootstrapGroups: opts.bootstrapGroups && keep.map(i => opts.bootstrapGroups![i]) } })
  return result
}

/** York et al. (2004), uncorrelated errors; mirrors calibration_explorer/hips_lab.py. */
export function york(xs: number[], ys: number[], sx: number[], sy: number[]): { slope: number; intercept: number } | null {
  const n = xs.length
  if (n < 3 || [ys, sx, sy].some(a => a.length !== n) || sx.some(v => !Number.isFinite(v) || v <= 0) || sy.some(v => !Number.isFinite(v) || v <= 0)) return null
  const mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n
  const den = xs.reduce((a, v) => a + (v - mx) ** 2, 0)
  if (!den) return null
  let b = xs.reduce((a, v, i) => a + (v - mx) * (ys[i] - my), 0) / den
  const wx = sx.map(v => 1 / v ** 2), wy = sy.map(v => 1 / v ** 2)
  for (let it = 0; it < 200; it++) {
    const w = wx.map((v, i) => v * wy[i] / (v + b * b * wy[i]))
    const sum = w.reduce((a, v) => a + v, 0)
    const xb = xs.reduce((a, v, i) => a + w[i] * v, 0) / sum, yb = ys.reduce((a, v, i) => a + w[i] * v, 0) / sum
    const u = xs.map(v => v - xb), v = ys.map(y => y - yb)
    const beta = w.map((q, i) => q * (u[i] / wy[i] + b * v[i] / wx[i]))
    const d = w.reduce((a, q, i) => a + q * beta[i] * u[i], 0)
    if (!d) return null
    const next = w.reduce((a, q, i) => a + q * beta[i] * v[i], 0) / d
    if (!Number.isFinite(next)) return null
    if (Math.abs(next - b) < 1e-12 * Math.max(1, Math.abs(b))) {
      const weights = wx.map((q, i) => q * wy[i] / (q + next * next * wy[i]))
      const total = weights.reduce((a, q) => a + q, 0)
      const intercept = ys.reduce((a, y, i) => a + weights[i] * (y - next * xs[i]), 0) / total
      return { slope: next, intercept }
    }
    b = next
  }
  return null
}

/** Deming's slope from the centred sums, λ = Var(y error) / Var(x error) (regression() and deming() share it) */
function demingSlopeOf(sxx: number, syy: number, sxy: number, lambda: number): number {
  const d = syy - lambda * sxx
  return (d + Math.sqrt(d * d + 4 * lambda * sxy * sxy)) / (2 * sxy)
}

/** a correlation's floor: null below `minN` finite pairs (default 3; the weather tab uses 5) */
export interface MinN { minN?: number }

/**
 * Pearson correlation over the finite pairs (over the shorter array when the two differ): regression()'s r
 * (same two-pass sums), null below `minN` pairs (default 3) or with no spread in either.
 */
export function pearson(xs: readonly Num[], ys: readonly Num[], { minN = 3 }: MinN = {}): number | null {
  const len = Math.min(xs.length, ys.length)
  let n = 0, sx = 0, sy = 0
  for (let i = 0; i < len; i++) { const a = xs[i], b = ys[i]; if (isFiniteNumber(a) && isFiniteNumber(b)) { n++; sx += a; sy += b } }
  if (n < Math.max(2, minN)) return null
  const mx = sx / n, my = sy / n
  let sxx = 0, syy = 0, sxy = 0
  for (let i = 0; i < len; i++) {
    const a = xs[i], b = ys[i]
    if (!isFiniteNumber(a) || !isFiniteNumber(b)) continue
    const dx = a - mx, dy = b - my
    sxx += dx * dx; syy += dy * dy; sxy += dx * dy
  }
  return sxx === 0 || syy === 0 ? null : sxy / Math.sqrt(sxx * syy)
}

/** ranks from 1, ties sharing their average rank (scipy.stats.rankdata) */
export function ranks(values: readonly number[]): number[] {
  const order = values.map((x, i) => [x, i] as const).sort((a, b) => a[0] - b[0])
  const out = new Array<number>(values.length)
  for (let i = 0; i < order.length;) {
    let j = i
    while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++
    for (let k = i; k <= j; k++) out[order[k][1]] = (i + j) / 2 + 1
    i = j + 1
  }
  return out
}

/** Spearman's rank correlation: Pearson on the average ranks of the finite pairs (nulls as pearson's, `minN` included) */
export function spearman(xs: readonly Num[], ys: readonly Num[], opts: MinN = {}): number | null {
  const { x, y } = finitePairs(xs, ys)
  return pearson(ranks(x), ranks(y), opts)
}

/**
 * The correlogram's matrix: r for every pair of fields over the rows where both are numbers,
 * null on the diagonal's restatements (`skip`) or where r is undefined. r is symmetric, so each
 * pair is computed once and mirrored.
 */
export function corrMatrix(rows: Record<string, unknown>[], fields: string[], skip: (a: string, b: string) => boolean): (number | null)[][] {
  const cols = fields.map((f) => rows.map((r) => { const v = r[f]; return typeof v === 'number' && Number.isFinite(v) ? v : NaN }))
  const m: (number | null)[][] = fields.map(() => fields.map(() => null))
  const xs: number[] = [], ys: number[] = []
  for (let i = 0; i < fields.length; i++) {
    for (let j = i; j < fields.length; j++) {
      if (skip(fields[i], fields[j])) continue
      xs.length = 0; ys.length = 0
      const a = cols[i], b = cols[j]
      for (let k = 0; k < a.length; k++) if (a[k] === a[k] && b[k] === b[k]) { xs.push(a[k]); ys.push(b[k]) }
      m[i][j] = m[j][i] = pearson(xs, ys)
    }
  }
  return m
}

/** Five-number summary + outliers, matching matplotlib's boxplot whiskers. */
export interface BoxStats {
  min: number
  q1: number
  median: number
  q3: number
  max: number
  outliers: number[]
  n: number
  /** Tukey's fences, Q1 − 1.5·IQR and Q3 + 1.5·IQR: the whiskers stop at the last values inside them */
  loFence: number
  hiFence: number
}

/**
 * Quartiles (numpy's linear), whiskers at the most extreme values within 1.5·IQR of the box, and the values
 * beyond them, over the finite values; null below `minN` of them (default 4; 1 for a plain median and IQR).
 */
export function boxStats(values: readonly Num[], { minN = 4 }: MinN = {}): BoxStats | null {
  const v = finite(values).sort((a, b) => a - b)
  if (v.length < Math.max(1, minN)) return null
  const q = (p: number) => quantileSorted(v, p)
  const q1 = q(0.25)
  const median = q(0.5)
  const q3 = q(0.75)
  const iqr = q3 - q1
  const loFence = q1 - 1.5 * iqr
  const hiFence = q3 + 1.5 * iqr
  const inside = v.filter((d) => d >= loFence && d <= hiFence)
  return {
    min: inside.length ? inside[0] : v[0],
    q1,
    median,
    q3,
    max: inside.length ? inside[inside.length - 1] : v[v.length - 1],
    outliers: v.filter((d) => d < loFence || d > hiFence),
    n: v.length,
    loFence,
    hiFence,
  }
}

/** Gaussian KDE on a fixed grid — powers the density, violin and ridgeline charts. */
export function kde(values: readonly Num[], grid: number[], bandwidth?: number): number[] {
  const v = finite(values)
  if (!v.length) return grid.map(() => 0)
  // Silverman's rule of thumb; guard the degenerate all-identical-values (and single-value) case
  const silverman = 1.06 * (sd(v) || 1) * Math.pow(v.length, -0.2)
  const bw = bandwidth ?? (silverman > 0 ? silverman : 1)
  const norm = 1 / (v.length * bw * Math.sqrt(2 * Math.PI))
  return grid.map((g) => {
    let acc = 0
    for (const d of v) {
      const z = (g - d) / bw
      acc += Math.exp(-0.5 * z * z)
    }
    return acc * norm
  })
}

/** Formatting lives in lib/format.ts; re-exported here for the charts that import it from stats. */
export { fmt } from './format.ts'

/**
 * Two-sample comparison for the mirror histogram: Welch's t (unequal
 * variances), its two-sided p from a normal approximation of the t
 * distribution (n ≥ 30 on both sides in practice), Cohen's d on the pooled
 * SD, and the Mann–Whitney AUC (probability a random A exceeds a random B).
 * The gallery's "stop chasing the p-value" page is the reason the effect
 * sizes come first.
 */
export interface TwoSample {
  nA: number; nB: number; medianA: number; medianB: number; medianDiff: number
  t: number; p: number; d: number; auc: number
}

export function twoSample(a: readonly Num[], b: readonly Num[]): TwoSample | null {
  const A = finite(a).sort((x, y) => x - y)
  const B = finite(b).sort((x, y) => x - y)
  if (A.length < 3 || B.length < 3) return null
  const vari = (v: number[], m: number) => v.reduce((s, x) => s + (x - m) ** 2, 0) / (v.length - 1)
  const mA = mean(A), mB = mean(B)
  const vA = vari(A, mA), vB = vari(B, mB)
  const se = Math.sqrt(vA / A.length + vB / B.length)
  const t = se > 0 ? (mA - mB) / se : 0
  // Welch–Satterthwaite df, then a t → p via the regularised incomplete beta
  const df = se > 0 ? (vA / A.length + vB / B.length) ** 2 / ((vA / A.length) ** 2 / (A.length - 1) + (vB / B.length) ** 2 / (B.length - 1)) : 1
  const p = 2 * (1 - tCdf(Math.abs(t), df))
  const sp = Math.sqrt(((A.length - 1) * vA + (B.length - 1) * vB) / (A.length + B.length - 2))
  const d = sp > 0 ? (mA - mB) / sp : 0
  // AUC = U / (nA·nB), with U counted by a merge over the sorted arrays
  let u = 0
  let j = 0
  let ties = 0
  for (const x of A) {
    while (j < B.length && B[j] < x) j++
    let k = j
    while (k < B.length && B[k] === x) k++
    ties = k - j
    u += j + ties / 2
  }
  return { nA: A.length, nB: B.length, medianA: medianSorted(A), medianB: medianSorted(B), medianDiff: medianSorted(A) - medianSorted(B), t, p, d, auc: u / (A.length * B.length) }
}

/**
 * Mann–Whitney U of a against b, two-sided, as scipy.stats.mannwhitneyu computes it for samples this size: U of the
 * first sample, the normal approximation with the tie correction and a continuity correction of 0.5.
 */
export function mannWhitney(a: readonly number[], b: readonly number[]): { u: number; p: number } | null {
  const n1 = a.length, n2 = b.length
  if (n1 < 1 || n2 < 1) return null
  const all = [...a.map((v) => ({ v, g: 0 })), ...b.map((v) => ({ v, g: 1 }))].sort((x, y) => x.v - y.v)
  const rank = new Array<number>(all.length)
  let tie = 0
  for (let i = 0; i < all.length;) {
    let j = i
    while (j + 1 < all.length && all[j + 1].v === all[i].v) j++
    const t = j - i + 1
    for (let k = i; k <= j; k++) rank[k] = (i + j) / 2 + 1
    tie += t ** 3 - t
    i = j + 1
  }
  const r1 = all.reduce((s, x, i) => s + (x.g === 0 ? rank[i] : 0), 0)
  const u = r1 - (n1 * (n1 + 1)) / 2
  const n = n1 + n2, mu = (n1 * n2) / 2
  const sd = Math.sqrt((n1 * n2 / 12) * ((n + 1) - tie / (n * (n - 1))))
  if (!(sd > 0)) return { u, p: 1 }
  const z = (Math.abs(u - mu) - 0.5) / sd
  return { u, p: Math.min(1, 2 * normalSf(Math.max(0, z))) }
}

/** upper tail of the standard normal, accurate in the far tail (erfc by its continued fraction) */
function normalSf(z: number): number {
  const x = z / Math.SQRT2
  if (x < 3) {
    // erfc via the series for erf
    let sum = x, term = x
    for (let k = 1; k < 200; k++) { term *= (-x * x) / k; const add = term / (2 * k + 1); sum += add; if (Math.abs(add) < 1e-17) break }
    return 0.5 * (1 - (2 / Math.sqrt(Math.PI)) * sum)
  }
  // Lentz continued fraction for erfc(x)
  let f = x, c = x, d = 0
  for (let k = 1; k < 300; k++) {
    const a = k / 2
    d = x + a * d; d = 1 / d
    c = x + a / c
    const delta = c * d
    f *= delta
    if (Math.abs(delta - 1) < 1e-15) break
  }
  return 0.5 * (Math.exp(-x * x) / Math.sqrt(Math.PI)) / f
}

/** CDF of Student's t via the regularised incomplete beta function. */
function tCdf(t: number, df: number): number {
  const x = df / (df + t * t)
  const ib = incompleteBeta(x, df / 2, 0.5)
  return 1 - 0.5 * ib
}

/** Regularised incomplete beta I_x(a, b) by Lentz's continued fraction (Numerical Recipes betai / betacf). */
export function incompleteBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0
  if (x >= 1) return 1
  const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
  const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lbeta) / a
  const cf = (xx: number, aa: number, bb: number) => {
    const tiny = 1e-30
    let c = 1
    let d = 1 - ((aa + bb) * xx) / (aa + 1)
    if (Math.abs(d) < tiny) d = tiny
    d = 1 / d
    let h = d
    for (let m = 1; m <= 200; m++) {
      const m2 = 2 * m
      let aa2 = (m * (bb - m) * xx) / ((aa + m2 - 1) * (aa + m2))
      d = 1 + aa2 * d; if (Math.abs(d) < tiny) d = tiny; d = 1 / d
      c = 1 + aa2 / c; if (Math.abs(c) < tiny) c = tiny
      h *= d * c
      aa2 = (-(aa + m) * (aa + bb + m) * xx) / ((aa + m2) * (aa + m2 + 1))
      d = 1 + aa2 * d; if (Math.abs(d) < tiny) d = tiny; d = 1 / d
      c = 1 + aa2 / c; if (Math.abs(c) < tiny) c = tiny
      const del = d * c
      h *= del
      if (Math.abs(del - 1) < 3e-14) break
    }
    return h
  }
  // use the symmetry relation for convergence
  return x < (a + 1) / (a + b + 2) ? front * cf(x, a, b) : 1 - (Math.exp(Math.log(1 - x) * b + Math.log(x) * a - lbeta) / b) * cf(1 - x, b, a)
}

function lgamma(z: number): number {
  const g = 7
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z)
  z -= 1
  let x = c[0]
  for (let i = 1; i < g + 2; i++) x += c[i] / (z + i)
  const t = z + g + 0.5
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x)
}

/**
 * Split a date-sorted series wherever consecutive points are more than
 * `maxGapDays` apart, so a smoothed line breaks at sampling gaps instead of
 * drawing a straight bridge across months with no filters behind it.
 */
export function splitAtGaps<T>(pts: T[], dateOf: (p: T) => Date, maxGapDays = 45): T[][] {
  const out: T[][] = []
  let cur: T[] = []
  pts.forEach((p, i) => {
    if (i > 0 && +dateOf(p) - +dateOf(pts[i - 1]) > maxGapDays * 864e5) {
      out.push(cur)
      cur = []
    }
    cur.push(p)
  })
  if (cur.length) out.push(cur)
  return out
}

/* ---------------------------------------------------------------------------------------------------------
 * The fits and screens the HIPS / meteorology notebooks run (formerly sections/research/common/stats.ts).
 * Each follows the notebook it stands in for, so the gallery's numbers match the notebook's on the same
 * filters; node-tested against scipy / numpy in src/lib/researchStats.test.mjs.
 * ------------------------------------------------------------------------------------------------------- */

/** an OLS line with scipy.stats.linregress's statistics */
export interface OlsFit {
  n: number
  slope: number
  intercept: number
  r: number
  r2: number
  /** standard errors (scipy linregress stderr and intercept_stderr) */
  seSlope: number
  seIntercept: number
  rmse: number
}

/** OLS of y on x over the finite pairs (scipy.stats.linregress); null below `minN` points (at least 3) or with no spread in x. r is 0 when y is constant. */
export function ols(xs: readonly Num[], ys: readonly Num[], minN = 3): OlsFit | null {
  const { x, y } = finitePairs(xs, ys)
  const n = x.length
  if (n < Math.max(3, minN)) return null
  const mx = mean(x), my = mean(y)
  let sxx = 0, syy = 0, sxy = 0
  for (let i = 0; i < n; i++) { const dx = x[i] - mx, dy = y[i] - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy }
  if (sxx === 0) return null
  const slope = sxy / sxx, intercept = my - slope * mx
  const r = syy === 0 ? 0 : sxy / Math.sqrt(sxx * syy)
  let rss = 0
  for (let i = 0; i < n; i++) { const e = y[i] - (intercept + slope * x[i]); rss += e * e }
  const s2 = rss / (n - 2)
  const sumX2 = x.reduce((a, v) => a + v * v, 0)
  return { n, slope, intercept, r, r2: r * r, seSlope: Math.sqrt(s2 / sxx), seIntercept: Math.sqrt(s2 * sumX2 / (n * sxx)), rmse: Math.sqrt(rss / n) }
}

/** the slope through the origin, Σxy / Σx², with the uncentred R² 1 − RSS / Σy² */
export function originFit(xs: readonly Num[], ys: readonly Num[]): { n: number; slope: number; r2: number } | null {
  const { x, y } = finitePairs(xs, ys)
  const sxx = x.reduce((a, v) => a + v * v, 0)
  if (x.length < 3 || sxx === 0) return null
  const slope = x.reduce((a, v, i) => a + v * y[i], 0) / sxx
  const rss = x.reduce((a, v, i) => a + (y[i] - slope * v) ** 2, 0)
  const syy = y.reduce((a, v) => a + v * v, 0)
  return { n: x.length, slope, r2: syy > 0 ? 1 - rss / syy : NaN }
}

/** Theil–Sen: the median pairwise slope; intercept median(y) − slope · median(x) (scipy.stats.theilslopes) */
export function theilSen(xs: readonly Num[], ys: readonly Num[]): { n: number; slope: number; intercept: number } | null {
  const { x, y } = finitePairs(xs, ys)
  const slopes: number[] = []
  for (let i = 0; i < x.length; i++) for (let j = i + 1; j < x.length; j++) if (x[j] !== x[i]) slopes.push((y[j] - y[i]) / (x[j] - x[i]))
  if (x.length < 3 || !slopes.length) return null
  const slope = median(slopes)
  return { n: x.length, slope, intercept: median(y) - slope * median(x) }
}

/**
 * Deming regression with λ = Var(y error) / Var(x error): λ → ∞ is OLS of y on x, λ → 0 the inverse OLS,
 * λ = 1 orthogonal regression (addis_fabs_ec_deming's deming(x, y, (σy/σx)²)). The same slope as
 * regression(…, { errorsInVariables: true, lambda }), without the rest of the fit.
 */
export function deming(xs: readonly Num[], ys: readonly Num[], lambda: number): { slope: number; intercept: number } | null {
  const { x, y } = finitePairs(xs, ys)
  if (x.length < 3) return null
  const mx = mean(x), my = mean(y)
  let sxx = 0, syy = 0, sxy = 0
  for (let i = 0; i < x.length; i++) { const dx = x[i] - mx, dy = y[i] - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy }
  if (sxy === 0) return null
  const slope = demingSlopeOf(sxx, syy, sxy, lambda)
  return { slope, intercept: my - slope * mx }
}

/** least squares with an intercept and k predictors (columns of X): R², adjusted R², RMSE, coefficients */
export function multiOls(X: readonly (readonly Num[])[], y: readonly Num[]): { n: number; coef: number[]; r2: number; adjR2: number; rmse: number } | null {
  const rows: number[][] = [], ys: number[] = []
  for (let i = 0; i < y.length; i++) {
    const r = X.map((c) => c[i])
    const yi = y[i]
    if (isFiniteNumber(yi) && r.every(isFiniteNumber)) { rows.push([1, ...(r as number[])]); ys.push(yi) }
  }
  const n = ys.length, k = X.length
  if (n <= k + 1) return null
  // normal equations (X'X) b = X'y, solved by Gaussian elimination with partial pivoting
  const p = k + 1
  const A = Array.from({ length: p }, (_, i) => Array.from({ length: p + 1 }, (_, j) => j < p
    ? rows.reduce((s, r) => s + r[i] * r[j], 0) : rows.reduce((s, r, t) => s + r[i] * ys[t], 0)))
  for (let c = 0; c < p; c++) {
    let piv = c
    for (let r = c + 1; r < p; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r
    if (Math.abs(A[piv][c]) < 1e-12) return null
    ;[A[c], A[piv]] = [A[piv], A[c]]
    for (let r = 0; r < p; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let j = c; j <= p; j++) A[r][j] -= f * A[c][j] }
  }
  const coef = A.map((r, i) => r[p] / r[i])
  const my = mean(ys)
  let rss = 0, tss = 0
  rows.forEach((r, t) => { const e = ys[t] - r.reduce((s, v, j) => s + v * coef[j], 0); rss += e * e; tss += (ys[t] - my) ** 2 })
  const r2 = tss > 0 ? 1 - rss / tss : NaN
  return { n, coef, r2, adjR2: 1 - (1 - r2) * (n - 1) / (n - k - 1), rmse: Math.sqrt(rss / n) }
}

/** y = c0 + c1 x + c2 x² by least squares (numpy.polyfit(x, y, 2) reversed), with its R² */
export function quadratic(xs: readonly Num[], ys: readonly Num[]): { coef: [number, number, number]; r2: number } | null {
  const { x, y } = finitePairs(xs, ys)
  const fit = multiOls([x, x.map((v) => v * v)], y)
  return fit ? { coef: fit.coef as [number, number, number], r2: fit.r2 } : null
}

/** the standard normal CDF (Abramowitz–Stegun 7.1.26 erf, |error| < 1.5e-7) */
export function normalCdf(z: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2)
  const erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-(z * z) / 2)
  return z >= 0 ? (1 + erf) / 2 : (1 - erf) / 2
}

/** two-sided p of Pearson's r on n points: the t test, through the regularised incomplete beta I_{df/(df+t²)}(df/2, 1/2) */
export function correlationP(r: number, n: number): number {
  if (n < 4 || !Number.isFinite(r) || Math.abs(r) >= 1) return NaN
  const t = r * Math.sqrt((n - 2) / (1 - r * r)), df = n - 2
  return incompleteBeta(df / (df + t * t), df / 2, 0.5)
}

/**
 * Do two correlations that share a variable differ? r(X,Y) against r(X,Z) with r(Y,Z) known, on n points:
 * pw_mac_calibration's steiger_test (Williams's form, as the notebook computes it). Two-sided p.
 */
export function steiger(rxy: number, rxz: number, ryz: number, n: number): { z: number; p: number } {
  const det = 1 - rxy * rxy - rxz * rxz - ryz * ryz + 2 * rxy * rxz * ryz
  const z = (Math.atanh(rxy) - Math.atanh(rxz)) * Math.sqrt((n - 3) * (1 + ryz) / (2 * det))
  return { z, p: 2 * (1 - normalCdf(Math.abs(z))) }
}

export interface XY { x: number; y: number }
export interface SplitPoint extends XY { s: number }
export interface SplitFit { threshold: number; nLow: number; nHigh: number; low: OlsFit | null; high: OlsFit | null }

/**
 * The threshold sweep (rh_sweep_seasonal, hips_ftir_intercept_rh_recursive_diagnostics): at each threshold t,
 * OLS on the points with s ≤ t (low) and s > t (high), each only with at least `minN` points.
 */
export function splitSweep(points: readonly SplitPoint[], thresholds: readonly number[], minN = 5): SplitFit[] {
  return thresholds.map((t) => {
    const lo = points.filter((p) => p.s <= t), hi = points.filter((p) => p.s > t)
    const fit = (ps: readonly SplitPoint[]) => ps.length >= minN ? ols(ps.map((p) => p.x), ps.map((p) => p.y)) : null
    return { threshold: t, nLow: lo.length, nHigh: hi.length, low: fit(lo), high: fit(hi) }
  })
}

/** the threshold whose low-side slope is closest to 1, among those with ≥ minN points on the low side */
export function bestLowThreshold(sweep: readonly SplitFit[], minN: number): SplitFit | null {
  return sweep.filter((s) => s.low && s.nLow >= minN).sort((a, b) => Math.abs(a.low!.slope - 1) - Math.abs(b.low!.slope - 1))[0] ?? null
}

export interface ScanRow { variable: string; quantile: number; threshold: number; nLow: number; nHigh: number; low: OlsFit; high: OlsFit; divergence: number; closestTo1: number; coverage: number; score: number }

/**
 * The recursive driver scan (hips_ftir_intercept_rh_recursive_diagnostics variable_split_scan): split the pairs at
 * quantiles 0.2 … 0.8 of a candidate variable, fit each side, and score the split by how far the two lines
 * diverge, |Δslope| + 0.2 |Δintercept|, times the share of the pairs both sides use. A variable needs at least
 * five distinct values and both sides `minN` points.
 */
export function splitScan(variable: string, points: readonly SplitPoint[], total: number, minN = 30): ScanRow[] {
  const vals = points.map((p) => p.s)
  if (new Set(vals).size < 5) return []
  const out: ScanRow[] = []
  for (let k = 2; k <= 8; k++) {
    const q = k / 10, thr = quantile(vals, q)
    const lo = points.filter((p) => p.s <= thr), hi = points.filter((p) => p.s > thr)
    if (lo.length < minN || hi.length < minN) continue
    const low = ols(lo.map((p) => p.x), lo.map((p) => p.y)), high = ols(hi.map((p) => p.x), hi.map((p) => p.y))
    if (!low || !high) continue
    const divergence = Math.abs(low.slope - high.slope) + 0.2 * Math.abs(low.intercept - high.intercept)
    const coverage = (lo.length + hi.length) / total
    out.push({ variable, quantile: q, threshold: thr, nLow: lo.length, nHigh: hi.length, low, high, divergence,
      closestTo1: Math.min(Math.abs(low.slope - 1), Math.abs(high.slope - 1)), coverage, score: divergence * coverage })
  }
  return out
}

export interface WindowFit { center: number; median: number; min: number; max: number; n: number; slope: number; intercept: number; r: number }

/** Sliding-window OLS along x (two_problem_decomposition §4): windows of `size` points sorted by x, moved by `step`. */
export function slidingWindow(points: readonly XY[], size: number, step: number): WindowFit[] {
  const s = points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y)).slice().sort((a, b) => a.x - b.x)
  const out: WindowFit[] = []
  for (let start = 0; start < s.length - size; start += step) {
    const w = s.slice(start, start + size), xs = w.map((p) => p.x)
    if (new Set(xs).size <= 3) continue
    const f = ols(xs, w.map((p) => p.y))
    if (f) out.push({ center: mean(xs), median: median(xs), min: xs[0], max: xs[xs.length - 1], n: w.length, slope: f.slope, intercept: f.intercept, r: f.r })
  }
  return out
}

/** OLS leverage, standardised residuals and Cook's distance (hips_intercept_investigation §6) */
export function influence(points: readonly XY[]): { h: number; stdResid: number; cook: number }[] | null {
  const f = ols(points.map((p) => p.x), points.map((p) => p.y))
  if (!f) return null
  const n = points.length, mx = mean(points.map((p) => p.x))
  const sxx = points.reduce((a, p) => a + (p.x - mx) ** 2, 0)
  const s2 = points.reduce((a, p) => a + (p.y - f.intercept - f.slope * p.x) ** 2, 0) / (n - 2)
  return points.map((p) => {
    const h = 1 / n + (p.x - mx) ** 2 / sxx, e = p.y - f.intercept - f.slope * p.x
    const stdResid = e / Math.sqrt(s2 * (1 - h))
    return { h, stdResid, cook: (stdResid * stdResid / 2) * (h / (1 - h)) }
  })
}
