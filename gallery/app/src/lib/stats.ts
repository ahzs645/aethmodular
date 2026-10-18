/** Regression helpers; optional OLS, Deming and York display fits.
 * York follows the uncorrelated-error implementation in calibration_explorer/hips_lab.py.
 * Display preferences never refit EC calibrations or change their saved scores.
 */

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
    const d = syy - lambda * sxx
    demingSlope = (d + Math.sqrt(d * d + 4 * lambda * sxy * sxy)) / (2 * sxy)
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

/** Pearson correlation, NaN-safe on paired arrays: regression()'s r (same two-pass sums, same
 * nulls: fewer than three pairs, or no spread), without the fit's other statistics. */
export function pearson(xs: number[], ys: number[]): number | null {
  if (xs.length !== ys.length) return null
  let n = 0, sx = 0, sy = 0
  for (let i = 0; i < xs.length; i++) if (Number.isFinite(xs[i]) && Number.isFinite(ys[i])) { n++; sx += xs[i]; sy += ys[i] }
  if (n < 3) return null
  const mx = sx / n, my = sy / n
  let sxx = 0, syy = 0, sxy = 0
  for (let i = 0; i < xs.length; i++) {
    if (!Number.isFinite(xs[i]) || !Number.isFinite(ys[i])) continue
    const dx = xs[i] - mx, dy = ys[i] - my
    sxx += dx * dx; syy += dy * dy; sxy += dx * dy
  }
  return sxx === 0 || syy === 0 ? null : sxy / Math.sqrt(sxx * syy)
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
}

export function boxStats(values: number[]): BoxStats | null {
  const v = values.filter((d) => Number.isFinite(d)).sort((a, b) => a - b)
  if (v.length < 4) return null
  const q = (p: number) => {
    const idx = (v.length - 1) * p
    const lo = Math.floor(idx)
    const hi = Math.ceil(idx)
    return lo === hi ? v[lo] : v[lo] + (idx - lo) * (v[hi] - v[lo])
  }
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
  }
}

/** Gaussian KDE on a fixed grid — powers the density, violin and ridgeline charts. */
export function kde(values: number[], grid: number[], bandwidth?: number): number[] {
  const v = values.filter((d) => Number.isFinite(d))
  if (!v.length) return grid.map(() => 0)
  const mean = v.reduce((a, b) => a + b, 0) / v.length
  const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, v.length - 1))
  // Silverman's rule of thumb; guard the degenerate all-identical-values case
  const silverman = 1.06 * (sd || 1) * Math.pow(v.length, -0.2)
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

export function twoSample(a: number[], b: number[]): TwoSample | null {
  const A = a.filter(Number.isFinite).sort((x, y) => x - y)
  const B = b.filter(Number.isFinite).sort((x, y) => x - y)
  if (A.length < 3 || B.length < 3) return null
  const mean = (v: number[]) => v.reduce((s, x) => s + x, 0) / v.length
  const vari = (v: number[], m: number) => v.reduce((s, x) => s + (x - m) ** 2, 0) / (v.length - 1)
  const median = (v: number[]) => (v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2)
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
  return { nA: A.length, nB: B.length, medianA: median(A), medianB: median(B), medianDiff: median(A) - median(B), t, p, d, auc: u / (A.length * B.length) }
}

/** CDF of Student's t via the regularised incomplete beta function. */
function tCdf(t: number, df: number): number {
  const x = df / (df + t * t)
  const ib = regIncBeta(x, df / 2, 0.5)
  return 1 - 0.5 * ib
}

/** Regularised incomplete beta I_x(a, b) by Lentz's continued fraction (Numerical Recipes). */
function regIncBeta(x: number, a: number, b: number): number {
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
