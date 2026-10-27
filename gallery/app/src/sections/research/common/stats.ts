/**
 * The fits and screens the HIPS / meteorology notebooks run, as pure functions over numbers (node-tested in
 * src/lib/researchStats.test.mjs). Each follows the notebook it stands in for, so the gallery's numbers match
 * the notebook's on the same filters: OLS is scipy.stats.linregress, the through-origin R² is uncentred
 * (hips_ftir_intercept_rh_recursive_diagnostics fit_line), Theil–Sen's intercept is scipy's
 * (median y − slope · median x), quantiles are numpy's linear default.
 */

export interface Fit {
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

const finitePairs = (xs: readonly number[], ys: readonly number[]) => {
  const x: number[] = [], y: number[] = []
  for (let i = 0; i < Math.min(xs.length, ys.length); i++) if (Number.isFinite(xs[i]) && Number.isFinite(ys[i])) { x.push(xs[i]); y.push(ys[i]) }
  return { x, y }
}
const mean = (v: readonly number[]) => v.reduce((a, b) => a + b, 0) / v.length

/** OLS of y on x; null below `minN` points or with no spread in x */
export function ols(xs: readonly number[], ys: readonly number[], minN = 3): Fit | null {
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
export function originFit(xs: readonly number[], ys: readonly number[]): { n: number; slope: number; r2: number } | null {
  const { x, y } = finitePairs(xs, ys)
  const sxx = x.reduce((a, v) => a + v * v, 0)
  if (x.length < 3 || sxx === 0) return null
  const slope = x.reduce((a, v, i) => a + v * y[i], 0) / sxx
  const rss = x.reduce((a, v, i) => a + (y[i] - slope * v) ** 2, 0)
  const syy = y.reduce((a, v) => a + v * v, 0)
  return { n: x.length, slope, r2: syy > 0 ? 1 - rss / syy : NaN }
}

export function median(values: readonly number[]): number {
  return quantile(values, 0.5)
}

/** numpy's default (linear) quantile of unsorted values; NaN when empty */
export function quantile(values: readonly number[], p: number): number {
  const s = values.filter(Number.isFinite).slice().sort((a, b) => a - b)
  if (!s.length) return NaN
  const h = (s.length - 1) * p, lo = Math.floor(h)
  return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (h - lo)
}

/** Theil–Sen: the median pairwise slope; intercept median(y) − slope · median(x) (scipy.stats.theilslopes) */
export function theilSen(xs: readonly number[], ys: readonly number[]): { n: number; slope: number; intercept: number } | null {
  const { x, y } = finitePairs(xs, ys)
  const slopes: number[] = []
  for (let i = 0; i < x.length; i++) for (let j = i + 1; j < x.length; j++) if (x[j] !== x[i]) slopes.push((y[j] - y[i]) / (x[j] - x[i]))
  if (x.length < 3 || !slopes.length) return null
  const slope = median(slopes)
  return { n: x.length, slope, intercept: median(y) - slope * median(x) }
}

/**
 * Deming regression with λ = Var(y error) / Var(x error): λ → ∞ is OLS of y on x, λ → 0 the inverse OLS,
 * λ = 1 orthogonal regression (addis_fabs_ec_deming's deming(x, y, (σy/σx)²)).
 */
export function deming(xs: readonly number[], ys: readonly number[], lambda: number): { slope: number; intercept: number } | null {
  const { x, y } = finitePairs(xs, ys)
  if (x.length < 3) return null
  const mx = mean(x), my = mean(y)
  let sxx = 0, syy = 0, sxy = 0
  for (let i = 0; i < x.length; i++) { const dx = x[i] - mx, dy = y[i] - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy }
  if (sxy === 0) return null
  const a = syy - lambda * sxx
  const slope = (a + Math.sqrt(a * a + 4 * lambda * sxy * sxy)) / (2 * sxy)
  return { slope, intercept: my - slope * mx }
}

/** least squares with an intercept and k predictors (columns of X): R², adjusted R², RMSE, coefficients */
export function multiOls(X: readonly (readonly number[])[], y: readonly number[]): { n: number; coef: number[]; r2: number; adjR2: number; rmse: number } | null {
  const rows: number[][] = [], ys: number[] = []
  for (let i = 0; i < y.length; i++) {
    const r = X.map((c) => c[i])
    if (Number.isFinite(y[i]) && r.every(Number.isFinite)) { rows.push([1, ...r]); ys.push(y[i]) }
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
export function quadratic(xs: readonly number[], ys: readonly number[]): { coef: [number, number, number]; r2: number } | null {
  const { x, y } = finitePairs(xs, ys)
  const fit = multiOls([x, x.map((v) => v * v)], y)
  return fit ? { coef: fit.coef as [number, number, number], r2: fit.r2 } : null
}

/** average ranks (ties share their mean rank), 1-based */
export function ranks(values: readonly number[]): number[] {
  const order = values.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0])
  const out = new Array<number>(values.length)
  for (let i = 0; i < order.length;) {
    let j = i
    while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++
    for (let k = i; k <= j; k++) out[order[k][1]] = (i + j) / 2 + 1
    i = j + 1
  }
  return out
}

export function pearsonR(xs: readonly number[], ys: readonly number[]): number | null {
  const { x, y } = finitePairs(xs, ys)
  if (x.length < 3) return null
  const mx = mean(x), my = mean(y)
  let sxx = 0, syy = 0, sxy = 0
  for (let i = 0; i < x.length; i++) { const dx = x[i] - mx, dy = y[i] - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy }
  return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : null
}

/** Spearman's ρ: Pearson's r of the average ranks */
export function spearman(xs: readonly number[], ys: readonly number[]): number | null {
  const { x, y } = finitePairs(xs, ys)
  return x.length < 3 ? null : pearsonR(ranks(x), ranks(y))
}

/** the standard normal CDF (Abramowitz–Stegun 7.1.26 erf, |error| < 1.5e-7) */
export function normalCdf(z: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2)
  const erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-(z * z) / 2)
  return z >= 0 ? (1 + erf) / 2 : (1 - erf) / 2
}

/** two-sided p of Pearson's r on n points (t test, normal approximation to t for n > 30, else a series) */
export function correlationP(r: number, n: number): number {
  if (n < 4 || !Number.isFinite(r) || Math.abs(r) >= 1) return NaN
  const t = r * Math.sqrt((n - 2) / (1 - r * r)), df = n - 2
  // Student t two-sided p through the regularised incomplete beta: I_{df/(df+t²)}(df/2, 1/2)
  return incompleteBeta(df / (df + t * t), df / 2, 0.5)
}

function logGamma(z: number): number {
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]
  let x = z, y = z, tmp = x + 5.5
  tmp -= (x + 0.5) * Math.log(tmp)
  let ser = 1.000000000190015
  for (const v of c) ser += v / ++y
  return -tmp + Math.log(2.5066282746310005 * ser / x)
}
/** regularised incomplete beta I_x(a, b) (Numerical Recipes betai / betacf) */
export function incompleteBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0
  if (x >= 1) return 1
  const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x))
  const cf = (xx: number, aa: number, bb: number) => {
    let c = 1, d = 1 - (aa + bb) * xx / (aa + 1)
    d = Math.abs(d) < 1e-30 ? 1e-30 : d
    d = 1 / d
    let h = d
    for (let m = 1; m <= 200; m++) {
      const m2 = 2 * m
      let aa2 = m * (bb - m) * xx / ((aa + m2 - 1) * (aa + m2))
      d = 1 + aa2 * d; d = Math.abs(d) < 1e-30 ? 1e-30 : d; c = 1 + aa2 / c; c = Math.abs(c) < 1e-30 ? 1e-30 : c; d = 1 / d; h *= d * c
      aa2 = -(aa + m) * (aa + bb + m) * xx / ((aa + m2) * (aa + m2 + 1))
      d = 1 + aa2 * d; d = Math.abs(d) < 1e-30 ? 1e-30 : d; c = 1 + aa2 / c; c = Math.abs(c) < 1e-30 ? 1e-30 : c; d = 1 / d
      const del = d * c
      h *= del
      if (Math.abs(del - 1) < 3e-12) break
    }
    return h
  }
  return x < (a + 1) / (a + b + 2) ? bt * cf(x, a, b) / a : 1 - bt * cf(1 - x, b, a) / b
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
export interface SplitFit { threshold: number; nLow: number; nHigh: number; low: Fit | null; high: Fit | null }

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

export interface ScanRow { variable: string; quantile: number; threshold: number; nLow: number; nHigh: number; low: Fit; high: Fit; divergence: number; closestTo1: number; coverage: number; score: number }

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
