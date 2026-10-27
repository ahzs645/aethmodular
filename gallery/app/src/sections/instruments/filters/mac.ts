/**
 * The arithmetic behind the "Against the filters" page, kept apart from the charts so it can be tested
 * (mac.test.mjs). Each function is the browser form of a notebook calculation, named after it.
 */

/** a finite number, or null */
export const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length

/**
 * The four ways AAARpos.ipynb (and ETAD_comprehensive_absorption_analysis.ipynb §12) turned paired
 * absorption y and EC x into one MAC: (1) the mean of the per-filter ratios y/x, (2) the ratio of the
 * means, (3) an ordinary least-squares line with an intercept, (4) least squares through the origin
 * (Σxy / Σx²). Only (3) has an intercept, and so a non-zero EC where absorption is zero: -b/m, the
 * notebook's "physical constraint" check. Pairs with x ≤ 0 are left out of (1), whose ratio they break.
 */
export interface MacMethods {
  n: number
  meanOfRatios: number
  ratioOfMeans: number
  ols: { slope: number; intercept: number; r2: number }
  origin: number
  /** EC (x) at which the OLS line reaches zero absorption: -intercept/slope */
  xAtZero: number
}
export function macMethods(x: number[], y: number[]): MacMethods | null {
  const pairs = x.map((xi, i) => [xi, y[i]] as const).filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b))
  if (pairs.length < 3) return null
  const xs = pairs.map((p) => p[0]), ys = pairs.map((p) => p[1])
  const ratios = pairs.filter(([a]) => a > 0).map(([a, b]) => b / a)
  const mx = mean(xs), my = mean(ys)
  let sxx = 0, sxy = 0, syy = 0, s0xx = 0, s0xy = 0
  for (const [a, b] of pairs) {
    sxx += (a - mx) ** 2; sxy += (a - mx) * (b - my); syy += (b - my) ** 2
    s0xx += a * a; s0xy += a * b
  }
  if (sxx === 0 || s0xx === 0) return null
  const slope = sxy / sxx, intercept = my - slope * mx
  return {
    n: pairs.length,
    meanOfRatios: ratios.length ? mean(ratios) : NaN,
    ratioOfMeans: my / mx,
    ols: { slope, intercept, r2: syy === 0 ? NaN : (sxy * sxy) / (sxx * syy) },
    origin: s0xy / s0xx,
    xAtZero: slope === 0 ? NaN : -intercept / slope,
  }
}

/** Ordinary least squares of y on x (with intercept): slope, intercept, R². */
export function ols(x: number[], y: number[]): { slope: number; intercept: number; r2: number; n: number } | null {
  const m = macMethods(x, y)
  return m ? { ...m.ols, n: m.n } : null
}

/**
 * y = c + a·x1 + b·x2 by least squares (iron_correction_test.ipynb Test 2: Fabs = MAC_BC·EC + MAC_Fe·Iron + const,
 * statsmodels OLS). Solved from the centred normal equations. Returns the coefficients, R² and the
 * residuals in the order of the complete rows (`keep` gives their indices in the input).
 */
export interface Ols2 { c: number; a: number; b: number; r2: number; n: number; residuals: number[]; keep: number[] }
export function ols2(y: number[], x1: number[], x2: number[]): Ols2 | null {
  const keep = y.map((_, i) => i).filter((i) => [y[i], x1[i], x2[i]].every(Number.isFinite))
  if (keep.length < 4) return null
  const Y = keep.map((i) => y[i]), A = keep.map((i) => x1[i]), B = keep.map((i) => x2[i])
  const my = mean(Y), ma = mean(A), mb = mean(B)
  let saa = 0, sbb = 0, sab = 0, say = 0, sby = 0, syy = 0
  for (let k = 0; k < Y.length; k++) {
    const da = A[k] - ma, db = B[k] - mb, dy = Y[k] - my
    saa += da * da; sbb += db * db; sab += da * db; say += da * dy; sby += db * dy; syy += dy * dy
  }
  const det = saa * sbb - sab * sab
  if (!(Math.abs(det) > 1e-12 * Math.max(1, saa * sbb)) || syy === 0) return null
  const a = (say * sbb - sby * sab) / det
  const b = (sby * saa - say * sab) / det
  const c = my - a * ma - b * mb
  const residuals = Y.map((v, k) => v - (c + a * A[k] + b * B[k]))
  const rss = residuals.reduce((s, e) => s + e * e, 0)
  return { c, a, b, r2: 1 - rss / syy, n: Y.length, residuals, keep }
}

/**
 * iron_correction_test.ipynb Test 3: subtract k·Iron from the absorption and refit it against EC, for each k.
 * `target` (the expected MAC, 10 m²/g) picks the k whose slope comes closest, as the notebook's sweep did.
 */
export interface SweepStep { k: number; slope: number; intercept: number; r2: number }
export function ironSweep(ec: number[], abs: number[], iron: number[], ks: number[]): SweepStep[] {
  const keep = ec.map((_, i) => i).filter((i) => [ec[i], abs[i], iron[i]].every(Number.isFinite))
  const x = keep.map((i) => ec[i])
  return ks.flatMap((k) => {
    const f = ols(x, keep.map((i) => abs[i] - k * iron[i]))
    return f ? [{ k, slope: f.slope, intercept: f.intercept, r2: f.r2 }] : []
  })
}
export const closestTo = (steps: SweepStep[], target: number) =>
  steps.reduce<SweepStep | null>((best, s) => (!best || Math.abs(s.slope - target) < Math.abs(best.slope - target) ? s : best), null)

/** n evenly spaced values from lo to hi inclusive (numpy.linspace) */
export const linspace = (lo: number, hi: number, n: number) => Array.from({ length: n }, (_, i) => (n === 1 ? lo : lo + ((hi - lo) * i) / (n - 1)))

/** Tukey's fences, Q1 - 1.5 IQR and Q3 + 1.5 IQR, with pandas' (and d3's) linear quantiles: the notebooks' remove_outliers_iqr */
export function iqrFences(values: number[]): [number, number] | null {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b)
  if (v.length < 4) return null
  const q = (p: number) => { const h = (v.length - 1) * p, lo = Math.floor(h); return v[lo] + (v[Math.min(lo + 1, v.length - 1)] - v[lo]) * (h - lo) }
  const q1 = q(0.25), q3 = q(0.75), iqr = q3 - q1
  return [q1 - 1.5 * iqr, q3 + 1.5 * iqr]
}

/**
 * The "season instance" averages of AAARpos.ipynb and ETAD_Aug13.ipynb (the diamonds on the MAC time
 * series): each season's filters in date order, split wherever two consecutive ones are more than
 * `gapDays` apart, and each run with at least `minN` filters averaged. The run's date is its middle
 * filter's (the notebooks' `iloc[len // 2]`).
 */
export interface SeasonRun<T> { season: string; date: string; mean: number; n: number; members: T[] }
export function seasonRuns<T>(points: T[], season: (p: T) => string, date: (p: T) => string, value: (p: T) => number,
  { gapDays = 60, minN = 5 }: { gapDays?: number; minN?: number } = {}): SeasonRun<T>[] {
  const day = (s: string) => Date.parse(s.slice(0, 10)) / 86_400_000
  const out: SeasonRun<T>[] = []
  const bySeason = new Map<string, T[]>()
  for (const p of points) if (Number.isFinite(value(p))) bySeason.set(season(p), [...(bySeason.get(season(p)) ?? []), p])
  for (const [s, ps] of bySeason) {
    const sorted = [...ps].sort((a, b) => date(a).localeCompare(date(b)))
    let run: T[] = []
    const close = () => {
      if (run.length >= minN) out.push({ season: s, date: date(run[Math.floor(run.length / 2)]), mean: mean(run.map(value)), n: run.length, members: run })
      run = []
    }
    for (const p of sorted) {
      if (run.length && day(date(p)) - day(date(run[run.length - 1])) > gapDays) close()
      run.push(p)
    }
    close()
  }
  return out.sort((a, b) => a.date.localeCompare(b.date))
}
