import { york, regressionSamples, type RegressionStats } from './stats.ts'
import type { Estimator } from './regressionMethods.ts'

export interface FitCI {
  slope: [number, number]; intercept: [number, number]
  draws: [number, number][]; description: string
}
const cache = new WeakMap<RegressionStats, Map<Estimator, FitCI | null>>()
export function percentile(values: number[], p: number): number {
  return sortedPercentile(Float64Array.from(values).sort(), p)
}
/** a percentile of values already in ascending order (a typed array sorts numerically and fast) */
function sortedPercentile(v: ArrayLike<number>, p: number): number {
  const i = (v.length - 1) * p
  return v[Math.floor(i)] + (v[Math.ceil(i)] - v[Math.floor(i)]) * (i % 1)
}
/** the 2.5 and 97.5 percentiles from one sort */
function interval95(values: Float64Array): [number, number] {
  values.sort()
  return [sortedPercentile(values, .025), sortedPercentile(values, .975)]
}
export function monthBlocks(rows: { site?: string; date?: string | null }[]): string[] | undefined {
  return rows.length && rows.every(r => r.site && /^\d{4}-\d{2}/.test(r.date ?? ''))
    ? rows.map(r => `${r.site}|${r.date!.slice(0, 7)}`) : undefined
}
/** Conditional percentile confidence intervals, not prediction intervals.
 * Resample complete x/y/uncertainty tuples; month blocks stay intact within site.
 * Saved coefficient-only summaries cannot provide a joint line distribution.
 */
export function confidence(stats: RegressionStats | null, method: Estimator): FitCI | null {
  if (!stats) return null
  let byMethod = cache.get(stats)
  if (!byMethod) { byMethod = new Map(); cache.set(stats, byMethod) }
  if (byMethod.has(method)) return byMethod.get(method)!
  const source = regressionSamples.get(stats)
  const fittedSlope = method === 'ols' ? stats.slope : method === 'deming' ? stats.demingSlope : stats.yorkSlope
  if (!source || stats.n < 8 || fittedSlope == null || !Number.isFinite(fittedSlope)) { byMethod.set(method, null); return null }
  const { x, y, opts } = source
  const strata = new Map<string, Map<string, number[]>>()
  x.forEach((_, i) => {
    const key = opts.bootstrapGroups?.[i] ?? String(i)
    const site = opts.bootstrapGroups ? key.split('|')[0] : 'pairs'
    if (!strata.has(site)) strata.set(site, new Map())
    const groups = strata.get(site)!
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key)!.push(i)
  })
  // Few blocks cannot support a credible 95% block interval.
  if (opts.bootstrapGroups && [...strata.values()].some(s => s.size < 6)) { byMethod.set(method, null); return null }
  let seed = 873211
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296 }
  const draws: [number, number][] = []
  // the blocks of each stratum, built once (a resample draws as many blocks as the stratum has, so its
  // size varies with the blocks drawn; the buffer holds the largest possible)
  const strataBlocks = [...strata.values()].map((groups) => [...groups.values()])
  const largest = strataBlocks.reduce((sum, blocks) => sum + blocks.length * Math.max(...blocks.map((b) => b.length)), 0)
  const buffer = new Int32Array(largest)
  for (let b = 0; b < 500; b++) {
    let size = 0
    for (const blocks of strataBlocks) {
      for (let j = 0; j < blocks.length; j++) {
        const block = blocks[Math.floor(random() * blocks.length)]
        for (let k = 0; k < block.length; k++) buffer[size++] = block[k]
      }
    }
    const indices = buffer.subarray(0, size)
    // Only compute the requested coefficients, not all three regressions and
    // residual diagnostics for every resample. Constant-error York = Deming.
    const lambda = opts.lambda ?? 1
    let m: number | undefined, c: number | undefined
    if (method === 'york' && (opts.sigmaX || opts.sigmaY)) {
      const pick = (f: (i: number) => number) => Array.from(indices, f)
      const fit = york(pick(i => x[i]), pick(i => y[i]), pick(i => opts.sigmaX?.[i] ?? 1), pick(i => opts.sigmaY?.[i] ?? Math.sqrt(lambda)))
      m = fit?.slope; c = fit?.intercept
    } else {
      let sx = 0, sy = 0
      for (const i of indices) { sx += x[i]; sy += y[i] }
      const mx = sx / size, my = sy / size
      let xx = 0, yy = 0, xy = 0
      for (const i of indices) { const dx = x[i] - mx, dy = y[i] - my; xx += dx * dx; yy += dy * dy; xy += dx * dy }
      if (xx && yy && (method === 'ols' || xy)) {
        const d = yy - lambda * xx
        m = method === 'ols' ? xy / xx : (d + Math.sqrt(d * d + 4 * lambda * xy * xy)) / (2 * xy)
        c = my - m * mx
      }
    }
    if (m != null && c != null && Number.isFinite(m) && Number.isFinite(c)) draws.push([m, c])
  }
  const result: FitCI | null = draws.length < 450 ? null : {
    draws, slope: interval95(Float64Array.from(draws, d => d[0])), intercept: interval95(Float64Array.from(draws, d => d[1])),
    description: `95% ${opts.bootstrapGroups ? 'month-block bootstrap within site' : 'paired bootstrap (independent pairs assumed)'}; 500 resamples; conditional on the displayed model and error assumptions`,
  }
  byMethod.set(method, result)
  return result
}
const bands = new WeakMap<FitCI, Map<number, [number, number]>>()
/**
 * Evaluate joint slope/intercept draws to retain their covariance. A band asks at ~60 x positions on every
 * render, so each x is computed once per CI (one sort of the 500 draws) and kept.
 */
export function lineInterval(ci: FitCI, x: number): [number, number] {
  let byX = bands.get(ci)
  if (!byX) { byX = new Map(); bands.set(ci, byX) }
  const hit = byX.get(x)
  if (hit) return hit
  const ys = new Float64Array(ci.draws.length)
  for (let i = 0; i < ci.draws.length; i++) ys[i] = ci.draws[i][0] * x + ci.draws[i][1]
  const out = interval95(ys)
  if (byX.size > 4000) byX.clear()
  byX.set(x, out)
  return out
}
