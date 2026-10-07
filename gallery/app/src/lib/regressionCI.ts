import { york, regressionSamples, type RegressionStats } from './stats.ts'
import type { Estimator } from './regressionMethods.ts'

export interface FitCI {
  slope: [number, number]; intercept: [number, number]
  draws: [number, number][]; description: string
}
const cache = new WeakMap<RegressionStats, Map<Estimator, FitCI | null>>()
export function percentile(values: number[], p: number): number {
  const v = values.slice().sort((a, b) => a - b), i = (v.length - 1) * p
  return v[Math.floor(i)] + (v[Math.ceil(i)] - v[Math.floor(i)]) * (i % 1)
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
  for (let b = 0; b < 500; b++) {
    const indices: number[] = []
    for (const groups of strata.values()) {
      const blocks = [...groups.values()]
      for (let j = 0; j < blocks.length; j++) indices.push(...blocks[Math.floor(random() * blocks.length)])
    }
    // Only compute the requested coefficients, not all three regressions and
    // residual diagnostics for every resample. Constant-error York = Deming.
    const lambda = opts.lambda ?? 1
    let m: number | undefined, c: number | undefined
    if (method === 'york' && (opts.sigmaX || opts.sigmaY)) {
      const fit = york(indices.map(i => x[i]), indices.map(i => y[i]), indices.map(i => opts.sigmaX?.[i] ?? 1), indices.map(i => opts.sigmaY?.[i] ?? Math.sqrt(lambda)))
      m = fit?.slope; c = fit?.intercept
    } else {
      const mx = indices.reduce((sum, i) => sum + x[i], 0) / indices.length
      const my = indices.reduce((sum, i) => sum + y[i], 0) / indices.length
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
    draws, slope: [percentile(draws.map(d => d[0]), .025), percentile(draws.map(d => d[0]), .975)],
    intercept: [percentile(draws.map(d => d[1]), .025), percentile(draws.map(d => d[1]), .975)],
    description: `95% ${opts.bootstrapGroups ? 'month-block bootstrap within site' : 'paired bootstrap (independent pairs assumed)'}; 500 resamples; conditional on the displayed model and error assumptions`,
  }
  byMethod.set(method, result)
  return result
}
/** Evaluate joint slope/intercept draws to retain their covariance. */
export function lineInterval(ci: FitCI, x: number): [number, number] {
  const ys = ci.draws.map(([m, b]) => m * x + b)
  return [percentile(ys, .025), percentile(ys, .975)]
}
