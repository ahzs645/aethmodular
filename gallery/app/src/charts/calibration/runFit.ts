import { regression, type RegressionStats } from '../../lib/stats.ts'
import { demingLambda } from '../../lib/macMath.ts'
import type { RunEval } from '../../lib/types.ts'

/**
 * The explorer's readout rows rebuilt in the browser from a run's per-filter readouts (calibration_explorer/app.py
 * crossplot_metrics, _split_indices), on the filters `keep` admits: the charts pass Filters → excluded's one rule
 * (lib/markedRows useCountedFilter) by filter id, so the fits follow the export's exclusions, Include and every
 * hand exclusion. With every filter kept they reproduce the saved rows (runFit.test.mjs checks the export).
 */

/** a crossplot_metrics row (lib/types MetricRow's fields) with the regression it came from */
export interface LiveMetricRow {
  evaluation_set: 'fixed' | 'all'; MAC: number | null; n: number
  ols_slope: number; ols_intercept: number; R2: number; RMSE: number
  deming_slope: number | null; deming_intercept: number | null
  stats: RegressionStats
}

/** which of a run's filters count: by index into its eval arrays */
export type KeepIndex = (i: number) => boolean
export const KEEP_ALL: KeepIndex = () => true

/** the pairs of a set of filters: x = ref ÷ MAC (or ref for an EC reference), y = the prediction; finite only */
export function runPairs(ev: RunEval, idx: readonly number[], mac: number | null): { x: number[]; y: number[]; i: number[] } {
  const x: number[] = [], y: number[] = [], at: number[] = []
  for (const i of idx) {
    const xv = mac ? ev.ref[i] / mac : ev.ref[i], yv = ev.pred[i]
    if (Number.isFinite(xv) && Number.isFinite(yv)) { x.push(xv); y.push(yv); at.push(i) }
  }
  return { x, y, i: at }
}

/**
 * One crossplot_metrics row on the filters `idx` (default: every filter `keep` admits): OLS slope, intercept and R²
 * (np.polyfit), Deming at λ = demingLambda(MAC) (1 off an Fabs axis), RMSE against the 1:1 line. `set` 'fixed'
 * keeps the fixed test set only. Null below 3 pairs. `stats` is the regression the row came from (for intervals).
 */
export function metricRowLive(ev: RunEval, set: 'fixed' | 'all', mac: number | null, lambda10: number,
  keep: KeepIndex = KEEP_ALL, idx?: readonly number[]): LiveMetricRow | null {
  const base = idx ?? ev.pred.map((_, i) => i)
  const pick = base.filter((i) => keep(i) && (set === 'all' || ev.fixed[i]))
  const p = runPairs(ev, pick, mac)
  if (p.x.length < 3) return null
  const stats = regression(p.x, p.y, { errorsInVariables: true, lambda: demingLambda(mac, lambda10) })
  if (!stats) return null
  const rmse = Math.sqrt(p.x.reduce((a, x, k) => a + (p.y[k] - x) ** 2, 0) / p.x.length)
  return {
    evaluation_set: set, MAC: mac, n: p.x.length,
    ols_slope: stats.slope, ols_intercept: stats.intercept, R2: stats.r2, RMSE: rmse,
    deming_slope: stats.demingSlope, deming_intercept: stats.demingIntercept, stats,
  }
}

/** the explorer's equal-n blind halves */
export const EVAL_SPLITS = ['all', 'early', 'late', 'odd', 'even'] as const
export type EvalSplit = (typeof EVAL_SPLITS)[number]

/** date order of a stratum, missing dates last, ties in file order (_eval_order); file order with no dates at all */
function evalOrder(ev: RunEval, idx: readonly number[]): number[] {
  const dates = ev.date
  if (!dates || !idx.some((i) => dates[i])) return [...idx]
  const key = (i: number) => dates[i] || '9999-12-31'
  return idx.map((i, k) => ({ i, k })).sort((a, b) => (key(a.i) < key(b.i) ? -1 : key(a.i) > key(b.i) ? 1 : a.k - b.k)).map((o) => o.i)
}

/**
 * The filters of one equal-n half (_split_indices): each fixed / non-fixed stratum is date-ordered and halved
 * separately (a stratum too small to halve is dropped from both halves), so the halves keep the same n. Null when
 * fewer than 3 filters remain (the explorer skips that split).
 */
export function splitIndices(ev: RunEval, idx: readonly number[], split: EvalSplit): number[] | null {
  if (split === 'all') return [...idx]
  const hasFixed = ev.fixed.length === ev.pred.length
  const strata = hasFixed ? [idx.filter((i) => ev.fixed[i]), idx.filter((i) => !ev.fixed[i])] : [[...idx]]
  const out: number[] = []
  for (const stratum of strata) {
    const order = evalOrder(ev, stratum)
    const half = Math.floor(order.length / 2)
    if (half < 1) continue
    const pick = split === 'early' ? order.slice(0, half)
      : split === 'late' ? order.slice(order.length - half)
      : order.filter((_, k) => k % 2 === (split === 'odd' ? 1 : 0)).slice(0, half)
    out.push(...pick)
  }
  return out.length < 3 ? null : out.sort((a, b) => a - b)
}

/**
 * The blind-half check (the run's split_check) on the filters `keep` admits, at one MAC: for each split its n and
 * its readout row on the fixed test set when the half holds at least 3 fixed filters (else every pair, as the
 * explorer's sliced target drops a fixed mask under 3), else every pair.
 */
export function splitCheckLive(ev: RunEval, mac: number | null, lambda10: number, keep: KeepIndex = KEEP_ALL):
  { split: EvalSplit; n: number; set: 'fixed' | 'all'; m: LiveMetricRow }[] {
  const idx = ev.pred.map((_, i) => i).filter(keep)
  return EVAL_SPLITS.flatMap((split) => {
    const half = splitIndices(ev, idx, split)
    if (!half) return []
    const set = half.filter((i) => ev.fixed[i]).length >= 3 ? 'fixed' as const : 'all' as const
    const m = metricRowLive(ev, set, mac, lambda10, KEEP_ALL, half)
    return m ? [{ split, n: half.length, set, m }] : []
  })
}
