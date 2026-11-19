/**
 * A saved calibration run under the linearised HIPS view (lib/hipsLinearise; docs/hips-loading-protocol.md).
 *
 * A run keeps each evaluation filter's prediction and its HIPS Fabs, so the correction can be applied filter by
 * filter and every readout fitted again. The calibration grid cannot follow: it keeps one fitted line per
 * configuration, and a per-filter factor cannot be recovered from a line.
 */
import { correction, EXPECTED_FROM, opticalDepth, SEEN_SITES, siteReading, type HipsScope } from './hipsLinearise.ts'
import { demingLambda } from './macMath.ts'
import { regression } from './stats.ts'
import type { CalibRun, MetricRow, RunEval } from './types'

/** One readout row fitted again on these filters, in the explorer's convention: y = prediction, x = reference ÷ MAC. */
export function refitRow(e: RunEval, idx: readonly number[], row: MetricRow, lambda10: number): MetricRow {
  const set = row.evaluation_set === 'fixed' ? idx.filter((i) => e.fixed[i]) : idx
  const mac = row.MAC
  const xs = set.map((i) => (mac ? e.ref[i] / mac : e.ref[i]))
  const ys = set.map((i) => e.pred[i])
  const st = xs.length >= 3 ? regression(xs, ys, { errorsInVariables: true, lambda: demingLambda(mac, lambda10) }) : null
  const rmse = xs.length ? Math.sqrt(xs.reduce((a, x, k) => a + (ys[k] - x) ** 2, 0) / xs.length) : null
  return {
    ...row,
    n: set.length,
    R2: st?.r2 ?? null,
    RMSE: rmse,
    ols_slope: st?.slope ?? null,
    ols_intercept: st?.intercept ?? null,
    deming_slope: st?.demingSlope ?? null,
    deming_intercept: st?.demingIntercept ?? null,
  }
}

/**
 * Whether the view's scope covers a site (hipsLinearise HIPS_SCOPES), read from the 11 Nov screen's table so that it
 * needs no filters loaded: 'expected' is a median filter expected to under-read by more than 10 %, 'seen' the sites
 * whose own filters show the bend. A site the screen does not list is covered only by 'all'.
 */
export function siteInScope(code: string | undefined, scope: HipsScope): boolean {
  if (scope === 'all') return true
  if (!code) return false
  if (scope === 'seen') return SEEN_SITES.has(code)
  const reading = siteReading(code)
  return !!reading && reading.expected_under_read > EXPECTED_FROM
}

/**
 * The run with every evaluation filter's Fabs linearised at a ceiling, each by its own factor 1 / (1 − τ/τ∞) with τ
 * from its Fabs and sampled volume, and every metrics row fitted again. A filter without a sampled volume, or at or
 * past the ceiling, drops out: it never keeps its measured value beside corrected filters (lib/viewFields has the
 * same rule). A run scored against a reference EC, not Fabs, is returned as it is.
 */
export function lineariseRun(run: CalibRun, volumeOf: (id: string | null) => number | null | undefined, ceiling: number, lambda10: number): CalibRun {
  if (run.ref_kind !== 'fabs') return run
  const e = run.eval
  const idx: number[] = [], ref: number[] = []
  e.ref.forEach((fabs, i) => {
    const volume = volumeOf(e.id[i])
    const k = typeof volume === 'number' && volume > 0 && Number.isFinite(fabs) ? correction(opticalDepth(fabs, volume), ceiling) : NaN
    if (Number.isFinite(k)) { idx.push(i); ref.push(fabs * k) }
  })
  const pick = <T,>(a: T[]) => idx.map((i) => a[i])
  const ev: RunEval = { id: pick(e.id), ref, pred: pick(e.pred), group: pick(e.group), date: e.date ? pick(e.date) : null, deployed: e.deployed ? pick(e.deployed) : null, fixed: pick(e.fixed) }
  const every = ev.pred.map((_, i) => i)
  return { ...run, eval: ev, metrics: run.metrics.map((row) => refitRow(ev, every, row, lambda10)), hips_linearised: { ceiling, dropped: e.ref.length - idx.length } }
}

/** ', linearised' for the axis title of a run under the view, '' otherwise */
export const runHipsMark = (run: Pick<CalibRun, 'hips_linearised'> | null | undefined): string => (run?.hips_linearised ? ', linearised' : '')
