/**
 * The calibration explorer's saved fits (calibration_runs.json runs[].metrics at MAC 10), refitted from its
 * per-filter readouts (runs[].eval), so a chart's lines follow Filters → excluded: on the explorer's own set they
 * reproduce the saved row (explorerFit.test.mjs); on the filters that count they move with the exclusions.
 */
import { deming, median, ols } from './stats.ts'

/** the columns of a saved metrics row a chart draws, refitted: OLS and Deming of predicted EC on Fabs/10 */
export interface ExplorerFit { n: number; ols_slope: number; ols_intercept: number; deming_slope: number | null; deming_intercept: number | null; R2: number }

/**
 * OLS and Deming (λ at MAC 10, calibration.json deming_lambda_mac10) of `pred` on `fabs` / 10, as the explorer fits
 * its metrics; null below three filters. Re-express at another MAC with lib/mac (slope × MAC/10, intercept as is).
 */
export function explorerFit(fabs: readonly number[], pred: readonly number[], lambda10: number): ExplorerFit | null {
  const x = fabs.map((f) => f / 10)
  const o = ols(x, pred)
  if (!o) return null
  const d = deming(x, pred, lambda10)
  return { n: o.n, ols_slope: o.slope, ols_intercept: o.intercept, deming_slope: d?.slope ?? null, deming_intercept: d?.intercept ?? null, R2: o.r2 }
}

/** one estimator's line of a refit ('ols' or 'deming', the two the explorer saves); null for any other */
export function lineOf(f: ExplorerFit | null | undefined, est: string): { slope: number; intercept: number } | null {
  if (!f) return null
  const slope = est === 'ols' ? f.ols_slope : est === 'deming' ? f.deming_slope : null
  const intercept = est === 'ols' ? f.ols_intercept : est === 'deming' ? f.deming_intercept : null
  return slope == null || intercept == null ? null : { slope, intercept }
}

/** the explorer's plausibility card (runs[].plausibility), recomputed over the filters given */
export interface Plausibility { n: number; median: number | null; negative_pct: number | null; above_8_pct: number | null; group_medians: Record<string, number> }

/**
 * A run's plausibility over its dated Addis filters that `counts` (by index into runs[].eval): median predicted EC,
 * percent negative and above 8 µg/m³, and the median per season group. On every dated filter it is the explorer's
 * saved card (percentages to its one decimal; explorerFit.test.mjs).
 */
export function plausibilityOf(ev: { pred: readonly number[]; group: readonly string[]; date?: readonly (string | null)[] | null }, counts: (i: number) => boolean = () => true): Plausibility {
  const v: number[] = [], byGroup = new Map<string, number[]>()
  ev.pred.forEach((p, i) => {
    if (!ev.date?.[i] || !Number.isFinite(p) || !counts(i)) return
    v.push(p)
    const g = ev.group[i]
    if (!byGroup.has(g)) byGroup.set(g, [])
    byGroup.get(g)!.push(p)
  })
  const pct = (k: number) => (v.length ? (100 * k) / v.length : null)
  return { n: v.length, median: v.length ? median(v) : null, negative_pct: pct(v.filter((x) => x < 0).length), above_8_pct: pct(v.filter((x) => x > 8).length),
    group_medians: Object.fromEntries([...byGroup].map(([g, xs]) => [g, median(xs)])) }
}
