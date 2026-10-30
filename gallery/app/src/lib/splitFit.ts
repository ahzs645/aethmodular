/**
 * Split at a cutoff, fit each side, and look at the residual against the splitting variable: the threshold analysis
 * the FTIR/HIPS notebooks ran again and again on iron/EC (primary_tasks_notebook Task 1.2, Task_Analysis_Notebook
 * c34–38 by concentration group, follow_up_analysis c24 at each site's median). Pure, so the Correlation tab's
 * panel (charts/correlation/SplitFit) and its node test (splitFit.test.mjs) share it.
 *
 * The notebooks' rules, kept here: a filter needs x, y and the variable; "low" is variable ≤ cutoff and "high" is
 * > cutoff (`df[ratio] <= threshold`); a side needs three filters for a fit (scipy linregress); the residual is
 * y − x; ΔR² is R²(low) − R²(high).
 */
import { regression, type RegressionOptions, type RegressionStats } from './stats.ts'

export interface SplitItem { x: number; y: number; t: number; site: string }

const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

/**
 * The splitting variable of a row: one field, or one field over another (Fe / EC). A ratio with a zero or missing
 * denominator has no value (the notebooks' division gave ±inf and dropped it with the NaNs).
 */
export function splitValue(row: Record<string, unknown>, field: string, per: string | null): number | null {
  const a = row[field]
  if (!finite(a)) return null
  if (!per) return a
  const b = row[per]
  return finite(b) && b !== 0 ? a / b : null
}

/** the p-quantile of ascending values, linear between order statistics (numpy's default); NaN when empty */
export function quantileSorted(sorted: readonly number[], p: number): number {
  if (!sorted.length) return NaN
  const h = (sorted.length - 1) * p
  const lo = Math.floor(h)
  return sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (h - lo)
}

export const medianOf = (values: readonly number[]) => quantileSorted([...values].filter(Number.isFinite).sort((a, b) => a - b), 0.5)

/** each site's median of the variable over these items (follow_up_analysis: `df['iron_ec_ratio'].median()` per site) */
export function siteMedianCuts(items: readonly SplitItem[]): Map<string, number> {
  const by = new Map<string, number[]>()
  for (const it of items) (by.get(it.site) ?? by.set(it.site, []).get(it.site)!).push(it.t)
  return new Map([...by].map(([s, v]) => [s, medianOf(v)]))
}

/** the two sides: low ≤ the item's cutoff < high; an item whose cutoff is unknown (NaN) is on neither */
export function splitAt<T extends { t: number }>(items: readonly T[], cutOf: (it: T) => number): { low: T[]; high: T[] } {
  const low: T[] = [], high: T[] = []
  for (const it of items) {
    const c = cutOf(it)
    if (!Number.isFinite(c)) continue
    if (it.t <= c) low.push(it)
    else high.push(it)
  }
  return { low, high }
}

/** a fit of y on x over these items (null below three, or with no spread), with the regression options of the chart */
export function fitItems(items: readonly { x: number; y: number }[], options?: RegressionOptions): RegressionStats | null {
  return regression(items.map((p) => p.x), items.map((p) => p.y), options)
}

export type ResidualKind = 'difference' | 'fit'

/**
 * Each item's residual: y − x (the notebooks', a difference between two methods in the same units), or y less the
 * pooled OLS line through every item (for a pair whose units differ, where y − x means nothing). Null without a fit.
 */
export function residuals(items: readonly { x: number; y: number }[], kind: ResidualKind): number[] | null {
  if (kind === 'difference') return items.map((p) => p.y - p.x)
  const f = fitItems(items)
  return f ? items.map((p) => p.y - (f.intercept + f.slope * p.x)) : null
}

/** one row of the notebooks' summary table: a cutoff and each side's n, R² and OLS slope, ΔR² = R²(low) − R²(high) */
export interface SplitRow { cut: number; nLow: number; nHigh: number; low: RegressionStats | null; high: RegressionStats | null; dR2: number | null }

export function splitTable(items: readonly SplitItem[], cuts: readonly number[], options?: RegressionOptions): SplitRow[] {
  return cuts.map((cut) => {
    const { low, high } = splitAt(items, () => cut)
    const fl = fitItems(low, options), fh = fitItems(high, options)
    return { cut, nLow: low.length, nHigh: high.length, low: fl, high: fh, dR2: fl && fh ? fl.r2 - fh.r2 : null }
  })
}

/** cutoffs as typed ("50, 100, 200" or "0.05 0.1"), ascending, without repeats */
export function parseCuts(text: string): number[] {
  const out = text.split(/[\s,;]+/).filter(Boolean).map(Number).filter(Number.isFinite)
  return [...new Set(out)].sort((a, b) => a - b)
}

/** a cutoff to start from: the median, rounded to three significant figures (so it reads as typed) */
export const roundCut = (v: number) => (Number.isFinite(v) ? +v.toPrecision(3) : NaN)
