/**
 * ftir_36/ftir_39's locked holdouts at any MAC (node-tested in src/lib/researchStats.test.mjs). ftir_39's per-filter
 * table carries each filter's residual against HIPS/10; gallery/data/export_holdout_fabs.py adds each filter's Fabs,
 * reconstructed from the raw HIPS as the notebooks did and checked against ftir_36's saved fits. With it a residual
 * moves exactly: prediction − Fabs/MAC = residual + Fabs/10 − Fabs/MAC, and the summary ftir_39 draws from those
 * residuals (RMSE inside and outside the source 95 %, Spearman ρ of the domain percentile against |error|) follows.
 */
import type { Row } from '../../../lib/table.ts'
import { correlationP, spearman } from './stats.ts'

/** holdout_fabs.json: each holdout filter's Fabs (Mm⁻¹), keyed by ftir_39's ExternalFilterId (a MediaId) */
export interface HoldoutFabs {
  schema_version: 1
  method: string
  rows: Record<string, { filter: string; site: string; date: string; lot: string; line: string; fabs: number }>
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

/** a per-filter row's Fabs, or null when the export lacks it */
export const holdoutFabs = (r: Row, fabs: HoldoutFabs | null | undefined): number | null => {
  const id = num(r.ExternalFilterId)
  return id == null ? null : fabs?.rows[String(Math.round(id))]?.fabs ?? null
}

/** a per-filter row's SPARTAN filter id (ETAD-0313-1), by which Filters → excluded joins it; null when unknown */
export const holdoutFilterId = (r: Row, fabs: HoldoutFabs | null | undefined): string | null => {
  const id = num(r.ExternalFilterId)
  return id == null ? null : fabs?.rows[String(Math.round(id))]?.filter ?? null
}

/**
 * ftir_39's per-filter rows with `residual_ugm3` and `abs_error_ugm3` against Fabs/`mac` (the notebook's are against
 * Fabs/`from`). Null when any filter has no Fabs: a panel mixing MACs would be worse than one left at the notebook's.
 */
export function perFilterAtMac(rows: readonly Row[], fabs: HoldoutFabs | null | undefined, mac: number, from = 10): Row[] | null {
  if (mac === from) return [...rows]
  const out: Row[] = []
  for (const r of rows) {
    const f = holdoutFabs(r, fabs), res = num(r.residual_ugm3)
    if (f == null || res == null) return null
    const moved = res + f / from - f / mac
    out.push({ ...r, residual_ugm3: moved, abs_error_ugm3: Math.abs(moved) })
  }
  return out
}

/** the larger of a filter's T² and Q source percentiles: ftir_39's distance for ρ and the 95 % flag */
export const domainPercentile = (r: Row): number | null => {
  const t = num(r.T2_source_percentile), q = num(r.Q_source_percentile)
  return t == null || q == null ? null : Math.max(t, q)
}

const rmse = (v: number[]) => (v.length ? Math.sqrt(v.reduce((s, x) => s + x * x, 0) / v.length) : null)

/**
 * ftir_39's domain summary rebuilt from per-filter rows, over each saved summary row (its other columns kept): n, the
 * percent outside the source 95 %, RMSE of the residuals inside and outside it, and Spearman ρ (with its t-test p)
 * of the domain percentile against |error|. At the notebook's MAC, on its filters, this reproduces the saved table;
 * on fewer (Filters → excluded) it is the same summary of those that count.
 */
export function domainSummaryFrom(perFilter: readonly Row[], summary: readonly Row[]): Row[] {
  return summary.map((s) => {
    const g = perFilter.filter((r) => r.config === s.config && r.target === s.target)
    const inside = g.filter((r) => r.outside_source_95 !== true).map((r) => num(r.residual_ugm3)!)
    const outside = g.filter((r) => r.outside_source_95 === true).map((r) => num(r.residual_ugm3)!)
    const pairs = g.map((r) => [domainPercentile(r), num(r.abs_error_ugm3)] as const).filter((p): p is readonly [number, number] => p[0] != null && p[1] != null)
    const rho = spearman(pairs.map((p) => p[0]), pairs.map((p) => p[1]))
    return { ...s, n: g.length, outside_source_95_pct: g.length ? (100 * outside.length) / g.length : null,
      RMSE_inside: rmse(inside), RMSE_outside: rmse(outside), rho_domain_percentile_vs_abs_error: rho,
      rho_p: rho == null ? null : correlationP(rho, pairs.length) }
  })
}

/** each holdout's HIPS BC (Fabs/`from`) mean and population variance, over its distinct filters */
export function holdoutMoments(perFilter: readonly Row[], fabs: HoldoutFabs | null | undefined, from = 10): Map<string, { mean: number; variance: number }> {
  const byTarget = new Map<string, Map<string, number>>()
  for (const r of perFilter) {
    const f = holdoutFabs(r, fabs)
    if (f == null) continue
    const t = String(r.target)
    if (!byTarget.has(t)) byTarget.set(t, new Map())
    byTarget.get(t)!.set(String(r.ExternalFilterId), f / from)
  }
  return new Map([...byTarget].map(([t, m]) => {
    const x = [...m.values()], mean = x.reduce((s, v) => s + v, 0) / x.length
    return [t, { mean, variance: x.reduce((s, v) => s + (v - mean) ** 2, 0) / x.length }]
  }))
}
