/**
 * The arithmetic of Research → Python PMF (pages/PythonPmf.tsx), as pure functions over numbers so node can test them
 * against the package's saved results (pythonPmf.test.mjs). The PMF fits themselves are the package's; what the
 * browser redoes is closed-form: the held-out total-OM scores (scripts/validate_mass_prediction.py's pooled RMSE, MAE,
 * R² = 1 − SSE/SST, Pearson r, mean bias), the Pearson r of a factor's contributions against an archived KF factor or
 * a gallery measurement, and the breaks a time series takes at long sampling gaps.
 */

/** ETAD-0017-1 → ETAD-0017: the gallery's row id for a SPARTAN filter id (lib/filterId baseFilterId, kept local for node) */
export const baseId = (id: string): string => id.replace(/^([A-Z]{4}-\d{4})-\d+$/, '$1')

export interface Scores { n: number; rmse: number; mae: number; r2: number; r: number; bias: number; sse: number }

/** prediction scores of `pred` against `obs` (finite pairs only); null below 2 pairs */
export function scores(obs: readonly number[], pred: readonly number[]): Scores | null {
  const o: number[] = [], p: number[] = []
  for (let i = 0; i < Math.min(obs.length, pred.length); i++) if (Number.isFinite(obs[i]) && Number.isFinite(pred[i])) { o.push(obs[i]); p.push(pred[i]) }
  const n = o.length
  if (n < 2) return null
  let sse = 0, sae = 0, bias = 0, mo = 0, mp = 0
  for (let i = 0; i < n; i++) { const e = p[i] - o[i]; sse += e * e; sae += Math.abs(e); bias += e; mo += o[i]; mp += p[i] }
  mo /= n; mp /= n
  let sst = 0, spp = 0, sop = 0
  for (let i = 0; i < n; i++) { const a = o[i] - mo, b = p[i] - mp; sst += a * a; spp += b * b; sop += a * b }
  return { n, rmse: Math.sqrt(sse / n), mae: sae / n, r2: sst > 0 ? 1 - sse / sst : NaN, r: sst > 0 && spp > 0 ? sop / Math.sqrt(sst * spp) : NaN, bias: bias / n, sse }
}

/** Pearson r over the finite pairs; null below 3 pairs or with no spread */
export function pearson(xs: readonly number[], ys: readonly number[]): number | null {
  const x: number[] = [], y: number[] = []
  for (let i = 0; i < Math.min(xs.length, ys.length); i++) if (Number.isFinite(xs[i]) && Number.isFinite(ys[i])) { x.push(xs[i]); y.push(ys[i]) }
  const n = x.length
  if (n < 3) return null
  const mx = x.reduce((a, b) => a + b, 0) / n, my = y.reduce((a, b) => a + b, 0) / n
  let sxx = 0, syy = 0, sxy = 0
  for (let i = 0; i < n; i++) { const a = x[i] - mx, b = y[i] - my; sxx += a * a; syy += b * b; sxy += a * b }
  return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : null
}

/** one five-factor case's per-filter contributions, as python_pmf.json carries them */
export interface CaseContributions { id: string[]; date: string[]; g: (number | null)[][]; rmse: (number | null)[]; gt3: (number | null)[] }

/** the archived (KF) solution's per-filter rows, as public/data/pmf.json carries them: µg/m³ per historical factor label */
export interface ArchivedRow { id: string; ugm3?: Record<string, number | null> }

/**
 * Each new factor against the archived KF factor the package matched it to (analysis/historical_factor_pattern_matches.csv),
 * on the filters both solutions hold and `counts` keeps: Pearson r of the new contribution against the archived µg/m³.
 * `kfLabel(k)` names archived factor k (1-based) in the archived rows.
 */
export function archivedMatch(c: CaseContributions, archived: readonly ArchivedRow[], pairs: readonly { factor: number; kf: number }[],
  kfLabel: (kf: number) => string, counts: (id: string) => boolean = () => true): { factor: number; kf: number; n: number; r: number | null }[] {
  const byId = new Map(archived.map((a) => [a.id, a]))
  return pairs.map(({ factor, kf }) => {
    const x: number[] = [], y: number[] = []
    c.id.forEach((sid, i) => {
      const b = baseId(sid), a = byId.get(b), g = c.g[factor - 1]?.[i]
      const v = a?.ugm3?.[kfLabel(kf)]
      if (!a || !counts(b) || g == null || v == null || !Number.isFinite(g) || !Number.isFinite(v)) return
      x.push(g); y.push(v)
    })
    return { factor, kf, n: x.length, r: pearson(x, y) }
  })
}

/** the indices after which a dated series breaks (the next sample is more than `days` later): long gaps are not joined */
export function gapAfter(dates: readonly string[], days: number): Set<number> {
  const out = new Set<number>()
  for (let i = 0; i + 1 < dates.length; i++) {
    const a = Date.parse(dates[i]), b = Date.parse(dates[i + 1])
    if (Number.isFinite(a) && Number.isFinite(b) && (b - a) / 86_400_000 > days) out.add(i)
  }
  return out
}

/** the unavoidable scaled RMSE from negative observations alone: sqrt(Σ_{X<0} (X/U)² / cells) (report §4) */
export const negativeRmseBound = (negativeQ: number, cells: number): number => Math.sqrt(negativeQ / cells)

/**
 * Where a filter of the package's QC inventory stands against the gallery: modelled or held by the PMF, against
 * counted, excluded (by the export or by hand) or absent from the gallery rows.
 */
export type GalleryStand = 'counted' | 'excluded' | 'absent'
export interface QcCell { modelled: boolean; gallery: GalleryStand }
export function qcCell(modelled: boolean, known: boolean, counted: boolean): QcCell {
  return { modelled, gallery: !known ? 'absent' : counted ? 'counted' : 'excluded' }
}
/** the two disagreements: the PMF holds a filter the gallery counts, or models one the gallery excludes */
export const disagrees = (c: QcCell): boolean => (c.modelled ? c.gallery === 'excluded' : c.gallery === 'counted')
