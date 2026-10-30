/**
 * The decision workbook's §6 site tables (decision_workbook_20261022 cell [23], site_season_table), rebuilt from
 * its per-filter export so they follow the reader's MAC: HIPS BC = Fabs / MAC, the FTIR ÷ HIPS medians and the
 * Deming fit of published FTIR EC on HIPS BC with the workbook's Addis error ratio (λ ∝ MAC²). At MAC 10 they
 * reproduce section6_sites.csv and section6_site_seasons.csv (decision.test.mjs). Node-tested, so relative imports.
 */
import { deming, median } from '../../../research/common/stats.ts'
import { demingLambda } from '../../../../lib/macMath.ts'
import { spearman } from '../../../../lib/stats.ts'

export interface FlagFilter {
  id: string
  code: string
  season: string
  flagSpectrum: number
  flagHips: number
  outside: boolean
  /** this workbook's spline model and SPARTAN's published FTIR EC (µg/m³) */
  ecModel: number | null
  ecPublished: number | null
  /** HIPS absorption (Mm⁻¹) */
  fabs: number | null
}

export interface FlagGroup {
  code: string
  /** '' for a whole site */
  season: string
  n: number
  meanFlagSpectrum: number
  meanFlagHips: number
  /** share with the + HIPS flag ≥ flagAt (the notebook's), and with the spectrum-only flag ≥ flagAt (null from a saved table) */
  flaggedShare: number
  flaggedShareSpectrum: number | null
  outsideShare: number
  /** medians over the filters with HIPS and published EC */
  modelRatio: number | null
  publishedRatio: number | null
  /** paired filters behind the ratios and the Deming fit */
  pairs: number
  slope: number | null
  intercept: number | null
  /** intercept ÷ mean published EC of the paired filters */
  interceptRatio: number | null
}

const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length
const finite = (v: number | null | undefined): v is number => v != null && Number.isFinite(v)

/** One group's row of the notebook's site_season_table at `mac` (λ10 the workbook's LAMBDA_ADDIS at MAC 10). */
export function summarise(code: string, season: string, g: FlagFilter[], mac: number, lambda10: number, flagAt = 0.5, minPairs = 15): FlagGroup {
  const ok = g.filter((f) => finite(f.fabs) && f.fabs > 0 && finite(f.ecPublished))
  const x = ok.map((f) => f.fabs! / mac), y = ok.map((f) => f.ecPublished!)
  const ratio = (pick: (f: FlagFilter) => number | null) => {
    const r = ok.map((f, i) => (finite(pick(f)) ? pick(f)! / x[i] : NaN)).filter(Number.isFinite)
    return r.length ? median(r) : null
  }
  const fit = ok.length >= minPairs ? deming(x, y, demingLambda(mac, lambda10)) : null
  return {
    code, season, n: g.length,
    meanFlagSpectrum: mean(g.map((f) => f.flagSpectrum)), meanFlagHips: mean(g.map((f) => f.flagHips)),
    flaggedShare: g.filter((f) => f.flagHips >= flagAt).length / g.length, flaggedShareSpectrum: g.filter((f) => f.flagSpectrum >= flagAt).length / g.length,
    outsideShare: g.filter((f) => f.outside).length / g.length,
    modelRatio: ratio((f) => f.ecModel), publishedRatio: ratio((f) => f.ecPublished), pairs: ok.length,
    slope: fit?.slope ?? null, intercept: fit?.intercept ?? null, interceptRatio: fit ? fit.intercept / mean(y) : null,
  }
}

/** By site, or by site and season (the notebook keeps site-seasons with at least `minN` filters), most flagged first. */
export function flagTable(filters: FlagFilter[], by: 'site' | 'site-season', mac: number, lambda10: number, minN = 10): FlagGroup[] {
  const groups = new Map<string, FlagFilter[]>()
  for (const f of filters) {
    const key = by === 'site' ? f.code : `${f.code}\u0000${f.season}`
    const g = groups.get(key)
    if (g) g.push(f)
    else groups.set(key, [f])
  }
  return [...groups.entries()].flatMap(([key, g]) => {
    const [code, season = ''] = key.split('\u0000')
    return by === 'site-season' && g.length < minN ? [] : [summarise(code, season, g, mac, lambda10)]
  }).sort((a, b) => b.meanFlagHips - a.meanFlagHips)
}

/**
 * The notebook's cross-check over site-seasons with a Deming fit: Spearman of the spectrum-only flag against the
 * Deming intercept ÷ mean, and against |ln (published ÷ HIPS)| (the second moves with the MAC; the first does not).
 */
export function crossCheck(siteSeasons: FlagGroup[]) {
  const ok = siteSeasons.filter((g) => finite(g.interceptRatio) && finite(g.publishedRatio) && g.publishedRatio > 0)
  const flag = ok.map((g) => g.meanFlagSpectrum)
  return {
    n: ok.length,
    intercept: spearman(flag, ok.map((g) => g.interceptRatio!)),
    ratio: spearman(flag, ok.map((g) => Math.abs(Math.log(g.publishedRatio!)))),
  }
}
