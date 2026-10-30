/**
 * The focus-site sides of the IMPROVE loading-envelope summaries (gallery/data/export_research_improve_envelope.py
 * `matched` and `mass_space`), recomputed from the gallery's rows and the HIPS optics export, so they follow
 * Filters → excluded as every chart does. On the rows the exporter used (the export's exclusions applied, nothing
 * excluded by hand) they are its saved rows (envelope.test.mjs). The IMPROVE sides need the FED exports and stay saved.
 */
import { ols, quantile } from './stats.ts'

/** the exporter's quantiles (QUANTILES: 5, 25, 50, 75, 95 %), numpy's linear rule */
export interface Quantiles { n: number; p05: number | null; p25: number | null; p50: number | null; p75: number | null; p95: number | null }
export function quantilesOf(values: readonly (number | null | undefined)[]): Quantiles {
  const v = values.filter((x): x is number => typeof x === 'number' && Number.isFinite(x))
  const q = (p: number) => (v.length ? quantile(v, p) : null)
  return { n: v.length, p05: q(0.05), p25: q(0.25), p50: q(0.5), p75: q(0.75), p95: q(0.95) }
}

/** one focus-site filter as the exporter's `spartan_reference` keeps it: a PM2.5 filter with Fabs and FTIR EC > 0 */
export interface FocusFilter { code: string; fabs: number; ecConc: number; ecUg: number | null; ocUg: number | null; volume: number | null; area: number | null }

/** the gallery row and optics fields a FocusFilter is read from (µg/m³ rows) */
export function focusFilter(row: { code: string; [field: string]: unknown }, o: { type: string; fabs: number | null; ec_ug: number | null; oc_ug: number | null; volume_m3: number | null; area_cm2: number | null }): FocusFilter | null {
  const ec = row['EC (FTIR)']
  if (o.type !== 'PM2.5' || o.fabs === null || !(o.fabs > 0) || typeof ec !== 'number' || !(ec > 0)) return null
  return { code: row.code, fabs: o.fabs, ecConc: ec, ecUg: o.ec_ug, ocUg: o.oc_ug, volume: o.volume_m3, area: o.area_cm2 }
}

export interface MatchedGroup { group: string; fabs: Quantiles; fabs_ec: Quantiles; ec_conc: Quantiles; ec_ug: Quantiles; fit: { n: number; slope: number | null; intercept: number | null; r2: number | null } }

/** `matched`'s Addis row: Fabs, Fabs / EC, EC and EC mass quantiles and OLS of Fabs on EC over the Addis filters */
export function addisMatched(filters: readonly FocusFilter[], group = 'ETAD (Addis)'): MatchedGroup {
  const a = filters.filter((f) => f.code === 'ETAD')
  const fit = ols(a.map((f) => f.ecConc), a.map((f) => f.fabs))
  return { group, fabs: quantilesOf(a.map((f) => f.fabs)), fabs_ec: quantilesOf(a.map((f) => f.fabs / f.ecConc)), ec_conc: quantilesOf(a.map((f) => f.ecConc)),
    ec_ug: quantilesOf(a.map((f) => f.ecUg)), fit: { n: fit?.n ?? a.length, slope: fit?.slope ?? null, intercept: fit?.intercept ?? null, r2: fit?.r2 ?? null } }
}

export interface MassRow extends Quantiles { group: string; quantity: string }

/** `mass_space`'s rows for each focus site: EC and OC mass on the filter (µg) and Fabs × V / A (A defaults to 3.53 cm²) */
export function siteMassSpace(filters: readonly FocusFilter[], areaDefault = 3.53): MassRow[] {
  const codes = [...new Set(filters.map((f) => f.code))].sort()
  return codes.flatMap((code) => {
    const g = filters.filter((f) => f.code === code)
    const q: [string, (number | null)[]][] = [
      ['EC mass (µg)', g.map((f) => f.ecUg)], ['OC mass (µg)', g.map((f) => f.ocUg)],
      ['fAbs × V / A', g.map((f) => (f.volume === null ? null : (f.fabs * 1e-6 * f.volume) / ((f.area ?? areaDefault) * 1e-4)))],
    ]
    return q.map(([quantity, v]) => ({ group: code, quantity, ...quantilesOf(v) }))
  })
}
