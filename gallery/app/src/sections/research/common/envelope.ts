/**
 * The focus-site sides of the IMPROVE loading-envelope summaries (gallery/data/export_research_improve_envelope.py
 * `matched` and `mass_space`), recomputed from the gallery's rows and the HIPS optics export, so they follow
 * Filters → excluded as every chart does. On the rows the exporter used (the export's exclusions applied, nothing
 * excluded by hand) they are its saved rows (envelope.test.mjs). A filter the HIPS optics export does not hold (any
 * other SPARTAN site, a picked IMPROVE site) is read from its gallery row instead (`rowFilter`), so these follow the
 * Samples sites too. The network-wide IMPROVE sides need the FED exports and stay saved.
 */
import { ols, quantile } from '../../../lib/stats.ts'
import type { Optics } from './optics.ts'

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

/** a deposit area where the optics export gives none: SPARTAN's, every row of that export; IMPROVE's as the exporter takes it */
export const SPARTAN_AREA = 3.53, IMPROVE_AREA = 3.5

const positive = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : null)

/**
 * The same filter from its gallery row alone (µg/m³ rows): HIPS Fabs, EC by FTIR or, at an IMPROVE site, TOR, and
 * mass on the filter as concentration × sampled volume, which matches the optics export's mass within 0.1 % on the
 * focus filters. Null when the row lacks Fabs or EC. No volume leaves the masses null.
 */
export function rowFilter(row: { code: string; [field: string]: unknown }): FocusFilter | null {
  const ftir = positive(row['EC (FTIR)']), ec = ftir ?? positive(row['EC (TOR)']), fabs = positive(row['HIPS Fabs'])
  if (ec === null || fabs === null) return null
  const volume = positive(row['Sampled volume'])
  const oc = ftir !== null ? positive(row['OC (FTIR)']) : positive(row['OC (TOR)'])
  return { code: row.code, fabs, ecConc: ec, ecUg: volume === null ? null : ec * volume, ocUg: volume === null || oc === null ? null : oc * volume,
    volume, area: ftir !== null ? SPARTAN_AREA : IMPROVE_AREA }
}

/** the volume and deposit area of a filter the optics export does not hold: an IMPROVE row is the one with TOR EC and no FTIR EC */
export function rowDeposit(row: { [field: string]: unknown }): { volume_m3: number | null; area_cm2: number | null; lot: null } {
  const volume = positive(row['Sampled volume'])
  const improve = positive(row['EC (TOR)']) !== null && positive(row['EC (FTIR)']) === null
  return { volume_m3: volume, area_cm2: volume === null ? null : improve ? IMPROVE_AREA : SPARTAN_AREA, lot: null }
}

/**
 * An optics row built from a gallery row, for a filter of a site the HIPS export does not hold: Fabs and, with a volume,
 * tau = Fabs·V / (100·A), within 1.3e-4 of the export (its rounding) on all 546 samples with a reading. The raw readings (t, r, R1, T1),
 * the lot and the deployed line are only in the export, so they stay null and whatever needs them leaves this filter out.
 */
export function rowOptics(row: { id: string; code: string; date?: string | null; [field: string]: unknown }): Optics | null {
  const fabs = positive(row['HIPS Fabs'])
  if (fabs === null) return null
  const { volume_m3, area_cm2 } = rowDeposit(row)
  return { id: row.id, filter_id: row.id, code: row.code, type: 'PM2.5', date: row.date ?? null, lot: null, analysis_date: null,
    t: null, r: null, tau: volume_m3 !== null && area_cm2 !== null ? (fabs * volume_m3) / (100 * area_cm2) : null, T1: null, R1: null,
    cal_intercept: null, cal_slope: null, fabs, mdl: null, area_cm2, volume_m3, ec_ug: null, oc_ug: null, om_ug: null }
}

/**
 * The optics row of a gallery row: the export's where it has the filter, a stand-in (`rowOptics`) at a site it does not
 * cover. A filter the export lacks at one of its own sites gets none, as it always has, so the four focus sites read exactly
 * as the export does.
 */
export function opticsJoin(optics: readonly Optics[]) {
  const byId = new Map(optics.filter((o) => o.type !== 'FB').map((o) => [o.id, o]))
  const held = new Set(optics.map((o) => o.code))
  return (row: { id: string; code: string; [field: string]: unknown }): Optics | null => byId.get(row.id) ?? (held.has(row.code) ? null : rowOptics(row))
}

export interface MatchedGroup { group: string; fabs: Quantiles; fabs_ec: Quantiles; ec_conc: Quantiles; ec_ug: Quantiles; fit: { n: number; slope: number | null; intercept: number | null; r2: number | null } }

/** `matched`'s Addis row: Fabs, Fabs / EC, EC and EC mass quantiles and OLS of Fabs on EC over the Addis filters */
export const addisMatched = (filters: readonly FocusFilter[], group = 'ETAD (Addis)') => siteMatched(filters, 'ETAD', group)

/** the same row for any site in view */
export function siteMatched(filters: readonly FocusFilter[], code: string, group: string): MatchedGroup {
  const a = filters.filter((f) => f.code === code)
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
