/**
 * Where a site sits in a reference network's ranking (charts/ranking/NetworkRankPlacement): the reference sites
 * ranked by the median of a field, or of a per-filter ratio of two fields, and each site in view placed at the rank
 * its own median would take in that order. Pure functions, node-tested (referenceRank.test.mjs).
 *
 * The reference is IMPROVE as per-site quantiles (improve_site_quantiles.json, gallery/data/export_improve_site_quantiles.py),
 * because ~626,000 filters cannot be shipped. A ratio's quantiles there come from per-filter ratios, and so do the
 * sites in view here, never a ratio of the two fields' quantiles. HIPS BC = Fabs / MAC is not exported: its
 * quantiles, and those of any ratio with it, are Fabs's scaled by the reader's MAC (a positive constant moves every
 * quantile alike), so the reference follows the MAC as the rows in view already do (lib/viewFields).
 *
 * A field in view stands for the reference's measurement of the same quantity: SPARTAN's FTIR EC and OC for
 * IMPROVE's thermal-optical (TOR) EC and OC, HIPS Fabs for IMPROVE's fAbs (also HIPS). Where the methods differ the
 * placement compares methods as much as aerosols, and `methods` says so for the chart's note.
 *
 * Populations (both sides apply the same rule):
 *  - paired: filters with OC, EC and Fabs all > 0, run_ftir_51 §11 (figures 11a and 11b);
 *  - own:    a field wherever it is reported; a ratio wherever both its parts are > 0.
 */

export type Population = 'paired' | 'own'

export interface SiteQuantiles { site: number[]; n: number[]; p25: (number | null)[]; p50: (number | null)[]; p75: (number | null)[] }

/** improve_site_quantiles.json */
export interface ReferenceFile {
  schema_version: number
  generated: string
  source: string
  note: string
  data_use: string
  min_n: number
  populations: Record<Population, { label: string; rule: string; n_filters: number; n_sites: number }>
  fields: Record<string, { column: string; unit: string }>
  sites: string[]
  site_names: Record<string, string>
  quantities: Record<Population, Record<string, SiteQuantiles>>
}

/** a reference measurement a field in view stands for: its name in the file, the factor that turns it into the field, the method gap */
interface Stand { ref: string; perMac?: boolean; method?: string }
const FTIR = (kind: 'EC' | 'OC'): Stand => ({ ref: `${kind} (TOR)`, method: `${kind}: FTIR in view, thermal-optical (TOR) in IMPROVE` })
const FABS_BC = (what: string): Stand => ({ ref: 'HIPS Fabs', perMac: true, method: `${what} = Fabs / MAC on both sides` })

/** field in view → IMPROVE measurement of the same quantity */
export const REFERENCE_OF: Record<string, Stand> = {
  'EC (TOR)': { ref: 'EC (TOR)' },
  'OC (TOR)': { ref: 'OC (TOR)' },
  'EC (FTIR)': FTIR('EC'),
  'EC (ChemSpec FTIR)': FTIR('EC'),
  'EC (FTIR, VIBES)': FTIR('EC'),
  'EC (FTIR, spline)': FTIR('EC'),
  'OC (FTIR)': FTIR('OC'),
  'OC (ChemSpec FTIR)': FTIR('OC'),
  'HIPS Fabs': { ref: 'HIPS Fabs' },
  'HIPS BC': FABS_BC('HIPS BC'),
  'BC (public HIPS)': FABS_BC('public HIPS BC'),
  'PM2.5 mass': { ref: 'PM2.5 mass' },
  'Iron': { ref: 'Iron' },
  'Soil': { ref: 'Soil' },
}
/** fields measured in µg/m³ in the reference (they become µg per filter in the mass view; Fabs does not) */
const CONCENTRATION = new Set(['EC (TOR)', 'OC (TOR)', 'PM2.5 mass', 'Iron', 'Soil'])

export interface Quantity {
  /** the field in view: one field, or a ratio `num / den` (lib/viewFields ratioKey) */
  field: string
  parts: string[]
  /** the reference quantity's key in the file */
  key: string
  /** what the reference ranks, in its own names (HIPS BC where the field is Fabs / MAC) */
  label: string
  /** the reference's quantiles × this give the field at the reader's MAC */
  scale: number
  /** µg/m³ that do not cancel: +1 a concentration, 0 a ratio of two (or none), −1 per µg/m³ */
  concPower: number
  /** where the two sides measure a part by different methods */
  methods: string[]
}

/** the parts of a ratio field (`A / B`), or the field alone */
export function partsOf(field: string, known: (f: string) => boolean = (f) => Object.prototype.hasOwnProperty.call(REFERENCE_OF, f)): string[] {
  const at = field.indexOf(' / ')
  if (at > 0) {
    const num = field.slice(0, at), den = field.slice(at + 3)
    if (known(num) && known(den)) return [num, den]
  }
  return [field]
}

/** The reference quantity a field in view stands for at `mac`, or why there is none. */
export function referenceQuantity(field: string, mac: number): Quantity | { reason: string } {
  const parts = partsOf(field)
  const stands = parts.map((p) => REFERENCE_OF[p])
  const missing = parts.filter((_, i) => !stands[i])
  if (missing.length) return { reason: `IMPROVE has no measurement standing for ${missing.join(' or ')}` }
  const [a, b] = stands
  if (b && a.ref === b.ref) return { reason: `${field} is the same IMPROVE measurement over itself` }
  const factor = (s: Stand) => (s.perMac ? 1 / mac : 1)
  const conc = (s: Stand) => (CONCENTRATION.has(s.ref) || s.perMac ? 1 : 0)
  return {
    field, parts,
    key: b ? `${a.ref} / ${b.ref}` : a.ref,
    label: stands.map((s) => (s.perMac ? 'HIPS BC' : s.ref)).join(' / '),
    scale: b ? factor(a) / factor(b) : factor(a),
    concPower: b ? conc(a) - conc(b) : conc(a),
    methods: [...new Set(stands.flatMap((s) => (s.method ? [s.method] : [])))],
  }
}

export interface RankedSite { code: string; name: string; n: number; p25: number; p50: number; p75: number }

/** The reference sites with at least `minN` filters, ranked by median (lowest first), at the reader's scale. */
export function rankedReference(file: ReferenceFile, pop: Population, q: Quantity, minN: number): RankedSite[] {
  const t = file.quantities[pop]?.[q.key]
  if (!t) return []
  const out: RankedSite[] = []
  t.site.forEach((s, i) => {
    const p25 = t.p25[i], p50 = t.p50[i], p75 = t.p75[i]
    if (t.n[i] < minN || p25 == null || p50 == null || p75 == null) return
    const code = file.sites[s]
    out.push({ code, name: file.site_names[code] ?? code, n: t.n[i], p25: p25 * q.scale, p50: p50 * q.scale, p75: p75 * q.scale })
  })
  // the export's order is by code; ties in the median keep it
  return out.sort((x, y) => x.p50 - y.p50)
}

type Row = Record<string, unknown>
const num = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

/** the carbon and absorption a filter must carry for the paired population, the quantity's own fields first */
const PAIR_CANDIDATES = { EC: ['EC (FTIR)', 'EC (TOR)', 'EC (ChemSpec FTIR)'], OC: ['OC (FTIR)', 'OC (TOR)', 'OC (ChemSpec FTIR)'] }
function pairedRow(r: Row, parts: string[]): boolean {
  const first = (kind: 'EC' | 'OC') => {
    const own = parts.find((p) => REFERENCE_OF[p]?.ref === `${kind} (TOR)`)
    for (const f of own ? [own, ...PAIR_CANDIDATES[kind]] : PAIR_CANDIDATES[kind]) if (num(r[f])) return r[f] as number
    return null
  }
  const ec = first('EC'), oc = first('OC'), fabs = r['HIPS Fabs']
  return ec !== null && ec > 0 && oc !== null && oc > 0 && num(fabs) && fabs > 0
}

/** One filter's value of the quantity under the population's rule, or null when it does not count. */
export function filterValue(r: Row, q: Pick<Quantity, 'parts'>, pop: Population): number | null {
  if (pop === 'paired' && !pairedRow(r, q.parts)) return null
  if (q.parts.length === 2) {
    const a = r[q.parts[0]], b = r[q.parts[1]]
    return num(a) && num(b) && a > 0 && b > 0 ? a / b : null
  }
  const v = r[q.parts[0]]
  return num(v) ? v : null
}

/** type-7 (linear) quantile of sorted values: pandas' default and d3.quantileSorted */
export function quantileSorted(v: readonly number[], p: number): number {
  if (!v.length) return NaN
  const h = (v.length - 1) * p, lo = Math.floor(h)
  return lo + 1 < v.length ? v[lo] + (h - lo) * (v[lo + 1] - v[lo]) : v[lo]
}

export interface PlacedSite { key: string; n: number; p25: number; p50: number; p75: number; values: number[] }

/** Per-group quantiles of the quantity over rows (groups with at least `minN` filters). */
export function groupQuantiles<R extends Row>(rows: readonly R[], groupOf: (r: R) => string, q: Pick<Quantity, 'parts'>, pop: Population, minN: number): PlacedSite[] {
  const by = new Map<string, number[]>()
  for (const r of rows) {
    const v = filterValue(r, q, pop)
    if (v === null) continue
    const k = groupOf(r)
    const a = by.get(k)
    if (a) a.push(v)
    else by.set(k, [v])
  }
  const out: PlacedSite[] = []
  for (const [key, values] of by) {
    if (values.length < minN) continue
    values.sort((a, b) => a - b)
    out.push({ key, n: values.length, p25: quantileSorted(values, 0.25), p50: quantileSorted(values, 0.5), p75: quantileSorted(values, 0.75), values })
  }
  return out.sort((a, b) => a.p50 - b.p50)
}

/** How many reference medians lie below `median`: the 0-based position it takes (numpy searchsorted, side left). Rank = this + 1. */
export function placeAmong(ranked: readonly Pick<RankedSite, 'p50'>[], median: number): number {
  let lo = 0, hi = ranked.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (ranked[mid].p50 < median) lo = mid + 1
    else hi = mid
  }
  return lo
}
