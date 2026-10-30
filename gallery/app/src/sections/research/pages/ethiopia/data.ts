/**
 * The joins and scores behind the Ethiopia calibrations page's July07 / spartan_ec additions, as pure functions
 * (node-tested in data.test.mjs against the notebooks' saved tables):
 *
 *  - SPARTAN's lot-251 biomass ("smoke") tool calibration against the regular ones at Addis, joined to the
 *    gallery's Addis rows so HIPS = Fabs / MAC follows the reader's MAC and the rows in view (exclusions, dates);
 *  - the five Adama PTFE filters against the date-paired quartz TOR, under every calibration tried on them;
 *  - spartan_ec 08's residual diagnostics on the 906 calibration filters.
 *
 * Naming: July07's "regular CAL-0" is the 906-filter IMPROVE set refitted at k = 20 (the gallery's
 * "Biomass-smoke (906)" preset trains on the same set). July07's "smoke" and spartan_ec's "biomass tool" are a
 * different model: the coefficients SPARTAN's calibration tool exported for lot 251.
 */
import { median, ols } from '../../common/stats.ts'

type Value = string | number | boolean | null
type Row = Record<string, Value>
const num = (v: Value | undefined): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

/** the gallery's row id for a SPARTAN ExternalFilterId: ETAD-0017-1 → ETAD-0017 (the scan suffix dropped) */
export const galleryId = (external: string) => external.replace(/-\d+$/, '')

/** the calibrations July07 compares, in its order; `reported` is the gallery row's own EC (FTIR) */
export const SMOKE_CALS = [
  { key: 'reported', label: 'Deployed general (reported EC)' },
  { key: 'cal0', label: 'Regular CAL-0 (906-filter IMPROVE set, k = 20)' },
  { key: 'smoke', label: 'Lot-251 biomass tool (07-07 "smoke")' },
] as const
export type SmokeCal = (typeof SMOKE_CALS)[number]['key']

/** the three ways the notebooks pick the Addis filters */
export const SMOKE_SETS = {
  filter: 'Addis filters (date join, 07-07 before-CAL-0)',
  spectrum: 'ETAD spectra (filter-id join, 07-07 regular vs smoke)',
  lot251: 'Lot-251 filters only (spartan_ec 02)',
} as const
export type SmokeSet = keyof typeof SMOKE_SETS

export interface SmokePoint { set: SmokeSet; id: string; date: string; scan?: string; hips: number; reported: number; cal0: number; smoke: number }

/** the gallery fields a join reads */
export interface AddisRow { id: string; code: string; date: string; [field: string]: Value }

/**
 * One point per Addis filter (07-07 before-CAL-0: its table is keyed on date, the two scans of a filter already
 * averaged), per ETAD spectrum (07-07 regular vs smoke: keyed on ExternalFilterId, one point per scan), and per
 * filter in spartan_ec 02's lot-251 set. HIPS and the reported EC come from the gallery row; a point needs all
 * three ECs, so every calibration in a set is scored on the same filters, as the notebooks do.
 */
export function smokePoints(rows: readonly AddisRow[], tables: { addis: readonly Row[]; etad: readonly Row[]; lot251: readonly Row[] }, mac: number): SmokePoint[] {
  const addis = rows.filter((r) => r.code === 'ETAD')
  const byId = new Map(addis.map((r) => [r.id, r]))
  const byDate = new Map(tables.addis.map((t) => [String(t.date), t]))
  const lot = new Set(tables.lot251.map((t) => galleryId(String(t.FilterId))))
  const out: SmokePoint[] = []
  const push = (set: SmokeSet, r: AddisRow, cal0: number | null, smoke: number | null, scan?: string) => {
    const fabs = num(r['HIPS Fabs']), reported = num(r['EC (FTIR)'])
    if (fabs === null || reported === null || cal0 === null || smoke === null) return
    out.push({ set, id: r.id, date: r.date, scan, hips: fabs / mac, reported, cal0, smoke })
  }
  for (const r of addis) {
    const t = byDate.get(r.date)
    if (!t) continue
    push('filter', r, num(t.ec_cal0), num(t.ec_smoke))
    if (lot.has(r.id)) push('lot251', r, num(t.ec_cal0), num(t.ec_smoke))
  }
  for (const t of tables.etad) {
    const r = byId.get(galleryId(String(t.ExternalFilterId)))
    if (r) push('spectrum', r, num(t.EC_regular), num(t.EC_smoke), `${t.ExternalFilterId} · MediaId ${t.MediaId}`)
  }
  return out
}

export interface SmokeScore {
  set: string; calibration: string; n: number
  /** OLS of EC on HIPS BC (07-07 regular vs smoke's orientation) */
  slope: number | null; intercept: number | null; r2: number | null
  /** OLS of HIPS BC on EC (07-07 before-CAL-0's orientation, FTIR EC on x) */
  slopeHipsOnEc: number | null; interceptHipsOnEc: number | null
  /** |slope − 1|, the notebooks' distance from the 1:1 */
  offOne: number | null
  /** median HIPS BC − EC */
  gap: number | null
  /** the MAC at which this calibration's slope against Fabs / MAC would be 1 (07-07's MAC / slope) */
  impliedMac: number | null
}

/** July07's scorecard, per set and calibration, at the reader's MAC */
export function smokeScores(points: readonly SmokePoint[], mac: number): SmokeScore[] {
  const out: SmokeScore[] = []
  for (const set of Object.keys(SMOKE_SETS) as SmokeSet[]) {
    const ps = points.filter((p) => p.set === set)
    for (const c of SMOKE_CALS) {
      const x = ps.map((p) => p.hips), y = ps.map((p) => p[c.key])
      const f = ols(x, y), inv = ols(y, x)
      out.push({ set: SMOKE_SETS[set], calibration: c.label, n: f?.n ?? 0,
        slope: f?.slope ?? null, intercept: f?.intercept ?? null, r2: f?.r2 ?? null,
        slopeHipsOnEc: inv?.slope ?? null, interceptHipsOnEc: inv?.intercept ?? null,
        offOne: f ? Math.abs(f.slope - 1) : null, gap: ps.length ? median(x.map((v, i) => v - y[i])) : null,
        impliedMac: f && f.slope > 0 ? mac / f.slope : null })
    }
  }
  return out
}

// ---------------------------------------------------------------- Adama

export interface AdamaPoint {
  calibration: string; source: string
  /** PTFE / quartz filter ids, and their shared sample date */
  ptfe: string; quartz: string; pair: string; date: string
  /** FTIR EC and TOR EC per m³ (µg/m³) and on each filter (µg) */
  ftirConc: number; torConc: number; ftirMass: number; torMass: number
  ratioConc: number; ratioMass: number
}

/**
 * Every Adama calibration on one footing. The June 2026 notebooks (02, 08, 15) give µg/m³; the spartan_ec ones
 * give µg on the filter. µg ÷ the PTFE volume June 08 saved is June's µg/m³ (the exporter checks it on the general
 * EC), and TOR is the date-paired quartz filter's ECTR in either unit (June 08's TOR_EC, spartan_ec 03's EC_TOR).
 * CAL-2 and CAL-4 (tiny-sample fits that predict about −60 to −190 µg/m³ here) are left out.
 */
export function adamaPoints(t: { variants: readonly Row[]; tor: readonly Row[]; han: readonly Row[]; june15: readonly Row[];
  calibration: readonly Row[]; faithful: readonly Row[]; optimized: readonly Row[]; stabilized: readonly Row[] }): AdamaPoint[] {
  const quartzOf = new Map(t.tor.map((r) => [String(r.ptfe_FilterId), String(r.quartz_FilterId)]))
  const torMass = new Map(t.han.map((r) => [String(r.sample), num(r.EC_TOR)]))
  const ptfeOfBarcode = new Map(t.variants.map((r) => [String(r.barcode), String(r.FilterId)]))
  const base = new Map(t.variants.map((r) => [String(r.FilterId), r]))
  const out: AdamaPoint[] = []
  const add = (calibration: string, source: string, rows: readonly Row[], key: (r: Row) => string, col: string, unit: 'conc' | 'mass') => {
    for (const r of rows) {
      const ptfe = key(r), v = base.get(ptfe), value = num(r[col])
      const quartz = quartzOf.get(ptfe) ?? '?'
      const volume = num(v?.volume_m3), tc = num(v?.TOR_EC), tm = torMass.get(quartz) ?? null
      if (!v || value === null || volume === null || tc === null || tm === null) continue
      const ftirConc = unit === 'conc' ? value : value / volume, ftirMass = unit === 'mass' ? value : value * volume
      out.push({ calibration, source, ptfe, quartz, pair: `${ptfe} / ${quartz}`, date: String(v.date), ftirConc, torConc: tc, ftirMass, torMass: tm,
        ratioConc: ftirConc / tc, ratioMass: ftirMass / tm })
    }
  }
  const fid = (r: Row) => String(r.FilterId)
  add('Deployed general (lot 241a)', 'June 02 / 08, the SPARTAN tool\'s general calibration', t.variants, fid, 'FTIR_EC_general_orig', 'conc')
  for (const c of ['CAL-0 all-nofilter', 'CAL-1 cleaned', 'CAL-3 below-1:1', 'CAL-5 Eth-range']) add(c, 'June 08 (906 set, k = 20)', t.variants, fid, c, 'conc')
  add('NEW lot-248+251 (15, k10)', 'June 15, flag-filtered lot-248+251 mirror', t.june15, (r) => ptfeOfBarcode.get(String(r.barcode)) ?? '', 'NEW lot-248+251', 'conc')
  add('Lot-251 biomass tool', 'spartan_ec adama comparison; July07\'s "smoke"', t.calibration, fid, 'EC_biomass_tool', 'mass')
  add('Local lot-251 biomass + σ', 'spartan_ec adama comparison', t.calibration, fid, 'EC_biomass_local_sig', 'mass')
  add('Tool replica (faithful)', 'spartan_ec faithful replica', t.faithful, fid, 'EC_biomass_faithful', 'mass')
  add('Tool replica + σ', 'spartan_ec faithful replica', t.faithful, fid, 'EC_biomass_faithful_sig', 'mass')
  add('Biomass baseline (05, K 31)', 'spartan_ec 05', t.optimized, fid, 'EC_biomass_baseline', 'mass')
  add('Biomass optimized (05)', 'spartan_ec 05', t.optimized, fid, 'EC_biomass_optimized', 'mass')
  add('Biomass stabilized (07)', 'spartan_ec 07', t.stabilized, fid, 'EC_biomass', 'mass')
  return out
}

// ---------------------------------------------------------------- spartan_ec 08 residuals

export interface ResidualDiagnostics {
  /** measured − predicted on the full 906-filter fit at k = 20 (µg), with each filter's id and status */
  points: { id: string; site: string; residual: number; removed: boolean; round: number }[]
  /** numpy's std (ddof 0) of the full-fit residuals, and the 3σ line 08 draws */
  sigma: number
  threshold: number
  /** filters beyond the full-fit 3σ line (08's first cleaning round drops exactly these) */
  beyond: number
}

export function residualDiagnostics(rows: readonly Row[]): ResidualDiagnostics | null {
  const points = rows.flatMap((r) => {
    const m = num(r.measured), p = num(r.predicted)
    return m === null || p === null ? [] : [{ id: String(r.AnalysisId), site: String(r.Site), residual: m - p, removed: r.status !== 'kept', round: num(r.round) ?? 0 }]
  })
  if (points.length < 3) return null
  const mean = points.reduce((a, p) => a + p.residual, 0) / points.length
  const sigma = Math.sqrt(points.reduce((a, p) => a + (p.residual - mean) ** 2, 0) / points.length)
  return { points, sigma, threshold: 3 * sigma, beyond: points.filter((p) => Math.abs(p.residual) > 3 * sigma).length }
}
