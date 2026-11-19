/**
 * The reader's own view of the measurements, applied to every row before the charts see them
 * (App, then every chart tab): a MAC for HIPS BC other than the export's 10, mass per filter
 * instead of concentration, and ratios the reader defines (EC / PM2.5 mass, OC / EC …), which
 * then sit in every field picker like a measured field. Pure functions, node-tested.
 *
 * - MAC: HIPS BC = HIPS Fabs / MAC, and the public HIPS BC (Fabs / 10 by SPARTAN's convention)
 *   rescaled the same way. MAC multiplies the absorption axis, so it scales a fit's slope; the
 *   intercept is unchanged for OLS and moves only slightly for Deming (fixed error ratio).
 * - Mass per filter: every µg/m³ field × the filter's sampled volume (µg on the filter, what FTIR
 *   sees). IMPROVE samples about 33 m³ a day, SPARTAN about 7, so concentrations hide a ~4x
 *   difference in loading. A filter without a volume drops out of mass-per-filter charts.
 * - HIPS linearised: HIPS reads low on dark filters (the IMPROVE curve, lib/hipsLinearise). With a ceiling set, HIPS
 *   Fabs, HIPS BC and the public HIPS BC of each filter are multiplied by 1 / (1 − τ/τ∞), τ = Fabs × sampled volume
 *   ÷ 353. A filter without a sampled volume, or at or past the ceiling, has no linearised value and drops out of
 *   the HIPS fields rather than keep its measured one beside corrected filters. Units carry ", linearised".
 *   The view can cover every site or only those the bend affects (lib/hipsLinearise HIPS_SCOPES): a site outside
 *   keeps HIPS as measured, and the units then say which sites, ", linearised where seen".
 * - HIPS loading: every filter with HIPS and a sampled volume carries its optical depth and the under-read the
 *   IMPROVE curve expects of it, as fields of their own (withHipsLoading), so any chart can show, colour by or
 *   band which filters the bend affects, whether or not the view corrects them.
 * - Ratios: numerator / denominator on the same filter, after the steps above.
 */
import { correction, HIPS_SCOPES, opticalDepth, underRead, type HipsScope } from './hipsLinearise.ts'
import { baseUnit, HIPS_LINEARISED_MARK, LINEARISED_MARK, linearisedScope, prettyUnit } from './theme.ts'

export interface RatioDef { num: string; den: string }
export type Basis = 'conc' | 'mass'

export const RATIO_GROUP = 'Ratios'
export const VOLUME = 'Sampled volume'
export const MAC_CHOICES = [5, 7.5, 10, 12.5, 15, 17.5, 20] as const
/** fields recomputed from HIPS Fabs when the MAC changes */
const HIPS_BC = 'HIPS BC'
const HIPS_FABS = 'HIPS Fabs'
const PUBLIC_BC = 'BC (public HIPS)'
/** the fields the linearised view changes */
export const HIPS_VIEW_FIELDS: readonly string[] = [HIPS_FABS, HIPS_BC, PUBLIC_BC]
/** how dark a filter is to HIPS, and the share of its absorption the IMPROVE curve expects HIPS not to report (withHipsLoading) */
export const HIPS_TAU = 'HIPS optical depth'
export const HIPS_UNDER_READ = 'HIPS expected under-read'
export const HIPS_LOADING_GROUP = 'HIPS loading'
export const HIPS_LOADING_UNITS: Record<string, string> = { [HIPS_TAU]: '', [HIPS_UNDER_READ]: '%' }

export const ratioKey = (d: RatioDef) => `${d.num} / ${d.den}`

/** ratios the meeting asked about; offered in the dialog when both fields exist */
export const RATIO_PRESETS: RatioDef[] = [
  { num: 'EC (FTIR)', den: 'PM2.5 mass' },
  { num: 'OC (FTIR)', den: 'EC (FTIR)' },
  { num: 'OM (FTIR)', den: 'EC (FTIR)' },
  { num: 'HIPS BC', den: 'EC (FTIR)' },
  { num: 'HIPS BC', den: 'PM2.5 mass' },
  { num: 'EC (TOR)', den: 'PM2.5 mass' },
  { num: 'OC (TOR)', den: 'EC (TOR)' },
  { num: 'Iron', den: 'PM2.5 mass' },
]

/** `ratios` in the URL: num:den pairs, each part URI-encoded (so ':' and ',' never appear inside), joined by commas */
export function parseRatios(v: string | null): RatioDef[] {
  if (!v) return []
  const out: RatioDef[] = []
  for (const part of v.split(',')) {
    const [a, b] = part.split(':')
    if (!a || !b) continue
    try {
      const d = { num: decodeURIComponent(a), den: decodeURIComponent(b) }
      if (d.num !== d.den && !out.some((x) => ratioKey(x) === ratioKey(d))) out.push(d)
    } catch { /* a malformed pair is skipped */ }
  }
  return out
}

export const ratiosParam = (defs: RatioDef[]): string | null =>
  defs.length ? defs.map((d) => `${encodeURIComponent(d.num)}:${encodeURIComponent(d.den)}`).join(',') : null

/** a µg/m³ field, in any of the spellings the exports use */
export const isConcentration = (unit: string | undefined) => !!unit && /^(µg\/m³|ug\/m3|µg\/m3|ug\/m³)$/.test(unit.trim())

const tidy = prettyUnit

/** the unit of a ratio from its parts' units (already in the view's basis) */
export function ratioUnit(num: string | undefined, den: string | undefined): string {
  const a = tidy(baseUnit(num)), b = tidy(baseUnit(den))
  const unit = a === b ? 'ratio' : a && b ? `${a} per ${b}` : 'ratio'
  // a ratio of a linearised HIPS field says so, and at which sites
  return num !== baseUnit(num) || den !== baseUnit(den) ? unit + HIPS_LINEARISED_MARK + (linearisedScope(num) || linearisedScope(den)) : unit
}

/** µg on the filter (the mass-per-filter view), linearised or not */
export const isPerFilter = (unit: string | undefined) => baseUnit(unit) === 'µg/filter'

export interface ViewOptions {
  mac: number
  /** the MAC the exports used (meta.mac_value) */
  exportMac: number
  basis: Basis
  /** field → unit, before this view */
  units: Record<string, string>
  ratios: RatioDef[]
  /** linearise HIPS with this ceiling τ∞ (lib/hipsLinearise); null or absent: HIPS as measured */
  hipsCeiling?: number | null
  /** which sites that covers (lib/hipsLinearise HIPS_SCOPES): 'all' or absent, every site */
  hipsScope?: HipsScope
  /** the site codes it covers under that scope (hipsLinearise scopeSites); null or absent: every site */
  hipsSites?: ReadonlySet<string> | null
}

/** field units after the view: concentrations become µg per filter, ratios get their own */
export function viewUnits(o: ViewOptions): Record<string, string> {
  const out: Record<string, string> = { ...o.units }
  if (o.basis === 'mass') for (const [f, u] of Object.entries(o.units)) if (isConcentration(u)) out[f] = 'µg/filter'
  if (o.hipsCeiling != null) {
    const mark = LINEARISED_MARK + (HIPS_SCOPES.find((x) => x.key === (o.hipsScope ?? 'all'))?.mark ?? '')
    for (const f of HIPS_VIEW_FIELDS) if (out[f]) out[f] += mark
  }
  for (const d of o.ratios) out[ratioKey(d)] = ratioUnit(out[d.num], out[d.den])
  return out
}

type Row = Record<string, unknown>
import { isFiniteNumber as num } from './stats.ts'

/**
 * A filter's HIPS optical depth, τ = Fabs × sampled volume ÷ 353; the public BC is Fabs / exportMac, so it stands in
 * for a missing Fabs. NaN when the filter has no HIPS or no sampled volume.
 */
export function hipsTau(r: Row, exportMac = 10): number {
  const v = r[VOLUME]
  const fabs = num(r[HIPS_FABS]) ? (r[HIPS_FABS] as number) : num(r[PUBLIC_BC]) ? (r[PUBLIC_BC] as number) * exportMac : null
  return fabs !== null && num(v) && v > 0 ? opticalDepth(fabs, v) : NaN
}

/**
 * The factor that linearises one filter's HIPS at a ceiling: 1 / (1 − τ/τ∞) with τ from its Fabs and sampled volume.
 * NaN when the filter has no HIPS, no sampled volume, or sits at or past the ceiling.
 */
export function hipsFactor(r: Row, ceiling: number, exportMac = 10): number {
  const tau = hipsTau(r, exportMac)
  return Number.isFinite(tau) ? correction(tau, ceiling) : NaN
}

/**
 * The rows with each filter's HIPS loading as two fields: its optical depth, and the under-read expected of it at
 * the central IMPROVE ceiling in percent (the protocol's statuses are marks on it: 20 % and 30 %). They are read off
 * the measured Fabs, so they do not move with the linearised view. A filter with a negative Fabs has neither.
 */
export function withHipsLoading<R extends Row>(rows: R[], exportMac = 10): R[] {
  return rows.map((r) => {
    const tau = hipsTau(r, exportMac)
    return Number.isFinite(tau) && tau > 0 ? { ...r, [HIPS_TAU]: tau, [HIPS_UNDER_READ]: 100 * underRead(tau) } : r
  })
}

/** one row under the view; the same object when nothing changes */
export function viewRow<R extends Row>(r: R, o: ViewOptions, concFields: string[]): R {
  const macChanged = o.mac !== o.exportMac
  const ceiling = o.hipsCeiling ?? null
  // a site outside the view's scope keeps HIPS as measured
  const linear = ceiling !== null && (o.hipsSites == null || o.hipsSites.has(r.code as string))
  if (!macChanged && o.basis === 'conc' && !o.ratios.length && !linear) return r
  const out: Row = { ...r }
  if (linear) {
    // the filter's own factor, from its own optical depth; the public BC is Fabs / exportMac, so it stands in for a missing Fabs
    const k = hipsFactor(r, ceiling!, o.exportMac)
    const ok = Number.isFinite(k)
    if (HIPS_FABS in r) out[HIPS_FABS] = ok && num(r[HIPS_FABS]) ? (r[HIPS_FABS] as number) * k : null
    if (HIPS_BC in r || HIPS_FABS in r) out[HIPS_BC] = ok && num(r[HIPS_FABS]) ? ((r[HIPS_FABS] as number) * k) / o.mac : null
    if (PUBLIC_BC in r) out[PUBLIC_BC] = ok && num(r[PUBLIC_BC]) ? ((r[PUBLIC_BC] as number) * k * o.exportMac) / o.mac : null
  } else if (macChanged) {
    if (num(r[HIPS_FABS])) out[HIPS_BC] = (r[HIPS_FABS] as number) / o.mac
    if (num(r[PUBLIC_BC])) out[PUBLIC_BC] = (r[PUBLIC_BC] as number) * o.exportMac / o.mac
  }
  if (o.basis === 'mass') {
    const v = r[VOLUME]
    for (const f of concFields) {
      if (!(f in out)) continue
      out[f] = num(out[f]) && num(v) && v > 0 ? (out[f] as number) * v : null
    }
  }
  for (const d of o.ratios) {
    const a = out[d.num], b = out[d.den]
    out[ratioKey(d)] = num(a) && num(b) && b !== 0 ? a / b : null
  }
  return out as R
}

export function viewRows<R extends Row>(rows: R[], o: ViewOptions): R[] {
  const conc = Object.entries(o.units).filter(([, u]) => isConcentration(u)).map(([f]) => f)
  return rows.map((r) => viewRow(r, o, conc))
}
