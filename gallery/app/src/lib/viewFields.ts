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
 * - Ratios: numerator / denominator on the same filter, after the two steps above.
 */
export interface RatioDef { num: string; den: string }
export type Basis = 'conc' | 'mass'

export const RATIO_GROUP = 'Ratios'
export const VOLUME = 'Sampled volume'
export const MAC_CHOICES = [5, 7.5, 10, 12.5, 15, 17.5, 20] as const
/** fields recomputed from HIPS Fabs when the MAC changes */
const HIPS_BC = 'HIPS BC'
const HIPS_FABS = 'HIPS Fabs'
const PUBLIC_BC = 'BC (public HIPS)'

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

const tidy = (u: string | undefined) => (u ?? '').replace('ug/m3', 'µg/m³').replace('Mm-1', 'Mm⁻¹')

/** the unit of a ratio from its parts' units (already in the view's basis) */
export function ratioUnit(num: string | undefined, den: string | undefined): string {
  const a = tidy(num), b = tidy(den)
  return a === b ? 'ratio' : a && b ? `${a} per ${b}` : 'ratio'
}

export interface ViewOptions {
  mac: number
  /** the MAC the exports used (meta.mac_value) */
  exportMac: number
  basis: Basis
  /** field → unit, before this view */
  units: Record<string, string>
  ratios: RatioDef[]
}

/** field units after the view: concentrations become µg per filter, ratios get their own */
export function viewUnits(o: ViewOptions): Record<string, string> {
  const out: Record<string, string> = { ...o.units }
  if (o.basis === 'mass') for (const [f, u] of Object.entries(o.units)) if (isConcentration(u)) out[f] = 'µg/filter'
  for (const d of o.ratios) out[ratioKey(d)] = ratioUnit(out[d.num], out[d.den])
  return out
}

type Row = Record<string, unknown>
const num = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

/** one row under the view; the same object when nothing changes */
export function viewRow<R extends Row>(r: R, o: ViewOptions, concFields: string[]): R {
  const macChanged = o.mac !== o.exportMac
  if (!macChanged && o.basis === 'conc' && !o.ratios.length) return r
  const out: Row = { ...r }
  if (macChanged) {
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
