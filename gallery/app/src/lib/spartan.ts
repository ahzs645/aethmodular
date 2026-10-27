/** Site/filter metadata is independent of the per-filter native spectra bytes. */
import { loadBinary, loadDerived, loadJson } from './data.ts'
import type { FilterRow, SiteMeta } from './types'
import type { SpartanReference } from './referencePopulation'

export interface SpartanSite extends Omit<SiteMeta, 'color' | 'timezone' | 'lat' | 'lon'> {
  lat: number | null; lon: number | null
  filters: string; native: string; n_ftir_analyses: number; n_hips_readings: number
}
/** the evenly spaced grid a trace lives on, from its first and last wavenumber */
export interface Grid { first: number; last: number; n: number }
export interface SpartanIndex {
  schema_version: number; fields: string[]; field_units: Record<string, string>; sites: SpartanSite[]; note: string
  /** schema 2: precomputed baselines (research/ftir_hips_chem/workflows/network_baselines_20261019.py) */
  baselines?: {
    airspec: { method: string } | null
    vibes: { method: string; grid: VibesGrid } | null
  }
}
/** one VIBES blank background: which blanks (and so which filter lots) trained it */
export interface VibesModel { label: string; rank: number; n_blanks: number; lots: string[]; blanks: string; note?: string; source: string }
/** the 26 Oct held-out field-blank test for one lot: median residue under the 87-blank background and under the one used */
export interface LotCheck { heldout_blanks: number; model: string; ptfe_rms_b87: number; ptfe_rms: number; rms_b87: number; rms: number }
export interface VibesGrid extends Grid {
  model: string; model_rank: number; model_blanks: number; resampling: string
  /** since 26 Oct: every background in use by key (NativeAnalysis.vibes_model), and the held-out check by filter lot */
  models?: Record<string, VibesModel>; lot_check?: Record<string, LotCheck>
}
/** one stored trace (gallery/data/spectrum_codec.py): integer steps, differences, byte planes; offset in bytes */
export interface EncodedTrace { offset: number; n: number; step: number; nan?: [number, number][]; start?: number }
export interface NativeAnalysis {
  analysis_id: string; media_id: string; filter_id: string; filter_type: string
  date: string; analysis_date: string; qc: string; sample_scan_id: string; background_scan_id: string
  first: number; last: number; n: number; invalid_points: number
  /** the filter's manufacturing lot (spartan.Filters.ExternalLotId), and the VIBES background behind its trace */
  lot?: string | null; vibes_model?: string
  /** schema 1: float32 raw absorbance at this offset (in values) */
  offset?: number
  /** schema 2: raw absorbance, and the baselines where the run has them */
  series?: { raw: EncodedTrace; airspec_baseline?: EncodedTrace; vibes_corrected?: EncodedTrace }
}
export interface HipsReading {
  analysis_id: string; filter_id: string; filter_type: string; timestamp: string; qc: string
  Transmittance: number | null; Reflectance: number | null
  TransmittanceRaw: number | null; ReflectanceRaw: number | null; Wavelength: number | null
}
export interface NativeSite {
  schema_version: number; code: string
  spectra: Record<string, { file: string; analyses: NativeAnalysis[] }>
  hips: Record<string, HipsReading[]>
}
export const loadSpartanIndex = () => loadJson<SpartanIndex>('spartan_network/index.json').then(index => {
  if (index.schema_version !== 1 && index.schema_version !== 2) throw new Error('Unsupported SPARTAN network index')
  return index
})
export const loadSpartanSite = (site: SpartanSite) => loadJson<{ rows: FilterRow[] }>(site.filters)
export const loadNativeSite = (site: SpartanSite) => loadJson<NativeSite>(site.native)

export async function loadSpartanFilters(picked: string[] | 'all') {
  const index = await loadSpartanIndex()
  const files = await Promise.all(index.sites.filter(s => picked === 'all' || picked.includes(s.code)).map(loadSpartanSite))
  return { index, rows: files.flatMap(f => f.rows) }
}

export const gridOf = (g: Grid) => Array.from({ length: g.n }, (_, i) => g.first + (g.last - g.first) * i / (g.n - 1))

/** One stored trace back to absorbance: byte planes → 32-bit differences → running sum × step. */
export function decodeTrace(buffer: ArrayBuffer, t: EncodedTrace): Float64Array {
  const { n, offset, step } = t
  if (!Number.isInteger(n) || n < 1 || !Number.isInteger(offset) || offset < 0 || offset + 4 * n > buffer.byteLength) {
    throw new Error('Spectrum length does not match its descriptor')
  }
  const b = new Uint8Array(buffer, offset, 4 * n)
  const y = new Float64Array(n)
  let q = 0
  for (let i = 0; i < n; i++) {
    q += (b[i] | (b[n + i] << 8) | (b[2 * n + i] << 16) | (b[3 * n + i] << 24))
    y[i] = q * step
  }
  for (const [start, len] of t.nan ?? []) y.fill(NaN, start, start + len)
  return y
}

export interface Trace { wn: number[]; y: ArrayLike<number> }
/** This analysis's traces: raw always; AIRSpec (corrected and its baseline) and VIBES (corrected) where present. */
export interface NativeTraces { raw: Trace; airspec?: Trace; airspecBaseline?: Trace; vibes?: Trace }

export function nativeTraces(buffer: ArrayBuffer, a: NativeAnalysis, vibesGrid?: Grid | null): NativeTraces {
  const wn = gridOf(a)
  if (!a.series) {
    // schema 1: float32 little-endian, offsets in values
    const off = a.offset ?? -1
    if (!Number.isInteger(a.n) || a.n < 2 || off < 0 || (off + a.n) * 4 > buffer.byteLength) {
      throw new Error('Native spectrum length does not match its descriptor')
    }
    const view = new DataView(buffer)
    return { raw: { wn, y: Float32Array.from({ length: a.n }, (_, i) => view.getFloat32((off + i) * 4, true)) } }
  }
  const raw = decodeTrace(buffer, a.series.raw)
  const out: NativeTraces = { raw: { wn, y: raw } }
  const ab = a.series.airspec_baseline
  if (ab) {
    const base = decodeTrace(buffer, ab), start = ab.start ?? 0
    const span = wn.slice(start, start + base.length)
    out.airspecBaseline = { wn: span, y: base }
    out.airspec = { wn: span, y: base.map((v, i) => raw[start + i] - v) }
  }
  if (a.series.vibes_corrected && vibesGrid) out.vibes = { wn: gridOf(vibesGrid), y: decodeTrace(buffer, a.series.vibes_corrected) }
  return out
}
const residue = (v: number) => String(+v.toPrecision(1))
/**
 * Where an analysis's VIBES trace comes from, for the drawer: a short visible line and a longer
 * tooltip. Says which blanks trained the background, whether this filter's lot is among them, and how
 * held-out field blanks of the lot corrected in the 26 Oct test. Null without a VIBES trace.
 */
export function vibesProvenance(a: Pick<NativeAnalysis, 'lot' | 'vibes_model'>, grid: VibesGrid | null | undefined): { line: string; detail: string } | null {
  if (!grid || !a.vibes_model) return null
  const model = grid.models?.[a.vibes_model]
  if (!model) return { line: `VIBES background: ${grid.model_blanks} blanks (IMPROVE and Addis), rank ${grid.model_rank}.`, detail: '' }
  const lot = a.lot ?? null
  const check = lot ? grid.lot_check?.[lot] : undefined
  const among = !!lot && model.lots.includes(lot)
  const lots = model.lots.join(', ')
  const head = `VIBES background: ${model.label} (lots ${lots}; rank ${model.rank}).`
  const tail = !lot ? 'This filter has no lot on record.'
    : a.vibes_model !== 'b87' ? `This filter is lot ${lot}, which the 87-blank background has no blank of, so it is corrected with its lot's own field blanks.`
    : among ? `This filter's lot ${lot} is among them.`
    : check ? `This filter's lot ${lot} is not among them; held-out lot-${lot} field blanks still correct as cleanly (median PTFE-band residue ${residue(check.ptfe_rms)}).`
    : `This filter's lot ${lot} is not among them and was not checked.`
  const checked = check ? `Held-out field blanks of lot ${lot} (${check.heldout_blanks}), median RMS left after correction: ${residue(check.rms)} over 4000–500 cm⁻¹, ${residue(check.ptfe_rms)} in the PTFE band (1050–1250)${a.vibes_model !== 'b87' ? `; with the 87-blank background they kept ${residue(check.rms_b87)} and ${residue(check.ptfe_rms_b87)}` : ''}.` : ''
  return { line: `${head} ${tail}`, detail: [`Blanks: ${model.blanks}.`, model.note ?? '', checked, `Source: ${model.source}.`].filter(Boolean).join(' ') }
}

export async function loadNativeTraces(file: string, a: NativeAnalysis, vibesGrid?: Grid | null) {
  return nativeTraces(await loadBinary(file), a, vibesGrid)
}

/** Only an explicit network-reference request loads every site's filter numbers.
 * Raw spectra are never fetched by this path. */
export const loadSpartanReference = () => loadDerived<SpartanReference>('spartan-network-reference', async () => {
  const index = await loadSpartanIndex()
  const files = await Promise.all(index.sites.map(loadSpartanSite))
  const rows = files.flatMap(f => f.rows)
  return { schema_version: 1, source: 'SPARTAN public, lab and native filter metadata', note: index.note,
    n: rows.length, n_sites: index.sites.length, fields: index.fields, field_units: index.field_units, field_sources: {}, rows }
})
