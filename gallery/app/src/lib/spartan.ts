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
    vibes: { method: string; grid: Grid & { model: string; model_rank: number; model_blanks: number; resampling: string } } | null
  }
}
/** one stored trace (gallery/data/spectrum_codec.py): integer steps, differences, byte planes; offset in bytes */
export interface EncodedTrace { offset: number; n: number; step: number; nan?: [number, number][]; start?: number }
export interface NativeAnalysis {
  analysis_id: string; media_id: string; filter_id: string; filter_type: string
  date: string; analysis_date: string; qc: string; sample_scan_id: string; background_scan_id: string
  first: number; last: number; n: number; invalid_points: number
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
