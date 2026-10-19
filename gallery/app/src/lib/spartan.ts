/** Site/filter metadata is independent of the per-filter native spectra bytes. */
import { loadBinary, loadDerived, loadJson } from './data.ts'
import type { FilterRow, SiteMeta } from './types'
import type { SpartanReference } from './referencePopulation'

export interface SpartanSite extends Omit<SiteMeta, 'color' | 'timezone' | 'lat' | 'lon'> {
  lat: number | null; lon: number | null
  filters: string; native: string; n_ftir_analyses: number; n_hips_readings: number
}
export interface SpartanIndex {
  schema_version: number; fields: string[]; field_units: Record<string, string>; sites: SpartanSite[]; note: string
}
export interface NativeAnalysis {
  analysis_id: string; media_id: string; filter_id: string; filter_type: string
  date: string; analysis_date: string; qc: string; sample_scan_id: string; background_scan_id: string
  first: number; last: number; n: number; offset: number; invalid_points: number
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
  if (index.schema_version !== 1) throw new Error('Unsupported SPARTAN network index')
  return index
})
export const loadSpartanSite = (site: SpartanSite) => loadJson<{ rows: FilterRow[] }>(site.filters)
export const loadNativeSite = (site: SpartanSite) => loadJson<NativeSite>(site.native)

export async function loadSpartanFilters(picked: string[] | 'all') {
  const index = await loadSpartanIndex()
  const files = await Promise.all(index.sites.filter(s => picked === 'all' || picked.includes(s.code)).map(loadSpartanSite))
  return { index, rows: files.flatMap(f => f.rows) }
}

/** Float32 little-endian; descriptors are offsets in values, never bytes. */
export function nativeTrace(buffer: ArrayBuffer, a: NativeAnalysis) {
  if (!Number.isInteger(a.n) || a.n < 2 || !Number.isInteger(a.offset) || a.offset < 0 || (a.offset + a.n) * 4 > buffer.byteLength) {
    throw new Error('Native spectrum length does not match its descriptor')
  }
  const view = new DataView(buffer)
  const y = Float32Array.from({ length: a.n }, (_, i) => view.getFloat32((a.offset + i) * 4, true))
  const wn = Array.from({ length: a.n }, (_, i) => a.first + (a.last - a.first) * i / (a.n - 1))
  return { wn, y }
}
export async function loadNativeTrace(file: string, a: NativeAnalysis) {
  return nativeTrace(await loadBinary(file), a)
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
