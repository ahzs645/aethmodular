import type { DashState, Sheet } from './dashboard.ts'
import type { FilterRow, MetaFile, PmfFile, PmfRow } from './types.ts'
import type { CalibrationData } from '../components/dashboard/extraTiles.tsx'
import type { ViewState } from './views.ts'

export interface SnapshotTile {
  id: string; kind: string; fields: string[]; rows: FilterRow[]; meta: MetaFile
  pmf: PmfFile | null; pmfRows: PmfRow[]
}
export interface DashboardSnapshot {
  schema: 'aeth-dashboard'; version: 1; created: string; revision: string; sourceFingerprint?: string; view: ViewState; dash: DashState; sheet: Sheet
  files?: Record<string, unknown>; calibration?: Omit<CalibrationData, 'knownIds'> & { knownIds: string[] };
  provenance: Record<string, unknown>; tiles: SnapshotTile[]; externalTiles: string[]
}
export interface SnapshotEnvelope { snapshot: DashboardSnapshot; sha256: string }

export async function snapshotDigest(value: unknown) {
  const bytes = new TextEncoder().encode(JSON.stringify(value))
  const hash = await crypto.subtle.digest('SHA-256', bytes)
  return [...new Uint8Array(hash)].map(v => v.toString(16).padStart(2, '0')).join('')
}

/** Validate before restoring any URL or data. A digest detects accidental edits, not authorship. */
export async function readSnapshot(text: string): Promise<DashboardSnapshot> {
  if (text.length > 25 * 1024 * 1024) throw new Error('Snapshot exceeds the 25 MiB import limit.')
  const envelope = JSON.parse(text) as SnapshotEnvelope
  const s = envelope?.snapshot
  if (!s || s.schema !== 'aeth-dashboard' || s.version !== 1) throw new Error('Unsupported dashboard snapshot version.')
  if (envelope.sha256 !== await snapshotDigest(s)) throw new Error('Snapshot checksum does not match its contents.')
  const strings = (v: unknown) => !!v && typeof v === 'object' && !Array.isArray(v) && Object.values(v).every(x => typeof x === 'string')
  if (typeof s.created !== 'string' || typeof s.revision !== 'string' || !s.view || !strings(s.view.h) || !strings(s.view.q) || !s.dash || !['site', 'sites', 'network'].includes(s.dash.scope)
    || !s.sheet || typeof s.sheet.id !== 'string' || typeof s.sheet.name !== 'string' || !Array.isArray(s.sheet.tiles)
    || !Array.isArray(s.tiles) || s.tiles.length > 200 || !Array.isArray(s.externalTiles) || !s.externalTiles.every(id => typeof id === 'string')) throw new Error('Invalid dashboard definition.')
  if (s.files && (typeof s.files !== 'object' || Array.isArray(s.files))) throw new Error('Invalid snapshot auxiliary files.')
  if (s.calibration && (!Array.isArray(s.calibration.knownIds) || !s.calibration.knownIds.every(id => typeof id === 'string') || !s.calibration.runs || !Array.isArray(s.calibration.runs.runs))) throw new Error('Invalid calibration snapshot.')
  const ids = new Set<string>()
  for (const tile of s.tiles) {
    if (typeof tile.id !== 'string' || ids.has(tile.id) || typeof tile.kind !== 'string' || !Array.isArray(tile.fields) || !tile.fields.every(f => typeof f === 'string')
      || !Array.isArray(tile.rows) || tile.rows.length > 200000 || !tile.meta || !Array.isArray(tile.meta.fields) || !tile.meta.field_units
      || !Array.isArray(tile.meta.sites) || !tile.meta.sites.every(v => v && typeof v.name === 'string' && typeof v.code === 'string') || !Array.isArray(tile.meta.seasons) || !tile.meta.seasons.every(v => v && typeof v.name === 'string' && Array.isArray(v.months))) throw new Error('Invalid snapshot tile.')
    ids.add(tile.id)
    for (const row of tile.rows) if (!row || typeof row.id !== 'string' || typeof row.site !== 'string' || typeof row.date !== 'string'
      || !Object.values(row).every(v => v === null || ['string', 'boolean', 'number'].includes(typeof v))) throw new Error('Invalid snapshot filter record.')
  }
  for (const t of s.sheet.tiles) if (!t || !ids.has(t.id) || !Number.isInteger(t.x) || !Number.isInteger(t.y)
    || !Number.isInteger(t.w) || !Number.isInteger(t.h) || t.x < 0 || t.y < 0 || t.w < 1 || t.w > 12 || t.h < 1 || t.h > 24
    || typeof t.kind !== 'string' || !Array.isArray(t.fields) || !t.fields.every(f => typeof f === 'string')) throw new Error('Invalid snapshot layout.')
  return s
}
