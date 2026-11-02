import type { ReferenceRow } from './referencePopulation.ts'
import { baseFilterId } from './filterId.ts'

/** The pilot filters export's shapes and its id lookup (gallery/data/export_pilot_filters.py); no browser imports, so node tests read it. */

export type PilotValue = number | string | null
export interface PilotRecord {
  id: string
  campaign: 'Adama' | 'Bishoftu'
  site: string
  code: string
  date: string
  medium: string
  partner: { label: string; id: string } | null
  /** the corrected-spectrum id in pilot_spectra.json */
  spectrum: string
  /** "provisional": the Adama analysis-to-filter map is the pilot's date-based one */
  spectrum_map: 'provisional' | 'confirmed'
  flags: string[]
  notes: string[]
  values: Record<string, PilotValue>
}
export interface PilotFilters {
  schema_version: number
  generated: string
  frozen_run_signature: string
  background_rank: number
  units: Record<string, string>
  groups: { label: string; fields: string[] }[]
  campaigns: Record<string, { site: string; n: number; note: string }>
  reference: { label: string; n: number; fields: string[]; rows: ReferenceRow[] }
  filters: PilotRecord[]
}
export interface PilotSpectra {
  schema_version: number
  ids: string[]
  units: string
  AIRSpec: { bin_wn: number[]; y: number[][]; window: [number, number] }
  VIBES: { bin_wn: number[]; y: number[][]; window: [number, number]; background_rank: number; presentation_status: string }
}

/** A pilot record by the id a chart or drawer knows: the filter's own (J1233, ETBI-0002-2) or the SPARTAN base id (ETBI-0002). */
export function pilotRecord(data: PilotFilters | null | undefined, id: string | null | undefined): PilotRecord | null {
  if (!data || !id) return null
  return data.filters.find((f) => f.id === id) ?? data.filters.find((f) => baseFilterId(f.id) === baseFilterId(id)) ?? null
}

