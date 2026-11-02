import { loadJson, useLoaded } from '@/lib/data'
import { pilotRecord, type PilotFilters, type PilotSpectra } from '@/lib/pilotRecord'

export * from '@/lib/pilotRecord'

/**
 * The Ethiopia pilot filters as per-filter records (gallery/data/export_pilot_filters.py): the five
 * Adama PTFE filters with their date-paired quartz TOR, and the 26 Bishoftu filters the frozen
 * external pilot scored. A filter drawer reads its record here: the Adama ones have no other
 * filter-level export, and the Bishoftu ones add the pilot's FTIR to their SPARTAN network row.
 * The corrected spectra are a separate file, fetched only when a drawer draws one.
 */

const check = (d: PilotFilters) => {
  if (d.schema_version !== 1) throw new Error('pilot_filters.json: unsupported schema')
  return d
}
export const loadPilotFilters = () => loadJson<PilotFilters>('pilot_filters.json').then(check)
export const loadPilotSpectra = () => loadJson<PilotSpectra>('pilot_spectra.json')

/**
 * The pilot record for a drawer's filter. The record file is small and cached, so every drawer can ask;
 * a missing file reads as no record (a build without the pilot export).
 */
export function usePilotRecord(id: string | null | undefined) {
  const data = useLoaded(id ? 'pilot-filters' : null, loadPilotFilters).data
  return { data: data ?? null, record: pilotRecord(data, id) }
}
