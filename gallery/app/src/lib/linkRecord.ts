import { loadSimilarityMeta } from '@/lib/similarity'
import { loadImprove, loadSpartan, networkColor } from '@/lib/networks'
import { loadSpartanIndex } from '@/lib/spartan'
import { improveSiteDate, matchImproveRow } from '@/lib/improveLink'
import { baseFilterId } from '@/lib/filterId'
import { loadPilotFilters, pilotRecord, type PilotRecord } from '@/lib/pilotFilters'
import type { FilterRow, SiteMeta } from '@/lib/types'

/**
 * Where a filter a chart opens can be found when the page's own rows do not carry it (App openRecord),
 * so the drawer shows the filter rather than only the clicked chart's values:
 *
 *  - an IMPROVE spectral sample ("improve:2005726") is a row of the IMPROVE network export, by the
 *    spectral export's site and date (lib/improveLink);
 *  - a SPARTAN filter of a site outside the gallery's four ("ETBI-0002", "INDH-0141") is a row of that
 *    site's SPARTAN network file;
 *  - an Ethiopia pilot filter with no network row (the Adama PTFE filters) is a record of the pilot
 *    export (lib/pilotFilters).
 *
 * A linked network row comes with its site's rows (the drawer's reference and its previous / next)
 * and that network's sites, fields and units for the drawer's meta. One site file is loaded, the same
 * one the network view reads (lib/networks).
 */
export interface LinkedRows {
  /** the id the chart opened, e.g. improve:2005726 */
  recId: string
  /** the network row it is, e.g. PINN1-2022-11-22 */
  rowId: string
  network: 'IMPROVE' | 'SPARTAN'
  rows: FilterRow[]
  sites: SiteMeta[]
  fields: string[]
  field_units: Record<string, string>
  field_groups: { label: string; fields: string[] }[]
}

export type RecordLookup =
  | { state: 'linked'; linked: LinkedRows }
  | { state: 'pilot'; record: PilotRecord }
  | { state: 'missing'; reason: string }

/** Which export a lookup searches, for the drawer's "looking…" line; null when nothing would be searched. */
export function lookupTarget(id: string): string | null {
  if (id.startsWith('improve:')) return 'the IMPROVE network export'
  if (/^[A-Z]{4}-\d{4}$/.test(baseFilterId(id))) return 'the SPARTAN network export'
  if (/^J\d{4}$/.test(id)) return 'the Ethiopia pilot filters'
  return null
}

/** One site's network rows, the site in the first network colour (as it would be picked alone in that view). */
function linked(recId: string, row: FilterRow, network: LinkedRows['network'], code: string,
  net: { rows: FilterRow[]; sites: SiteMeta[]; fields: string[]; field_units: Record<string, string>; field_groups: { label: string; fields: string[] }[] }): LinkedRows {
  const color = networkColor(0)
  return { recId, rowId: row.id, network, rows: net.rows.filter((r) => r.code === code).map((r) => ({ ...r, color })),
    sites: net.sites.map((s) => (s.code === code ? { ...s, color } : s)), fields: net.fields, field_units: net.field_units, field_groups: net.field_groups }
}

async function linkImprove(recId: string): Promise<RecordLookup> {
  const at = improveSiteDate(await loadSimilarityMeta(), recId)
  if (!at) return { state: 'missing', reason: 'the spectral export has no site and date for it' }
  const net = await loadImprove([at.code])
  if (!net.sites.some((s) => s.code === at.code)) return { state: 'missing', reason: `the IMPROVE network export has no site ${at.code}` }
  const row = matchImproveRow(net.rows, at.code, at.date)
  if (!row) {
    const n = net.rows.filter((r) => r.code === at.code && r.date === at.date).length
    return { state: 'missing', reason: n ? `${n} filters at ${at.code} on ${at.date}, so it is ambiguous` : `no filter at ${at.code} on ${at.date}` }
  }
  return { state: 'linked', linked: linked(recId, row, 'IMPROVE', at.code, net) }
}

async function linkSpartan(recId: string, pilot: PilotRecord | null): Promise<RecordLookup | null> {
  const base = baseFilterId(recId)
  const code = base.slice(0, 4)
  const index = await loadSpartanIndex()
  if (!index.sites.some((s) => s.code === code)) return null
  // the bundled site file: what the gallery's SPARTAN view shows unless the reader switched it to live
  const net = await loadSpartan([code], 'bundled', 0)
  const row = net.rows.find((r) => r.id === base)
  return row ? { state: 'linked', linked: linked(recId, row, 'SPARTAN', code, net) } : pilot ? null : { state: 'missing', reason: `no filter ${base} in the SPARTAN network export` }
}

export async function linkRecord(recId: string): Promise<RecordLookup> {
  if (recId.startsWith('improve:')) return linkImprove(recId)
  // the pilot records are small and cached: a pilot filter with a network row still opens on that row
  const pilot = pilotRecord(await loadPilotFilters().catch(() => null), recId)
  if (/^[A-Z]{4}-\d{4}$/.test(baseFilterId(recId))) {
    const found = await linkSpartan(recId, pilot)
    if (found) return found
  }
  if (pilot) return { state: 'pilot', record: pilot }
  return { state: 'missing', reason: 'no filter-level export carries it' }
}
