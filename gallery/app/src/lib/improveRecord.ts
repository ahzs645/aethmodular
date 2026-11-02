import { loadSimilarityMeta } from '@/lib/similarity'
import { loadImprove, networkColor } from '@/lib/networks'
import { improveSiteDate, matchImproveRow } from '@/lib/improveLink'
import type { FilterRow, SiteMeta } from '@/lib/types'

/**
 * An IMPROVE filter a chart knows only by its FTIR sample number ("improve:2005726", the Satoshi
 * and calibration pages), found in the IMPROVE network export so the full sample drawer can open
 * on it from any page: its site's rows (for the reference and the previous / next filter), and
 * that network's sites, fields and units for the drawer's meta. One site file is loaded, the
 * same one the IMPROVE network view reads (lib/networks loadImprove).
 */
export interface LinkedImprove {
  /** the id the chart opened, e.g. improve:2005726 */
  recId: string
  /** the network row it is, e.g. PINN1-2022-11-22 */
  rowId: string
  rows: FilterRow[]
  sites: SiteMeta[]
  fields: string[]
  field_units: Record<string, string>
  field_groups: { label: string; fields: string[] }[]
}

export type ImproveLookup = { state: 'linked'; linked: LinkedImprove } | { state: 'missing'; reason: string }

export async function linkImproveRecord(recId: string): Promise<ImproveLookup> {
  const at = improveSiteDate(await loadSimilarityMeta(), recId)
  if (!at) return { state: 'missing', reason: 'the spectral export has no site and date for it' }
  const net = await loadImprove([at.code])
  const site = net.sites.find((s) => s.code === at.code)
  if (!site) return { state: 'missing', reason: `the IMPROVE network export has no site ${at.code}` }
  // the one site in view takes the first network colour, as it would picked alone in the IMPROVE view
  const color = networkColor(0)
  const rows = net.rows.map((r) => ({ ...r, color }))
  const row = matchImproveRow(rows, at.code, at.date)
  if (!row) {
    const n = rows.filter((r) => r.date === at.date).length
    return { state: 'missing', reason: n ? `${n} filters at ${at.code} on ${at.date}, so it is ambiguous` : `no filter at ${at.code} on ${at.date}` }
  }
  return { state: 'linked', linked: { recId, rowId: row.id, rows, sites: net.sites.map((s) => (s.code === at.code ? { ...s, color } : s)),
    fields: net.fields, field_units: net.field_units, field_groups: net.field_groups } }
}
