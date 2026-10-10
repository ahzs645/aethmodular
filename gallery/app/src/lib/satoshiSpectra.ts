import type { SimilarityMeta } from './similarity'

type Columns = Record<string, (number | string | null)[]>
export type AnalogBaseline = 'AIRSpec' | 'VIBES-full-r70'
export type AnalogRegion = 'oct1' | 'presented'
export interface AddisSpectrumIds {
  ids: Record<string, string[]>
  status: Record<string, { in_test_set: boolean }>
}

/** Join saved median-spectrum analog ranks to the existing per-filter trace table. */
export function rankedAnalogSpectra(columns: Columns, meta: SimilarityMeta, baseline: AnalogBaseline, region: AnalogRegion, group: string) {
  const index = new Map(meta.filters.sample_id.map((id, i) => [id, i]))
  const rankColumn = baseline === 'AIRSpec' ? columns.rank_spline : columns.rank_vibes
  const scoreColumn = baseline === 'AIRSpec' ? columns.r_spline : columns.r_vibes
  const selected = columns.filter_id.map((_, i) => i)
    .filter(i => columns.mask[i] === region && columns.group[i] === group && rankColumn[i] != null)
    .sort((a, b) => Number(rankColumn[a]) - Number(rankColumn[b]))
  const rows: { filter: number; rank: number; r: number }[] = []
  const missing: string[] = []
  for (const i of selected) {
    const id = `improve:${columns.filter_id[i]}`
    const filter = index.get(id)
    if (filter == null) missing.push(id)
    else rows.push({ filter, rank: Number(rankColumn[i]), r: Number(scoreColumn[i]) })
  }
  return { rows, missing }
}

/** Frozen Addis test-set spectra, using the same dry-February calendar as the saved fits. */
export function addisTargetSpectra(meta: SimilarityMeta, ids: AddisSpectrumIds, group: string) {
  const site = meta.sites.findIndex(site => site.site === 'ETAD')
  if (site < 0) throw new Error('Addis is missing from the spectral export')
  const calendar = meta.calendars.ETAD
  if (!calendar) throw new Error('The Addis spectral calendar is missing')
  const season = group === 'All Addis' ? null : calendar.findIndex(season => season.name.replace(/–/g, '-') === group)
  if (season === -1) throw new Error(`No spectral season matches ${group}`)
  const target = meta.groups.findIndex(g => g.site === site && g.season === season)
  if (target < 0) throw new Error(`No spectral target matches ${group}`)
  const inTest = new Set(Object.entries(ids.ids).flatMap(([id, samples]) => ids.status[id]?.in_test_set ? samples : []))
  const filters = meta.filters.sample_id.map((_, i) => i).filter(i =>
    meta.filters.site[i] === site && inTest.has(meta.filters.sample_id[i]) && (season == null || meta.filters.season[i] === season))
  return { target, filters }
}
