import type { SimilarityMeta } from './similarity'

/**
 * The two names one IMPROVE filter goes by. The spectral and calibration exports call it by its
 * FTIR sample number ("improve:2005726", gallery/data/export_similarity.py); the IMPROVE network
 * export calls it by site and sampling day ("PINN1-2022-11-22", with "-<POC>" for a collocated
 * second sampler; gallery/data/export_improve_samples.py and lib/networks loadImprove). The
 * spectral export carries each sample's site and date, which is the whole link between them.
 *
 * A day that matches more than one filter on either side (collocated samplers) is left unlinked
 * rather than guessed: the drawer then shows only what the clicked chart knows.
 */

type SimFilters = Pick<SimilarityMeta, 'sites' | 'filters'>

/** "PINN1-2022-11-22" → its site code and date; a POC suffix ("…-2") is kept apart. */
export function parseImproveRowId(id: string): { code: string; date: string; poc: number } | null {
  const m = /^([A-Z0-9]{4,6})-(\d{4}-\d{2}-\d{2})(?:-(\d+))?$/.exec(id)
  return m ? { code: m[1], date: m[2], poc: m[3] ? Number(m[3]) : 1 } : null
}

/** The site code and sampling date of an "improve:<FilterId>" spectral sample, or null when the export lacks either. */
export function improveSiteDate(meta: SimFilters, sampleId: string): { code: string; date: string } | null {
  if (!sampleId.startsWith('improve:')) return null
  const i = meta.filters.sample_id.indexOf(sampleId)
  if (i < 0) return null
  const code = meta.sites[meta.filters.site[i]]?.site
  const date = meta.filters.date[i]
  return code && date ? { code, date } : null
}

/** The network row for a site and day: its id when exactly one row matches, else null. */
export function matchImproveRow<R extends { id: string; code: string; date: string }>(rows: readonly R[], code: string, date: string): R | null {
  const hits = rows.filter((r) => r.code === code && r.date === date)
  return hits.length === 1 ? hits[0] : null
}

/**
 * The spectral sample for a network row id, the other way round: only for the primary sampler's
 * row (no POC suffix) and only when the spectral export has exactly one IMPROVE sample that day.
 */
export function improveSampleFor(meta: SimFilters, rowId: string): string | null {
  const p = parseImproveRowId(rowId)
  if (!p || p.poc !== 1) return null
  const site = meta.sites.findIndex((s) => s.site === p.code)
  if (site < 0) return null
  const { sample_id, site: sites, date } = meta.filters
  let hit: string | null = null
  for (let i = 0; i < sample_id.length; i++) {
    if (sites[i] !== site || date[i] !== p.date || !sample_id[i].startsWith('improve:')) continue
    if (hit) return null
    hit = sample_id[i]
  }
  return hit
}
