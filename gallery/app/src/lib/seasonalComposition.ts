import type { FilterRow, MetaFile } from './types'
import { localSeasonsFor } from '../siteSeasons.ts'

export type CompositionCalendar = 'local' | 'dry_feb'

/** Re-bin the existing Addis chemistry without changing the source rows or
 * treating the frozen spectral evaluation as the chemistry population. */
export function seasonalComposition(rows: FilterRow[], meta: MetaFile, calendar: CompositionCalendar = 'local') {
  const seasons = calendar === 'local'
    ? localSeasonsFor('Addis Ababa', meta) ?? []
    : meta.season_conventions.dry_feb ?? []
  const clean = rows.filter(r => r.site === 'Addis Ababa' && !r.excluded)
  const relabelled: FilterRow[] = clean.map(r => ({ ...r, season: seasons.find(s => s.months.includes(r.month ?? -1))?.name ?? 'Unknown month' }))
  // Public ChemSpec EC/OC duplicate these FTIR products; HIPS BC and OM must
  // not be added to EC/OC as extra carbon slices.
  const species = ['OC (FTIR)', 'EC (FTIR)', ...meta.field_groups
    .filter(g => g.label === 'Ions' || g.label === 'Crustal elements' || g.label === 'Trace metals').flatMap(g => g.fields)]
    .filter(f => meta.fields.includes(f))
  const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
  const stats = seasons.map(season => {
    const sub = relabelled.filter(r => r.season === season.name)
    const dates = sub.map(r => r.date).filter(Boolean).sort()
    const fields = [...species, 'PM2.5 mass'].map(field => {
      const values = sub.map(r => r[field]).filter(finite)
      return { field, n: values.length, negative: values.filter(v => v < 0).length,
        mean: values.length ? values.reduce((a, b) => a + b, 0) / values.length : null }
    })
    return { season, n: sub.length, start: dates[0] ?? null, end: dates[dates.length - 1] ?? null, fields }
  })
  return { rows: relabelled, species, stats, unknownMonths: relabelled.filter(r => r.season === 'Unknown month').length,
    meta: { ...meta, season_convention: calendar, season_site: 'Addis Ababa', seasons } }
}
