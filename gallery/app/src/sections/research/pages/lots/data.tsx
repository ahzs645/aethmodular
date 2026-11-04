/**
 * Research → PTFE lot comparison: the export (public/data/research/lot_spectra/, from
 * gallery/data/export_research_lot_spectra.py), its per-view spectra decoded once per session, and the filters the
 * charts use, joined to the gallery rows in view (common/lotSpectra.ts) with the season names and lot colours.
 */
import { useMemo } from 'react'
import { loadBinary, loadJson, useDerived, useExport } from '@/lib/data'
import { ExcludedNote, useCountedFilter, useExclStatus, useMarkedRows, usePlaceFilter } from '@/lib/markedRows'
import type { FilterRow, FiltersFile, MetaFile } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import { decodeView, defaultSeason, membersOf, notebookSeason, type LotExport, type Member, type ViewKey } from '../../common/lotSpectra'
import { SERIES, shortSeason } from '../../common/ui'
import { LOT_COLOR } from '@/lib/theme'

export const EXPORTER = 'gallery/data/export_research_lot_spectra.py'
export const FOLDER = 'research/lot_spectra'
export const NB = 'research/ftir_etad_char/char_12_weakley_transferability.ipynb (scripts/run_char_12.py)'

/** the gallery's lot colours (lib/theme LOT_COLOR: 248 and 251 as char_12 draws them); other lots from the page ramp */
export const lotColor = (lot: number, i = 0) => LOT_COLOR[lot] ?? SERIES[(i + 3) % SERIES.length]

/** char_12's calendar among the gallery's (config.season_for_month: February in Dry), for the CohortPreset */
export const NOTEBOOK_CALENDAR = 'dry_feb'

export function useLotExport() {
  return useExport<LotExport>(`${FOLDER}/lot_spectra.json`, (d) => (d.schema_version === 1 ? null : 'unsupported data version'))
}

/** one view's spectra, decoded (null while loading) */
export function useView(exp: LotExport | null, key: ViewKey): { values: Float32Array | null; error: string } {
  const file = exp ? `${FOLDER}/${exp.views[key].file}` : null
  const { data, error } = useDerived(file && `lot-spectra:${file}`, async () => decodeView(await loadBinary(file!), exp!.filters.id.length, exp!.views[key].points))
  return { values: data, error }
}

/** every gallery row id (filters.json, already loaded by the app): a spectrum outside it has no gallery row at all */
function useKnownIds(): Set<string> | null {
  const { data, error } = useDerived('lot-spectra:known', async () => new Set((await loadJson<FiltersFile>('filters.json')).rows.map((r) => r.id)))
  return useMemo(() => data ?? (error ? new Set<string>() : null), [data, error])
}

export interface Lots {
  members: Member[]
  /**
   * Filters → excluded: Mark (lib/markedRows): the exported spectra of the excluded filters (the gallery rows in view
   * joined by id, or, for the spectra with no row, by filter id), for the charts' notes; in no centre, band, box or test.
   */
  marked: Member[]
  /** exported spectra of the sites in view with no gallery row */
  unmatched: number
  /** char_12's cohort: its size and sampling dates, for the CohortPreset */
  cohort: { n: number; from: string; to: string }
  /** the filters shown are char_12's 239 under its calendar (the CohortPreset), so its saved tables compare like for like */
  notebook: boolean
  /** site code → display name */
  siteName: (code: string) => string
  siteColor: (code: string) => string
  /** the seasons among the members, by short name ("Bega": members' `season` is the same), in calendar order from October */
  seasons: { name: string; color: string }[]
}

export function useLots(exp: LotExport | null, rows: FilterRow[], meta: MetaFile): Lots | null {
  const known = useKnownIds()
  const markedRows = useMarkedRows()
  const counts = useCountedFilter()
  const status = useExclStatus()
  const place = usePlaceFilter()
  return useMemo(() => {
    if (!exp || !known) return null
    const nameOf = new Map([...Object.entries(exp.site_names ?? {}), ...meta.sites.map((s) => [s.code, s.name] as [string, string])])
    for (const r of rows) nameOf.set(r.code, r.site)
    const colorOf = new Map(meta.sites.map((s) => [s.code, s.color]))
    for (const r of rows) colorOf.set(r.code, r.color)
    // the gallery rows in view, and the spectra of those sites with no gallery row (sampled after the series ends)
    // where the Samples settings would put a row with their site and date: the same dates, calendar and season chips
    const joined = membersOf(exp, 'plus', rows, known, (code, date, month) => place(nameOf.get(code) ?? code, code, date, month))
    const { unmatched } = joined
    // Filters → excluded: a member joined to a gallery row in view already counts (the rows are the counted ones); a
    // spectrum with no row counts by its filter id, as on every chart
    const members = joined.members.filter((m) => m.inView || counts(m.id))
    const ownMarked = joined.members.filter((m) => !m.inView && !counts(m.id) && status(m.id) === 'marked')
    // only spectra with no gallery row at all are added, so an excluded row is never among them
    const marked = [...(!markedRows.length ? [] : membersOf(exp, 'view', markedRows, known, () => null).members), ...ownMarked]
    const f = exp.filters
    const c12 = f.date.filter((_, i) => f.char12[i]).sort()
    const cohort = { n: c12.length, from: c12[0] ?? '', to: c12[c12.length - 1] ?? '' }
    const notebook = cohort.n > 0 && members.length === cohort.n && members.every((m) => f.char12[m.i] && m.season === notebookSeason(m.month))
    // a per-site calendar qualifies each season with its site ("Addis Ababa · Bega (Oct-Jan, dry)"), so two sites with
    // the same season would be two seasons: the charts group by the short name, and a name the sites share is one season
    const named = (ms: Member[]) => ms.map((m) => ({ ...m, season: shortSeason(m.season) }))
    const shown = named(members)
    const first = new Map<string, number>()
    for (const m of shown) first.set(m.season, Math.min(first.get(m.season) ?? 99, (m.month + 2) % 12))
    const names = [...first.keys()].sort((a, b) => first.get(a)! - first.get(b)!)
    const palette = [...meta.seasons, ...Object.values(meta.season_conventions ?? {}).flat()]
    const seasonColor = (name: string, i: number) => palette.find((s) => shortSeason(s.name) === name)?.color ?? SERIES[i % SERIES.length]
    return {
      members: shown, marked: named(marked), unmatched, cohort, notebook,
      siteName: (c: string) => nameOf.get(c) ?? c,
      siteColor: (c: string) => colorOf.get(c) ?? SERIES[0],
      seasons: names.map((name, i) => ({ name, color: seasonColor(name, i) })),
    }
  }, [exp, known, rows, markedRows, meta, counts, status, place])
}

/**
 * The season a chart holds fixed: the link's pick, else the dry one. Read through shortSeason, so a link saved when
 * the options still carried their site ("Addis Ababa · Belg …") keeps its season.
 */
export function useHeldSeason(lots: Lots, param: string) {
  const seasonNames = lots.seasons.map((s) => s.name)
  const [pick, setSeason] = useSearchParam<string>(param, '')
  const short = shortSeason(pick)
  return { seasonNames, season: seasonNames.includes(short) ? short : defaultSeason(seasonNames), setSeason }
}

/**
 * The one exclusion note a lot chart carries: under Mark, how many excluded filters with a spectrum it leaves out
 * (`of` narrows them to what the chart reads). The charts draw centres, bands and boxes, never a
 * filter, so nothing is drawn as ×.
 */
export function LotNote({ lots, of }: { lots: Lots; of?: (m: Member) => boolean }) {
  return <ExcludedNote count={(of ? lots.marked.filter(of) : lots.marked).length} />
}
