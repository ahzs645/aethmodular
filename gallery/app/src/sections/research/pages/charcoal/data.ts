/**
 * Research → Charcoal similarity: the export (public/data/research/charcoal/scores.json, from
 * gallery/data/export_research_charcoal.py) and its per-filter scores joined to the gallery's Addis rows in view
 * (common/charcoal.ts), with the season names and colours the charts use.
 */
import { useMemo } from 'react'
import { useCountedFilter, useExclStatus, useKnownFilter, useMarkedRows, usePlaceFilter } from '@/lib/markedRows'
import type { FilterRow, MetaFile } from '@/lib/types'
import { rowsOf } from '@/lib/table'
import { siteColor } from '@/lib/siteColours'
import { INK } from '@/lib/theme'
import { dryOf, joinNetworkScores, joinScores, markedScores, type CharcoalScore, type NetworkScore, type Scored } from '../../common/charcoal'
import { ADDIS, SERIES, shortSeason, useConcRows } from '../../common/ui'

/** Addis Ababa's SPARTAN site code (the PMF site; the Samples picker's code) */
const ADDIS_CODE = 'ETAD'
import { useSpectral } from '../spectralCommon'

export const EXPORTER = 'gallery/data/export_research_charcoal.py'
export const NB = {
  c02: 'research/ftir_etad_char/char_02_addis_vs_charcoal.ipynb',
  c03: 'research/ftir_etad_char/char_03_burned_vs_unburned.ipynb',
  c05: 'research/ftir_etad_char/char_05_addis_vs_unburned_feedstock.ipynb',
  c06: 'research/ftir_etad_char/char_06_dry_season_anomaly.ipynb',
  c07: 'research/ftir_etad_char/char_07_char05_without_dry_season.ipynb',
  c08: 'research/ftir_etad_char/char_08_fire_vs_furnace_char.ipynb',
  c09: 'research/ftir_etad_char/char_09_char08_dry_vs_nondry.ipynb',
  c12: 'research/ftir_etad_char/char_12_weakley_transferability.ipynb',
}

/** the notebooks' own season names (config.season_for_month: Dry Oct–Feb, Belg Mar–May, Kiremt Jun–Sep) */
export const NOTEBOOK_SEASONS = [
  { name: 'Dry Season', months: [10, 11, 12, 1, 2], key: 'dry' },
  { name: 'Belg Rainy Season', months: [3, 4, 5], key: 'belg' },
  { name: 'Kiremt Rainy Season', months: [6, 7, 8, 9], key: 'kiremt' },
]

/** the Samples settings that put the notebooks' 239 in view (NotebookPreset): their month bins, every date, exclusions included */
export const NOTEBOOK_CALENDAR = 'dry_feb'

export interface Season { name: string; color: string }
export interface Charcoal {
  scored: Scored<FilterRow>[]
  /**
   * Filters → excluded: Mark (lib/markedRows): the excluded scored filters (the Addis rows in view, or, for the
   * newer filters, any score whose filter is excluded), for the crossplots to draw as × and the notes to count; in no
   * median, share, test or count.
   */
  marked: Scored<FilterRow>[]
  /** the other sites in view, scored by char_05's recipe (network_char_scores.py): the raw-vs-char chart's, not the Addis charts' */
  others: Scored<FilterRow>[]
  othersMarked: Scored<FilterRow>[]
  /** the sites of `others`, with filters in view that have no FTIR spectrum to score */
  otherSites: Season[]
  unscored: number
  /** a site's colour by the name its rows carry */
  siteColorOf: (site: string) => string
  /** scored filters with no gallery Addis row in view */
  outside: number
  total: number
  /**
   * the filters shown are the notebooks' own: all `total` of them, each in the season the notebooks gave it (the
   * NotebookPreset). Only then do the charts print the notebooks' numbers beside their own ("nb").
   */
  notebook: boolean
  /** the notebooks' name for a season shown (same month bins), for looking up their saved tables; null if none */
  nbSeason: (season: string) => string | null
  seasons: Season[]
  colorOf: (season: string) => string
  isDry: (s: Pick<Scored, 'season' | 'month'>) => boolean
  /** the dry season's name in this set (for labels), and the Oct–Feb bands for time axes */
  dryName: string
  drySeasons: { name: string; months: number[]; color: string }[]
}

export function useCharcoalExport() {
  return useSpectral('scores', 'research/charcoal')
}

export function useCharcoal(viewRows: FilterRow[], meta: MetaFile, scoresTable: Parameters<typeof rowsOf>[0], networkTable?: Parameters<typeof rowsOf>[0]): Charcoal {
  // EC against loading is a µg/m³ reproduction: rows in concentration whatever the basis
  const concRows = useConcRows(viewRows, meta)
  const markedRows = useConcRows(useMarkedRows(), meta)
  const known = useKnownFilter()
  const counts = useCountedFilter()
  const status = useExclStatus()
  const place = usePlaceFilter()
  return useMemo(() => {
    const scores = rowsOf(scoresTable) as unknown as CharcoalScore[]
    const addis = concRows.filter((r) => r.site === ADDIS)
    const calendar = meta.seasons.filter((s) => !s.site || s.site === ADDIS)
    // the Addis rows in view, and the scored filters the gallery has no row for (sampled after its Addis series ends)
    // where the Samples filters would put a row with their date: the same sites, dates, calendar and season chips
    const joined = joinScores(scores, addis, 'plus', (date, month) => place(ADDIS, ADDIS_CODE, date, month), undefined, known)
    const { outside } = joined
    // Filters → excluded: a score joined to a gallery row in view already counts (the rows are the counted ones); a
    // newer filter counts by its filter id, as on every chart
    const scored = joined.scored.filter((s) => s.row || counts(s.id))
    const ownMarked = joined.scored.filter((s) => !s.row && !counts(s.id) && status(s.id) === 'marked')
    const marked = [...markedScores(scores, markedRows.filter((r) => r.site === ADDIS), scored), ...ownMarked]
    // seasons in calendar order from October, as the Addis calendars run
    const first = new Map<string, number>()
    for (const s of scored) first.set(s.season, Math.min(first.get(s.season) ?? 99, (s.month + 2) % 12))
    const names = [...first.keys()].sort((a, b) => first.get(a)! - first.get(b)!)
    const dryFeb = meta.season_conventions?.dry_feb ?? []
    const colorOf = (name: string) => {
      const own = [...meta.seasons, ...Object.values(meta.season_conventions ?? {}).flat()].find((s) => s.name === name)?.color
      if (own) return own
      // the notebooks' names take the dry_feb calendar's colours (the same month bins)
      const nb = NOTEBOOK_SEASONS.find((s) => s.name === name)
      const twin = nb && dryFeb.find((s) => s.name.toLowerCase().startsWith(nb.key))
      return twin?.color ?? SERIES[Math.max(0, names.indexOf(name)) % SERIES.length] ?? INK.muted
    }
    const isDry = dryOf(scored)
    const dryName = names.find((n) => isDry({ season: n, month: 0 })) ?? 'Oct–Feb'
    const drySeasons = calendar.filter((s) => isDry({ season: s.name, month: s.months[0] ?? 0 })).map((s) => ({ name: s.name, months: s.months, color: s.color }))
    // a season shown → the notebooks' season with the same months (the calendar in use names them its own way)
    const monthsOf = new Map(calendar.map((s) => [s.name, [...s.months].sort((a, b) => a - b).join(',')]))
    const nbByMonths = new Map(NOTEBOOK_SEASONS.map((s) => [[...s.months].sort((a, b) => a - b).join(','), s.name]))
    const nbSeason = (season: string) => nbByMonths.get(monthsOf.get(season) ?? '') ?? (NOTEBOOK_SEASONS.some((s) => s.name === season) ? season : null)
    const nbOwn = new Map(scores.map((s) => [s.FilterId, s.season]))
    const notebook = scored.length === scores.length && scored.every((s) => nbSeason(s.season) === nbOwn.get(s.filterId))
    // every other site in view: rows and their exclusions as they stand, in the order of the most filters first
    const network = rowsOf(networkTable) as unknown as NetworkScore[]
    const otherRows = concRows.filter((r) => r.site !== ADDIS)
    const others = joinNetworkScores(network, otherRows)
    const othersMarked = joinNetworkScores(network, markedRows.filter((r) => r.site !== ADDIS))
    const siteColorOf = (site: string) => siteColor(meta, site)
    const siteCounts = new Map<string, number>()
    for (const s of others) siteCounts.set(s.row!.site, (siteCounts.get(s.row!.site) ?? 0) + 1)
    const otherSites = [...siteCounts.keys()].sort((a, b) => siteCounts.get(b)! - siteCounts.get(a)!).map((name) => ({ name, color: siteColorOf(name) }))
    return { scored, others, othersMarked, otherSites, unscored: network.length ? otherRows.length - others.length : 0, siteColorOf, marked, outside, total: scores.length, notebook, nbSeason, seasons: names.map((name) => ({ name, color: colorOf(name) })), colorOf, isDry, dryName: shortSeason(dryName).replace(/\s+(Rainy\s+)?Season$/, ''), drySeasons }
  }, [scoresTable, networkTable, concRows, markedRows, meta, known, counts, status, place])
}
