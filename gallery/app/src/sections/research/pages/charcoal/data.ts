/**
 * Research → Charcoal similarity: the export (public/data/research/charcoal/scores.json, from
 * gallery/data/export_research_charcoal.py) and its per-filter scores joined to the gallery's Addis rows in view
 * (common/charcoal.ts), with the season names and colours the charts use.
 */
import { useMemo } from 'react'
import { useKnownFilter, useMarkedRows } from '@/lib/markedRows'
import { seasonOfDate } from '@/lib/seasons'
import { useSearchParam } from '@/lib/url'
import type { FilterRow, MetaFile } from '@/lib/types'
import { rowsOf } from '@/lib/table'
import { INK } from '@/lib/theme'
import { dryOf, joinScores, markedScores, SCORE_SETS, type CharcoalScore, type ScoreSet, type Scored } from '../../common/charcoal'
import { ADDIS, SERIES, shortSeason, useConcRows } from '../../common/ui'
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
const NOTEBOOK_SEASONS = [
  { name: 'Dry Season', months: [10, 11, 12, 1, 2], key: 'dry' },
  { name: 'Belg Rainy Season', months: [3, 4, 5], key: 'belg' },
  { name: 'Kiremt Rainy Season', months: [6, 7, 8, 9], key: 'kiremt' },
]

export const SET_LABEL: Record<ScoreSet, string> = { view: 'Gallery rows in view', plus: 'In view + newer', notebook: "Notebook's 239" }
export const SET_TITLE = 'Which scored filters the charts use: the gallery\'s Addis rows in view (your seasons, dates and exclusions); those plus the scored filters sampled after the gallery\'s Addis series ends (seasoned by your calendar); or every scored filter with the notebooks\' own seasons, Fabs and EC, which reproduces their tables.'

export interface Season { name: string; color: string }
export interface Charcoal {
  scored: Scored<FilterRow>[]
  /**
   * Filters → excluded: Mark (lib/markedRows): the excluded Addis rows in view that the notebooks scored, for the
   * crossplots to draw as × and the notes to count; in no median, share, test or count. None for the notebooks' set.
   */
  marked: Scored<FilterRow>[]
  /** scored filters with no gallery Addis row in view */
  outside: number
  total: number
  set: ScoreSet
  seasons: Season[]
  colorOf: (season: string) => string
  isDry: (s: Pick<Scored, 'season' | 'month'>) => boolean
  /** the dry season's name in this set (for labels), and the Oct–Feb bands for time axes */
  dryName: string
  drySeasons: { name: string; months: number[]; color: string }[]
}

/** the page's filter set, kept in the URL (ch.set) and shared by every chart */
export const useScoreSet = () => useSearchParam<ScoreSet>('ch.set', 'view', SCORE_SETS)

export function useCharcoalExport() {
  return useSpectral('scores', 'research/charcoal')
}

export function useCharcoal(viewRows: FilterRow[], meta: MetaFile, scoresTable: Parameters<typeof rowsOf>[0], set: ScoreSet): Charcoal {
  // EC against loading is a µg/m³ reproduction: rows in concentration whatever the basis
  const concRows = useConcRows(viewRows, meta)
  const markedRows = useConcRows(useMarkedRows(), meta)
  const known = useKnownFilter()
  return useMemo(() => {
    const scores = rowsOf(scoresTable) as unknown as CharcoalScore[]
    const addis = concRows.filter((r) => r.site === ADDIS)
    const calendar = meta.seasons.filter((s) => !s.site || s.site === ADDIS)
    const { scored, outside } = joinScores(scores, addis, set, (date, month) => seasonOfDate(date, month, calendar)?.name ?? null, undefined, known)
    // the exclusions act on the gallery's rows only: the notebooks' own set is fixed (the charts say so, FixedSetNote)
    const marked = set === 'notebook' ? [] : markedScores(scores, markedRows.filter((r) => r.site === ADDIS), scored)
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
    const drySeasons = set === 'notebook'
      ? [{ name: 'Dry Season', months: [10, 11, 12, 1, 2], color: colorOf('Dry Season') }]
      : calendar.filter((s) => isDry({ season: s.name, month: s.months[0] ?? 0 })).map((s) => ({ name: s.name, months: s.months, color: s.color }))
    return { scored, marked, outside, total: scores.length, set, seasons: names.map((name) => ({ name, color: colorOf(name) })), colorOf, isDry, dryName: shortSeason(dryName).replace(/\s+(Rainy\s+)?Season$/, ''), drySeasons }
  }, [scoresTable, concRows, markedRows, meta, set, known])
}
