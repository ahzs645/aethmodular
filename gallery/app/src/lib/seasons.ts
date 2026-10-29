/**
 * Season vocabulary shared by every tab (docs/site-seasonality.md).
 *
 * Two Addis calendars are in use, on purpose:
 * - `belg_feb`, the Ethiopian Meteorological Institute bins (Bega Oct–Jan, Belg
 *   Feb–May, Kiremt Jun–Sep): the gallery's local calendar for Addis.
 * - `dry_feb`, the older bins (Dry Oct–Feb, Belg Mar–May, Kiremt Jun–Sep): every
 *   frozen calibration analysis (meeting follow-ups, Satoshi, similarity, the
 *   explorer) was computed on these, so those pages keep them and say so.
 * February is the only month that moves. Pages that show frozen results name
 * their calendar with `ANALYSIS_CALENDAR_NOTE` rather than relabelling.
 */
import type { SeasonMeta } from './types'

export type SeasonConvention = 'local' | 'dry_feb' | 'belg_feb'

/** Names for the calendar picker and for calendar notes. */
export const CONVENTION_LABEL: Record<string, string> = {
  local: 'Local seasons · per site',
  dry_feb: 'Ethiopian seasons at every site · February in Dry',
  belg_feb: 'Ethiopian seasons at every site · February in Belg',
  quarters: 'Quarters at every site · Dec–Feb, Mar–May, Jun–Aug, Sep–Nov',
  custom: 'Custom · your own months at every site',
  pmf_paper: 'PMF paper dates · its 2023 season dates, reused every year',
  rain_rule: 'Rain rule · each site and year from its own daily rain (ERA5)',
}

/** a calendar's short name for a picker ('Rain rule', 'Per site', 'Shared Ethiopian month bins'): its label up to the ' · ' */
export const calendarShortName = (key: string | null | undefined) => (CONVENTION_LABEL[key ?? ''] ?? key ?? '').split(' · ')[0]

/** The calendar every frozen calibration export was binned with. */
export const ANALYSIS_CALENDAR: SeasonConvention = 'dry_feb'
export const ANALYSIS_CALENDAR_NOTE =
  'Seasons follow the calibration analyses: Dry Oct–Feb, Belg Mar–May, Kiremt Jun–Sep (dry_feb). ' +
  "The gallery's Addis local calendar moves February into Belg."

/** Every Addis filter, pooled over the seasons (a group in the meeting and Satoshi exports). */
export const ALL_ADDIS = 'All Addis'
export const isAllAddis = (group: string | null | undefined) => group === ALL_ADDIS || group === 'all'

/** Does a filter's season belong to a selected group ("All Addis" takes every season)? */
export const inSeasonGroup = (season: string | null | undefined, group: string) => isAllAddis(group) || season === group

/**
 * One spelling for season names: exports disagree on the dash in the month range
 * ("Dry (Oct-Feb)" vs "Dry (Oct–Feb)"); the calibration exports and URLs use the
 * ASCII hyphen, so every name is read in that form.
 */
export const normalizeSeasonName = (name: string) => name.replace(/\s*[–—]\s*/g, '-')

/** "Addis Ababa · Kiremt (Jun-Sep)" -> "Kiremt (Jun-Sep)": the season alone, where the site is already named. */
export const seasonOnly = (name: string) => name.replace(/^[^·]+ · /, '')

/** "Kiremt (Jun-Sep)" -> "Kiremt"; "All Addis" stays whole. */
export const shortSeason = (group: string) => (isAllAddis(group) ? ALL_ADDIS : group.split(' ')[0])

/** "Kiremt (Jun-Sep)" -> "Kiremt"; "All Addis" -> "all seasons", for running text. */
export const seasonPhrase = (group: string) => (isAllAddis(group) ? 'all seasons' : shortSeason(group))

/** The season a calendar month falls in, under one calendar. */
/**
 * A filter's season under a calendar: by its sampling date where the calendar has day boundaries
 * (`from` on its seasons), otherwise by month. One rule for every chart and filter.
 */
export function seasonOfDate(date: string | null | undefined, month: number | null | undefined, seasons: readonly SeasonMeta[]): SeasonMeta | null {
  // a calendar computed from weather (`spans`) classifies only the dates it covers: a date outside them
  // has no season, never the month's usual one (the months are for month axes)
  if (isDatedCalendar(seasons)) {
    if (!date) return null
    const d = date.slice(0, 10)
    return seasons.find((s) => s.spans?.some(([a, b]) => d >= a && d < b)) ?? null
  }
  if (date && seasons.some((s) => s.from)) {
    const md = date.slice(5, 10)
    const byStart = seasons.filter((s) => s.from).sort((a, b) => a.from!.localeCompare(b.from!))
    // before the first start of the year, the year's last season carries on from December
    return [...byStart].reverse().find((s) => md >= s.from!) ?? byStart[byStart.length - 1] ?? null
  }
  return seasonOfMonth(month, seasons)
}

/** A calendar computed from weather (lib/rainSeasons, lib/dynamicSeasons): its seasons carry dated `spans`. */
export const isDatedCalendar = (seasons: readonly SeasonMeta[]) => seasons.some((s) => Array.isArray(s.spans))

/** The label for a filter a weather calendar cannot classify: no weather loaded, none in the record, or a gap. */
export const NO_WEATHER = 'No weather'

export const seasonOfMonth = (month: number | null | undefined, seasons: readonly SeasonMeta[]) =>
  month ? seasons.find((s) => s.months.includes(month)) ?? null : null
