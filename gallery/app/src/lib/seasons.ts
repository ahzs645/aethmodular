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
  local: 'Per site · each site’s local calendar',
  dry_feb: 'Shared Ethiopian month bins · Feb in dry',
  belg_feb: 'Shared Ethiopian month bins · Feb in Belg',
  quarters: 'Shared month bins · Dec–Feb, Mar–May, Jun–Aug, Sep–Nov',
  custom: 'Custom bins · your own, shared by every site',
}

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

/** "Kiremt (Jun-Sep)" -> "Kiremt"; "All Addis" stays whole. */
export const shortSeason = (group: string) => (isAllAddis(group) ? ALL_ADDIS : group.split(' ')[0])

/** "Kiremt (Jun-Sep)" -> "Kiremt"; "All Addis" -> "all seasons", for running text. */
export const seasonPhrase = (group: string) => (isAllAddis(group) ? 'all seasons' : shortSeason(group))

/** The season a calendar month falls in, under one calendar. */
export const seasonOfMonth = (month: number | null | undefined, seasons: readonly SeasonMeta[]) =>
  month ? seasons.find((s) => s.months.includes(month)) ?? null : null
