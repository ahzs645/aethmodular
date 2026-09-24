import type { MetaFile, SeasonMeta } from '@/lib/types'

/** Fixed, source-backed local month bins; see docs/site-seasonality.md. */
const LOCAL_SEASONS: Record<string, SeasonMeta[]> = {
  Beijing: [
    { name: 'Winter (Dec–Feb)', months: [12, 1, 2], color: '#6F86BA' },
    { name: 'Spring (Mar–May)', months: [3, 4, 5], color: '#67A96A' },
    { name: 'Summer (Jun–Aug)', months: [6, 7, 8], color: '#D99145' },
    { name: 'Autumn (Sep–Nov)', months: [9, 10, 11], color: '#AA6B78' },
  ],
  Delhi: [
    { name: 'Winter (Jan–Feb)', months: [1, 2], color: '#6F86BA' },
    { name: 'Pre-monsoon (Mar–May)', months: [3, 4, 5], color: '#D99145' },
    { name: 'Monsoon (Jun–Sep)', months: [6, 7, 8, 9], color: '#428EAF' },
    { name: 'Post-monsoon (Oct–Dec)', months: [10, 11, 12], color: '#AA6B78' },
  ],
  JPL: [
    { name: 'Wet window (Oct–Apr)', months: [10, 11, 12, 1, 2, 3, 4], color: '#428EAF' },
    { name: 'Dry window (May–Sep)', months: [5, 6, 7, 8, 9], color: '#D99145' },
  ],
}

export function localSeasonsFor(site: string | null, meta: MetaFile | undefined): SeasonMeta[] | null {
  if (!site || !meta) return null
  if (site === 'Addis Ababa') return meta.season_conventions.belg_feb ?? null
  return LOCAL_SEASONS[site] ?? null
}

export function calendarLabel(meta: MetaFile & { season_site?: string }): string {
  if (meta.season_convention !== 'local') return `shared Ethiopian ${meta.season_convention} month bins`
  return meta.season_site ? `${meta.season_site} local calendar` : "each site's local calendar"
}

/** Season label under the per-site calendar: site-qualified wherever several calendars are in play. */
export const qualifySeason = (site: string, name: string, qualify: boolean) => (qualify ? `${site} · ${name}` : name)

/**
 * The seasons that apply to one site. Under the shared Ethiopian bins that is
 * every season; under the per-site calendar only that site's own. Month-based
 * charts (bands, clock, heatmap strip) must look seasons up through this, not
 * `meta.seasons.find(month)`, which would return whichever site came first.
 */
export function seasonsForSite(meta: MetaFile, site: string | null | undefined): SeasonMeta[] {
  if (!meta.seasons.some((s) => s.site)) return meta.seasons
  return meta.seasons.filter((s) => s.site === site)
}

/** More than one site's calendar is in force, so one month axis cannot carry a single season shading. */
export const mixedCalendars = (meta: MetaFile) => new Set(meta.seasons.map((s) => s.site).filter(Boolean)).size > 1
