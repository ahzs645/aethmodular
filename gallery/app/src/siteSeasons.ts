import type { MetaFile, SeasonMeta } from '@/lib/types'
import networkCalendars from './siteSeasons.network.json' with { type: 'json' }

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

/**
 * Every other SPARTAN site, plus the two Santiago IMPROVE sites: month bins from each
 * country's met service or a site paper (siteSeasons.network.json, sources and caveats
 * in docs/site-seasonality-network.md). Keyed by code and by the gallery's site name.
 */
interface RawSeason { name: string; months: number[] }
/** a second defensible calendar for a site (CODC: Bogotá or Palmira; Taiwan: review or outlook bins); the first is the default */
interface CalendarVariant { key: string; label: string; seasons: RawSeason[]; note?: string }
interface NetworkCalendar { site: string; seasons: RawSeason[]; confidence: string; note?: string; variants?: CalendarVariant[] }
// colour by what the season is, so "winter" reads the same at every site
const KEYWORD_COLORS: [RegExp, string][] = [
  [/post-monsoon|autumn|fall/i, '#AA6B78'],
  [/pre-monsoon|summer|hot|warm/i, '#D99145'],
  [/winter|cold|harmattan/i, '#6F86BA'],
  [/spring/i, '#67A96A'],
  [/monsoon|rain|wet/i, '#428EAF'],
  [/dry/i, '#B8860B'],
  [/transition/i, '#8A6FB0'],
]
const SPARE_COLORS = ['#6F86BA', '#67A96A', '#D99145', '#AA6B78', '#428EAF', '#B8860B', '#8A6FB0', '#5A9E9A']
function colourSeasons(seasons: { name: string; months: number[] }[]): SeasonMeta[] {
  const used = new Set<string>()
  return seasons.map((se) => {
    // two seasons of one kind at a site (Bujumbura's short and long dry) still need two colours
    let color = KEYWORD_COLORS.find(([re]) => re.test(se.name))?.[1]
    if (!color || used.has(color)) color = SPARE_COLORS.find((c) => !used.has(c)) ?? '#9aa5b1'
    used.add(color)
    return { name: se.name, months: se.months, color }
  })
}
const NETWORK_SEASONS = new Map<string, SeasonMeta[]>()
const NETWORK_VARIANTS = new Map<string, { key: string; label: string; note?: string; seasons: SeasonMeta[] }[]>()
for (const [code, cal] of Object.entries(networkCalendars as Record<string, NetworkCalendar>)) {
  const seasons = colourSeasons(cal.seasons)
  NETWORK_SEASONS.set(code, seasons)
  NETWORK_SEASONS.set(cal.site, seasons)
  if (cal.variants?.length) {
    const variants = cal.variants.map((v) => ({ key: v.key, label: v.label, note: v.note, seasons: colourSeasons(v.seasons) }))
    NETWORK_VARIANTS.set(code, variants)
    NETWORK_VARIANTS.set(cal.site, variants)
  }
}

/** The calendars a site can be binned by, when it has more than one (else null). */
export function calendarVariants(site: string, code?: string) {
  const v = NETWORK_VARIANTS.get(site) ?? (code ? NETWORK_VARIANTS.get(code) : undefined)
  return v ? v.map(({ key, label, note }) => ({ key, label, note })) : null
}

/**
 * A site's own calendar: the focus sites' (as before), then the network table by name or
 * code. `variant` picks one of a site's alternative calendars; unknown or absent = the default.
 */
export function localSeasonsFor(site: string | null, meta: MetaFile | undefined, code?: string, variant?: string): SeasonMeta[] | null {
  if (!site || !meta) return null
  if (site === 'Addis Ababa') return meta.season_conventions.belg_feb ?? null
  if (LOCAL_SEASONS[site]) return LOCAL_SEASONS[site]
  const alt = variant ? (NETWORK_VARIANTS.get(site) ?? (code ? NETWORK_VARIANTS.get(code) : undefined))?.find((v) => v.key === variant) : undefined
  return alt?.seasons ?? NETWORK_SEASONS.get(site) ?? (code ? NETWORK_SEASONS.get(code) : undefined) ?? null
}

/** Whether a site has a calendar of its own (otherwise the local view bins it by quarters). */
export const hasLocalCalendar = (site: string, code?: string) =>
  site === 'Addis Ababa' || site in LOCAL_SEASONS || NETWORK_SEASONS.has(site) || (!!code && NETWORK_SEASONS.has(code))

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
