import { useLoaded } from '@/lib/data'
import { loadDaily } from '@/lib/weather'
import { dynamicCalendar, type SiteRules } from '@/lib/dynamicSeasons'
import { collect, type WeatherCalendars } from '@/lib/weatherCalendars'

const NONE: WeatherCalendars = { cals: new Map(), failed: new Set() }

/**
 * Each site's dynamic seasons (lib/dynamicSeasons) from its daily ERA5 weather, for the sites
 * with a rule. Sites still loading are in neither `cals` nor `failed`; sites without daily weather
 * are in `failed`. Either way callers show the rule's names with no filter classified ("No weather").
 */
export function useDynamicCalendars(rules: SiteRules): WeatherCalendars {
  const key = JSON.stringify(Object.entries(rules).sort(([a], [b]) => a.localeCompare(b)))
  const entries = Object.entries(rules)
  // one object per load, so callers' memos see a change only when a load lands
  return useLoaded(key, () => entries.length
    ? Promise.all(entries.map(([code, rule]) => loadDaily(code).then((d) => [code, dynamicCalendar(d, rule)] as const).catch(() => code))).then(collect)
    : Promise.resolve(NONE), { keep: true }).data ?? NONE
}
